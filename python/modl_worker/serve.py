"""Persistent worker daemon — keeps models in VRAM between generation requests.

Listens on a Unix socket (default: ~/.modl/worker.sock). Clients send a JSON
request envelope, and the worker responds with the same JSONL event stream
used by one-shot mode. The model stays loaded in GPU memory across requests,
eliminating the 20-45s cold-start penalty.

Architecture:
    modl gen "prompt"
      → Rust CLI connects to Unix socket
      → sends {"action": "generate", "job_id": "...", "spec": {...}}
      → worker runs inference with cached pipeline
      → streams JSONL events back over the socket
      → connection closed, worker stays alive

    modl worker start   → spawns this as a background daemon
    modl worker stop    → sends {"action": "shutdown"}
    modl worker status  → sends {"action": "status"}
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modl_worker.protocol import EventEmitter


# ---------------------------------------------------------------------------
# Socket-aware EventEmitter — writes JSONL to a socket instead of stdout
# ---------------------------------------------------------------------------

class SocketEventEmitter(EventEmitter):
    """EventEmitter that writes JSONL events to a socket connection."""

    def __init__(self, conn: socket.socket, source: str = "modl_worker", job_id: str = "") -> None:
        super().__init__(source=source, job_id=job_id)
        self._conn = conn

    def emit(self, event: dict[str, Any]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "v1",
            "job_id": self.job_id,
            "sequence": self.sequence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": self.source,
            "event": event,
        }
        line = json.dumps(payload) + "\n"
        try:
            self._conn.sendall(line.encode())
        except (BrokenPipeError, OSError):
            pass  # Client disconnected — non-fatal


# ---------------------------------------------------------------------------
# Model Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheKey:
    model_id: str
    dtype: str  # "bfloat16", "float16"
    mode: str = "txt2img"  # "txt2img", "img2img", "inpaint"

    def __hash__(self) -> int:
        return hash((self.model_id, self.dtype, self.mode))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return NotImplemented
        return self.model_id == other.model_id and self.dtype == other.dtype and self.mode == other.mode


@dataclass
class CachedPipeline:
    pipeline: Any  # diffusers Pipeline object
    cls_name: str  # e.g. "FluxPipeline"
    loaded_at: float
    last_used: float
    vram_estimate_mb: int
    lora_id: str | None = None
    lora_weight: float = 0.0


class ModelCache:
    """LRU cache for loaded diffusers pipelines.

    Keeps up to `max_models` pipelines in GPU memory. When capacity is
    exceeded, the least-recently-used pipeline is evicted.
    """

    MAX_UTILITY_MODELS = 6  # cap to prevent unbounded VRAM growth

    def __init__(self, max_models: int = 2) -> None:
        self._cache: dict[CacheKey, CachedPipeline] = {}
        self._utility_cache: dict[str, Any] = {}
        self._utility_order: list[str] = []  # insertion order for LRU eviction
        self._max_models = max_models
        self._lock = threading.Lock()

    def _detect_mode(self, spec: dict) -> str:
        """Detect generation mode from spec params."""
        params = spec.get("params", {})
        if params.get("mask") and params.get("init_image"):
            return "inpaint"
        elif params.get("init_image"):
            return "img2img"
        return "txt2img"

    def get_or_load(self, spec: dict, emitter: EventEmitter) -> tuple[Any, str]:
        """Return (pipeline, cls_name) — cached or freshly loaded.

        Also handles LoRA reconciliation (hot-swap if base model matches
        but LoRA changed). Supports mode switching (txt2img -> img2img ->
        inpaint) via from_pipe(), caching each mode variant.
        """
        from modl_worker.adapters.pipeline_loader import (
            _resolve_pipeline_class,
            _get_pipeline,
            load_pipeline,
        )
        from modl_worker.adapters.arch_config import (
            resolve_pipeline_class_for_mode,
        )

        model_info = spec.get("model", {})
        base_model_id = model_info.get("base_model_id", "flux-schnell")
        base_model_path = model_info.get("base_model_path")
        mode = self._detect_mode(spec)

        key = CacheKey(model_id=base_model_id, dtype="bfloat16", mode=mode)
        base_key = CacheKey(model_id=base_model_id, dtype="bfloat16", mode="txt2img")

        with self._lock:
            # Check for exact mode match in cache
            if key in self._cache:
                cached = self._cache[key]
                cached.last_used = time.time()
                emitter.info(f"Model cache HIT: {base_model_id} mode={mode} (loaded {self._ago(cached.loaded_at)})")
                self._reconcile_lora(cached, spec, emitter)
                return cached.pipeline, cached.cls_name

            # If we need img2img/inpaint but have the base txt2img cached,
            # use from_pipe() to create the mode variant (shares weights, ~0ms)
            if mode != "txt2img" and base_key in self._cache:
                base_cached = self._cache[base_key]
                base_cached.last_used = time.time()
                target_cls_name = resolve_pipeline_class_for_mode(base_model_id, mode)
                emitter.info(f"Mode switch via from_pipe(): {base_cached.cls_name} -> {target_cls_name}")
                TargetClass = _get_pipeline(target_cls_name)
                mode_pipe = TargetClass.from_pipe(base_cached.pipeline)
                now = time.time()
                mode_cached = CachedPipeline(
                    pipeline=mode_pipe,
                    cls_name=target_cls_name,
                    loaded_at=now,
                    last_used=now,
                    vram_estimate_mb=0,  # shares weights with base
                    lora_id=base_cached.lora_id,
                    lora_weight=base_cached.lora_weight,
                )
                self._cache[key] = mode_cached
                self._reconcile_lora(mode_cached, spec, emitter)
                return mode_pipe, target_cls_name

            # Cache miss — evict LRU if at capacity
            if len(self._cache) >= self._max_models:
                self._evict_lru(emitter)

            # VRAM-aware eviction: loading a fresh model while a large one is
            # already resident OOMs even within the max_models count limit.
            # Evict LRU models until there's headroom before attempting load.
            self._ensure_vram_headroom(emitter)

            # Load fresh pipeline
            emitter.info(f"Model cache MISS: loading {base_model_id}...")
            cls_name = _resolve_pipeline_class(base_model_id)

            # Check if the base pipeline IS a native inpaint pipeline
            # (e.g. FluxFillPipeline). These don't have a separate txt2img
            # base — store directly under the requested mode key.
            is_native_inpaint = "Fill" in cls_name

            if is_native_inpaint:
                pipe = self._load_with_oom_recovery(
                    base_model_id, base_model_path, cls_name, emitter
                )
                now = time.time()
                cached = CachedPipeline(
                    pipeline=pipe,
                    cls_name=cls_name,
                    loaded_at=now,
                    last_used=now,
                    vram_estimate_mb=self._estimate_vram(pipe),
                )
                self._cache[key] = cached
                self._apply_lora(cached, spec, emitter)
                return pipe, cls_name

            # Standard model: load txt2img base first, then switch if needed
            pipe = self._load_with_oom_recovery(
                base_model_id, base_model_path, cls_name, emitter
            )
            now = time.time()

            cached = CachedPipeline(
                pipeline=pipe,
                cls_name=cls_name,
                loaded_at=now,
                last_used=now,
                vram_estimate_mb=self._estimate_vram(pipe),
            )
            self._cache[base_key] = cached

            # Handle LoRA for freshly-loaded pipeline
            self._apply_lora(cached, spec, emitter)

            # If we need a non-txt2img mode, switch via from_pipe()
            if mode != "txt2img":
                target_cls_name = resolve_pipeline_class_for_mode(base_model_id, mode)
                emitter.info(f"Mode switch via from_pipe(): {cls_name} -> {target_cls_name}")
                TargetClass = _get_pipeline(target_cls_name)
                mode_pipe = TargetClass.from_pipe(pipe)
                mode_cached = CachedPipeline(
                    pipeline=mode_pipe,
                    cls_name=target_cls_name,
                    loaded_at=now,
                    last_used=now,
                    vram_estimate_mb=0,
                    lora_id=cached.lora_id,
                    lora_weight=cached.lora_weight,
                )
                self._cache[key] = mode_cached
                return mode_pipe, target_cls_name

            return pipe, cls_name

    def get_utility(self, key: str) -> Any | None:
        """Get a cached utility model, or None."""
        val = self._utility_cache.get(key)
        if val is not None:
            # Move to end (most recently used)
            if key in self._utility_order:
                self._utility_order.remove(key)
                self._utility_order.append(key)
        return val

    def set_utility(self, key: str, model: Any) -> None:
        """Cache a utility model, evicting LRU entries if over capacity."""
        import gc
        import torch

        if key not in self._utility_cache:
            # Evict oldest entries if at capacity
            while len(self._utility_cache) >= self.MAX_UTILITY_MODELS and self._utility_order:
                lru_key = self._utility_order.pop(0)
                self._utility_cache.pop(lru_key, None)
            self._utility_order.append(key)
        self._utility_cache[key] = model

    def status(self) -> dict:
        """Return status info about loaded models."""
        with self._lock:
            models = []
            for key, cached in self._cache.items():
                models.append({
                    "model_id": key.model_id,
                    "dtype": key.dtype,
                    "mode": key.mode,
                    "loaded_at": cached.loaded_at,
                    "last_used": cached.last_used,
                    "vram_estimate_mb": cached.vram_estimate_mb,
                    "lora_id": cached.lora_id,
                    "lora_weight": cached.lora_weight,
                    "cls_name": cached.cls_name,
                })
            return {
                "models_loaded": len(self._cache),
                "max_models": self._max_models,
                "models": models,
                "utility_models": list(self._utility_cache.keys()),
            }

    def evict_all(self, emitter: EventEmitter | None = None) -> None:
        """Evict all cached models (diffusion + utility) and free VRAM."""
        import gc
        import torch
        with self._lock:
            for key in list(self._cache.keys()):
                if emitter:
                    emitter.info(f"Evicting model: {key.model_id}")
                del self._cache[key]
            if self._utility_cache:
                if emitter:
                    emitter.info(f"Evicting {len(self._utility_cache)} utility model(s)")
                self._utility_cache.clear()
                self._utility_order.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def _reconcile_lora(self, cached: CachedPipeline, spec: dict, emitter: EventEmitter) -> None:
        """Hot-swap LoRA if the base model matches but LoRA changed."""
        lora_info = spec.get("lora")
        requested_lora = lora_info.get("path") if lora_info else None
        requested_weight = lora_info.get("weight", 1.0) if lora_info else 0.0

        if cached.lora_id == requested_lora and cached.lora_weight == requested_weight:
            return  # Already the right LoRA at the right weight

        # Unfuse current LoRA
        if cached.lora_id:
            try:
                cached.pipeline.unfuse_lora()
                cached.pipeline.unload_lora_weights()
                emitter.info(f"Unloaded LoRA: {cached.lora_id}")
            except Exception as exc:
                emitter.warning("LORA_UNFUSE_WARN", f"LoRA unfuse warning: {exc}")
            cached.lora_id = None
            cached.lora_weight = 0.0

        # Apply new LoRA
        self._apply_lora(cached, spec, emitter)

    def _apply_lora(self, cached: CachedPipeline, spec: dict, emitter: EventEmitter) -> None:
        """Load and fuse a LoRA onto a cached pipeline."""
        from modl_worker.adapters.lora_utils import apply_lora_from_spec
        lora_info = spec.get("lora")
        if apply_lora_from_spec(cached.pipeline, spec, emitter):
            cached.lora_id = lora_info.get("path") if lora_info else None
            cached.lora_weight = lora_info.get("weight", 1.0) if lora_info else 0.0

    def _evict_lru(self, emitter: EventEmitter) -> None:
        """Remove the least-recently-used model from the cache."""
        import gc
        import torch
        if not self._cache:
            return

        lru_key = min(self._cache, key=lambda k: self._cache[k].last_used)
        emitter.info(f"Evicting LRU model: {lru_key.model_id}")
        del self._cache[lru_key]
        gc.collect()
        torch.cuda.empty_cache()

    # Minimum free VRAM (MB) we want before attempting to load a new model.
    # Diffusion pipelines need a few GB of headroom for activations on top
    # of weights; 4 GB covers small/quantized models. Bigger models still
    # get a second chance via _load_with_oom_recovery.
    _VRAM_HEADROOM_MB = 4096

    def _free_vram_mb(self) -> int | None:
        """Return currently free VRAM in MB, or None when unprobeable
        (CPU-only host, CUDA driver missing, nvml failure). None disables
        the pre-flight check so CPU/MPS paths still work."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        except Exception:
            return None

    def _ensure_vram_headroom(self, emitter: EventEmitter) -> None:
        """Evict LRU diffusion models until free VRAM >= headroom or cache empty.

        Called before loading a fresh pipeline so a big model sitting in
        VRAM doesn't cause the new load to OOM inside diffusers (where the
        error is harder to recover from cleanly)."""
        free_mb = self._free_vram_mb()
        if free_mb is None:
            return
        while self._cache and free_mb < self._VRAM_HEADROOM_MB:
            emitter.info(
                f"Low VRAM ({free_mb} MB free, need {self._VRAM_HEADROOM_MB} MB) — evicting LRU"
            )
            self._evict_lru(emitter)
            free_mb = self._free_vram_mb()
            if free_mb is None:
                return

    def _load_with_oom_recovery(
        self,
        base_model_id: str,
        base_model_path: Any,
        cls_name: str,
        emitter: EventEmitter,
    ) -> Any:
        """Load a pipeline; on CUDA OOM, evict everything and retry once.

        Covers the case where _ensure_vram_headroom's static threshold
        wasn't enough for a particularly large model, by falling back to
        a cold cache before giving up."""
        import gc
        import torch
        from modl_worker.adapters.pipeline_loader import load_pipeline

        try:
            return load_pipeline(base_model_id, base_model_path, cls_name, emitter)
        except torch.cuda.OutOfMemoryError as exc:
            if not self._cache:
                # Nothing left to evict — propagate with a clearer message.
                raise RuntimeError(
                    f"CUDA out of memory loading {base_model_id}: {exc}. "
                    f"Try a smaller variant (fp8/GGUF) or run with --no-worker."
                ) from exc
            emitter.warning(
                "VRAM_RETRY",
                f"CUDA OOM loading {base_model_id} — evicting all cached models and retrying.",
            )
            self.evict_all(emitter)
            gc.collect()
            torch.cuda.empty_cache()
            try:
                return load_pipeline(base_model_id, base_model_path, cls_name, emitter)
            except torch.cuda.OutOfMemoryError as exc2:
                raise RuntimeError(
                    f"CUDA out of memory loading {base_model_id} even after evicting "
                    f"cached models: {exc2}. Try a smaller variant (fp8/GGUF) or a "
                    f"larger GPU."
                ) from exc2

    @staticmethod
    def _estimate_vram(pipeline: Any) -> int:
        """Rough VRAM estimate in MB for a loaded pipeline."""
        try:
            import torch
            mem = torch.cuda.memory_allocated() / (1024 * 1024)
            return int(mem)
        except Exception:
            return 0

    @staticmethod
    def _ago(timestamp: float) -> str:
        """Human-readable 'X ago' string."""
        delta = time.time() - timestamp
        if delta < 60:
            return f"{int(delta)}s ago"
        elif delta < 3600:
            return f"{int(delta / 60)}m ago"
        else:
            return f"{delta / 3600:.1f}h ago"


# ---------------------------------------------------------------------------
# Worker Daemon
# ---------------------------------------------------------------------------

class WorkerDaemon:
    """Persistent worker process that listens on a Unix socket.

    Handles generation requests using a ModelCache so that the diffusers
    pipeline stays loaded in GPU memory across requests.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        pid_path: str | None = None,
        idle_timeout: int = 600,
        max_models: int = 2,
    ) -> None:
        modl_dir = Path.home() / ".modl"
        modl_dir.mkdir(parents=True, exist_ok=True)

        self.socket_path = Path(socket_path or str(modl_dir / "worker.sock"))
        self.pid_path = Path(pid_path or str(modl_dir / "worker.pid"))
        self.idle_timeout = idle_timeout
        self.cache = ModelCache(max_models=max_models)
        self._running = True
        self._last_activity = time.time()
        self._start_time = time.time()
        self._jobs_served = 0

    def run(self) -> None:
        """Start the worker daemon — blocks until shutdown."""
        # Redirect stderr to a log file so that writes from diffusers/torch
        # (tqdm progress bars, warnings, etc.) never hit a broken pipe when
        # the parent process has exited and closed its end of the stderr pipe.
        self._redirect_stderr()

        self._setup_signals()
        self._write_pid()
        self._cleanup_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(5)
        server.settimeout(1.0)  # Check idle timeout every second

        print(f"modl worker: listening on {self.socket_path}", file=sys.stderr)
        print(f"modl worker: idle timeout {self.idle_timeout}s, max models {self.cache._max_models}", file=sys.stderr)

        try:
            while self._running:
                # Check idle timeout
                idle = time.time() - self._last_activity
                if idle > self.idle_timeout:
                    print(f"modl worker: idle timeout ({self.idle_timeout}s) — shutting down", file=sys.stderr)
                    break

                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue

                # Handle connection in the main thread (single job at a time)
                try:
                    self._handle_connection(conn)
                except Exception as exc:
                    print(f"modl worker: connection error: {exc}", file=sys.stderr)
                finally:
                    conn.close()
                    self._last_activity = time.time()
        finally:
            self._shutdown(server)

    def _handle_connection(self, conn: socket.socket) -> None:
        """Read a request from the socket, dispatch it, write response events."""
        data = self._read_request(conn)
        if not data:
            return

        try:
            request = json.loads(data)
        except json.JSONDecodeError as exc:
            self._send_error(conn, "REQUEST_PARSE_ERROR", f"Invalid JSON: {exc}")
            return

        action = request.get("action", "")

        if action == "generate":
            self._handle_generate(conn, request)
        elif action == "edit":
            self._handle_edit(conn, request)
        elif action == "status":
            self._handle_status(conn)
        elif action == "shutdown":
            self._send_ok(conn, "shutting down")
            self._running = False
        elif action == "evict":
            model_id = request.get("model_id")
            if model_id:
                # Evict specific model
                emitter = SocketEventEmitter(conn, job_id="control")
                self.cache.evict_all(emitter)  # TODO: evict specific model
            self._send_ok(conn, "evicted")
        elif self._is_analysis_action(action):
            self._handle_analysis(conn, request, action)
        elif action == "ping":
            self._send_ok(conn, "pong")
        else:
            self._send_error(conn, "UNKNOWN_ACTION", f"Unknown action: {action}")

    def _handle_generate(self, conn: socket.socket, request: dict) -> None:
        """Run generation with a cached pipeline."""
        job_id = request.get("job_id", "gen-worker")
        spec = request.get("spec", {})

        emitter = SocketEventEmitter(conn, job_id=job_id)
        emitter.job_accepted(worker_pid=os.getpid())

        try:
            # Get or load pipeline from cache
            pipeline, cls_name = self.cache.get_or_load(spec, emitter)

            # Run inference using the gen_adapter's core logic
            from modl_worker.adapters.gen_adapter import run_generate_with_pipeline
            exit_code = run_generate_with_pipeline(spec, emitter, pipeline, cls_name)

            self._jobs_served += 1
            if exit_code != 0:
                emitter.info(f"Generation finished with exit code {exit_code}")

        except Exception as exc:
            emitter.error(
                "WORKER_GENERATE_ERROR",
                f"Generation failed: {exc}",
                recoverable=False,
            )

    def _handle_edit(self, conn: socket.socket, request: dict) -> None:
        """Run image editing with a cached pipeline."""
        job_id = request.get("job_id", "edit-worker")
        spec = request.get("spec", {})

        emitter = SocketEventEmitter(conn, job_id=job_id)
        emitter.job_accepted(worker_pid=os.getpid())

        try:
            # Get or load pipeline from cache (edit uses its own pipeline class)
            pipeline, cls_name = self.cache.get_or_load(spec, emitter)

            from modl_worker.adapters.edit_adapter import run_edit_with_pipeline
            exit_code = run_edit_with_pipeline(spec, emitter, pipeline)

            self._jobs_served += 1
            if exit_code != 0:
                emitter.info(f"Edit finished with exit code {exit_code}")

        except Exception as exc:
            emitter.error(
                "WORKER_EDIT_ERROR",
                f"Edit failed: {exc}",
                recoverable=False,
            )

    @staticmethod
    def _is_analysis_action(action: str) -> bool:
        """Check if an action is a daemon-capable analysis adapter."""
        from modl_worker.adapters.registry import get_adapters
        entry = get_adapters().get(action)
        return entry is not None and entry.daemon and entry.daemon_handler == "analysis"

    def _handle_analysis(self, conn: socket.socket, request: dict, action: str) -> None:
        """Run an analysis command with cached utility models."""
        job_id = request.get("job_id", f"{action}-worker")
        spec = request.get("spec", {})

        emitter = SocketEventEmitter(conn, job_id=job_id)
        emitter.job_accepted(worker_pid=os.getpid())

        config_path = None
        try:
            # Write spec to temp file (adapters expect a config path)
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(spec, f)
                config_path = Path(f.name)

            # Dispatch to the right adapter via the central registry
            from modl_worker.adapters.registry import get_adapters
            adapters = get_adapters()
            entry = adapters.get(action)
            if entry is None or not entry.daemon or entry.daemon_handler != "analysis":
                raise ValueError(f"No daemon-capable analysis adapter for: {action}")

            adapter_fn = entry.run_fn
            exit_code = adapter_fn(config_path, emitter, model_cache=self.cache._utility_cache)

            self._jobs_served += 1

        except Exception as exc:
            emitter.error(
                "WORKER_ANALYSIS_ERROR",
                f"{action} failed: {exc}",
                recoverable=False,
            )
        finally:
            if config_path is not None:
                config_path.unlink(missing_ok=True)

    def _handle_status(self, conn: socket.socket) -> None:
        """Return worker status as JSON."""
        cache_status = self.cache.status()
        status = {
            "action": "status_response",
            "pid": os.getpid(),
            "uptime_seconds": int(time.time() - self._start_time),
            "idle_timeout": self.idle_timeout,
            "idle_seconds": int(time.time() - self._last_activity),
            "jobs_served": self._jobs_served,
            "cache": cache_status,
        }
        line = json.dumps(status) + "\n"
        try:
            conn.sendall(line.encode())
        except (BrokenPipeError, OSError):
            pass

    def _read_request(self, conn: socket.socket) -> str | None:
        """Read a full JSON request from the socket (newline-delimited)."""
        conn.settimeout(30.0)  # 30s to receive the request
        buf = b""
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
        except socket.timeout:
            self._send_error(conn, "REQUEST_TIMEOUT", "Timed out waiting for request")
            return None

        return buf.decode().strip() if buf else None

    def _send_error(self, conn: socket.socket, code: str, message: str) -> None:
        """Send an error response and close."""
        payload = {
            "schema_version": "v1",
            "job_id": "",
            "sequence": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "modl_worker",
            "event": {"type": "error", "code": code, "message": message, "recoverable": False},
        }
        line = json.dumps(payload) + "\n"
        try:
            conn.sendall(line.encode())
        except (BrokenPipeError, OSError):
            pass

    def _send_ok(self, conn: socket.socket, message: str) -> None:
        """Send a simple OK response."""
        payload = {"action": "ok", "message": message}
        line = json.dumps(payload) + "\n"
        try:
            conn.sendall(line.encode())
        except (BrokenPipeError, OSError):
            pass

    def _redirect_stderr(self) -> None:
        """Redirect stderr to a log file.

        When the worker daemon is spawned as a background process, the parent
        may close its end of the stderr pipe after startup.  Any subsequent
        write to stderr (tqdm, torch warnings, diffusers logging) would then
        raise BrokenPipeError and kill the generation job.  Redirecting to a
        file avoids this entirely and preserves the output for debugging.
        """
        log_path = Path.home() / ".modl" / "worker.log"
        try:
            self._log_file = open(log_path, "a")  # noqa: SIM115
            sys.stderr = self._log_file
        except OSError:
            self._log_file = None  # If we can't redirect, continue anyway

    def _setup_signals(self) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        def handler(signum: int, frame: Any) -> None:
            print(f"\nmodl worker: received signal {signum}, shutting down...", file=sys.stderr)
            self._running = False

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _write_pid(self) -> None:
        """Write PID file."""
        self.pid_path.write_text(str(os.getpid()))

    def _cleanup_socket(self) -> None:
        """Remove stale socket file if it exists."""
        if self.socket_path.exists():
            self.socket_path.unlink()

    def _shutdown(self, server: socket.socket) -> None:
        """Clean shutdown: close socket, evict models, remove files."""
        print("modl worker: shutting down...", file=sys.stderr)
        server.close()
        self.cache.evict_all()
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_path.exists():
            self.pid_path.unlink()
        print(f"modl worker: served {self._jobs_served} job(s), goodbye", file=sys.stderr)
        if getattr(self, "_log_file", None) is not None:
            self._log_file.close()


# ---------------------------------------------------------------------------
# Entry point (called from main.py serve subcommand)
# ---------------------------------------------------------------------------

def run_serve(timeout: int = 600, max_models: int = 2) -> int:
    """Start the persistent worker daemon. Blocks until shutdown."""
    daemon = WorkerDaemon(idle_timeout=timeout, max_models=max_models)
    try:
        daemon.run()
    except Exception as exc:
        print(f"modl worker: fatal error: {exc}", file=sys.stderr)
        return 1
    return 0
