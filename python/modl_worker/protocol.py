import json
import sys
import time
from typing import Any, Dict


class EventEmitter:
    def __init__(self, source: str = "modl_worker", job_id: str | None = None) -> None:
        self.source = source
        self.job_id = job_id or ""
        self.sequence = 0

    def emit(self, event: Dict[str, Any]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "v1",
            "job_id": self.job_id,
            "sequence": self.sequence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": self.source,
            "event": event,
        }
        print(json.dumps(payload), flush=True)

    def info(self, message: str) -> None:
        self.emit({"type": "log", "level": "info", "message": message})

    def warning(self, code: str, message: str) -> None:
        self.emit({"type": "warning", "code": code, "message": message})

    def error(self, code: str, message: str, recoverable: bool = False, details: Dict[str, Any] | None = None) -> None:
        event: Dict[str, Any] = {
            "type": "error",
            "code": code,
            "message": message,
            "recoverable": recoverable,
        }
        if details:
            event["details"] = details
        self.emit(event)

    def job_accepted(self, worker_pid: int | None = None) -> None:
        event: Dict[str, Any] = {"type": "job_accepted"}
        if worker_pid is not None:
            event["worker_pid"] = worker_pid
        self.emit(event)

    def job_started(self, config: str | None = None, command: list[str] | None = None) -> None:
        event: Dict[str, Any] = {"type": "job_started"}
        if config is not None:
            event["config"] = config
        if command is not None:
            event["command"] = command
        self.emit(event)

    def progress(self, stage: str, step: int, total_steps: int, loss: float | None = None, eta_seconds: float | None = None) -> None:
        event: Dict[str, Any] = {
            "type": "progress",
            "stage": stage,
            "step": step,
            "total_steps": total_steps,
        }
        if loss is not None:
            event["loss"] = loss
        if eta_seconds is not None:
            event["eta_seconds"] = eta_seconds
        self.emit(event)

    def artifact(self, path: str, sha256: str | None = None, size_bytes: int | None = None) -> None:
        event: Dict[str, Any] = {"type": "artifact", "path": path}
        if sha256 is not None:
            event["sha256"] = sha256
        if size_bytes is not None:
            event["size_bytes"] = size_bytes
        self.emit(event)

    def completed(self, message: str = "Training completed") -> None:
        self.emit({"type": "completed", "message": message})


def fatal(message: str, exit_code: int = 1) -> None:
    print(
        json.dumps(
            {
                "schema_version": "v1",
                "sequence": 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "modl_worker",
                "event": {
                    "type": "error",
                    "code": "WORKER_INTERNAL_ERROR",
                    "message": message,
                    "recoverable": False,
                },
            }
        ),
        flush=True,
    )
    sys.exit(exit_code)
