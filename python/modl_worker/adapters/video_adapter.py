"""Video generation adapter — loads LTX-2 pipeline and generates video.

STATUS (2026-04-13): Loader scaffolding is in place but the backend is not
yet producing videos on this architecture. Known blockers:

  1. The community GGUF variants on HF (unsloth, QuantStack) are quantized
     from the ComfyUI-format LTX-2 checkpoint, whose key layout does not
     match diffusers' ``LTX2VideoTransformer3DModel``. Concretely the GGUF
     has keys like ``adaln_single.*``, ``audio_embeddings_connector.*``
     (connector baked into the transformer), while diffusers expects
     ``time_embed.*``, ``av_cross_attn_audio_scale_shift.*`` and treats the
     connector as a separate ``LTX2TextConnectors`` component. 982 keys in
     GGUF have no model counterpart, 724 model keys are missing from GGUF.

  2. Loading the bf16 diffusers-format transformer from ``dg845/LTX-2.3-
     Distilled-Diffusers`` works but requires ~45 GB download and more
     VRAM than the 24 GB target.

  3. The ComfyUI NVFP4 Gemma 3 text encoder file uses Nvidia NVFP4 block
     scales that diffusers' from_pretrained cannot consume directly;
     ``comfy_fp4.py`` dequantises to bf16 but the result is ~23 GB on disk
     and OOMs on load alongside the transformer.

Follow-up options (tracked in PR description):
  - Build a diffusers-format GGUF by quantising the dg845 safetensors with
    a custom converter so keys match ``LTX2VideoTransformer3DModel``.
  - Wire the existing ComfyUI REST API at :8188 as a video backend.
  - Wait for diffusers to ship native GGUF text-encoder loading + an
    LTX-2.3 GGUF conversion recipe.

What is currently wired up (so this file is ready for the working path):
  - Hybrid loader: local GGUF transformer + local fp4 text encoder +
    ``LTX2Pipeline.from_pretrained`` for audio_vae/connectors/vocoder/etc.
  - ComfyUI NVFP4 dequant helper in ``comfy_fp4.py``
  - txt2vid / img2vid mode plumbing, distilled-LoRA fusing, MP4 export
"""

import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter


def run_generate_video(config_path: Path, emitter: EventEmitter) -> int:
    """Run video generation from a GenerateJobSpec YAML file."""
    import yaml
    import torch

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "ltx-video-dev")
    base_model_path = model_info.get("base_model_path")
    lora_info = spec.get("lora")
    output_info = spec.get("output", {})
    params = spec.get("params", {})

    prompt = spec.get("prompt", "")
    width = params.get("width", 768)
    height = params.get("height", 512)
    num_frames = params.get("num_frames", 97)
    fps = params.get("fps", 24)
    steps = params.get("steps", 8)
    guidance = params.get("guidance", 1.0)
    seed = params.get("seed")
    count = params.get("count", 1)
    init_image_path = params.get("init_image")

    # Determine mode
    mode = "img2vid" if init_image_path else "txt2vid"

    # Load init image for img2vid
    init_image = None
    if init_image_path:
        from PIL import Image
        init_image = Image.open(init_image_path).convert("RGB")
        emitter.info(f"Init image loaded: {init_image.size[0]}x{init_image.size[1]}")

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    emitter.info(f"Loading LTX-2 video pipeline for {base_model_id} (mode={mode})...")
    emitter.progress(stage="load", step=0, total_steps=count)

    # --- Hybrid loading: local GGUF transformer + local fp4 text encoder + ---
    # --- HF downloads for the rest (audio_vae, connectors, vocoder, etc.). ---
    #
    # LTX2Pipeline requires 9 components (scheduler, vae, audio_vae, text_encoder,
    # tokenizer, connectors, transformer, vocoder, processor). We have the two big
    # ones locally; the smaller ones (audio_vae ~200MB, connectors ~3GB, vocoder ~?)
    # come from HF on first run.
    try:
        from modl_worker.adapters.arch_config import detect_arch
        arch = detect_arch(base_model_id, arch_key=model_info.get("arch_key"))
        is_distilled = arch == "ltx2_video_distilled"

        pipe = _load_ltx2_hybrid(
            base_model_id, base_model_path, is_distilled, mode, emitter,
        )

        # Video-specific memory optimization
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            emitter.info("VAE tiling enabled")
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            emitter.info("Model CPU offload enabled")

        # Apply distilled LoRA if we have a non-distilled base (19B Dev)
        if not is_distilled:
            _apply_distilled_lora(pipe, lora_info, emitter)

    except Exception as exc:
        emitter.error("PIPELINE_LOAD_FAILED", f"Failed to load video pipeline: {exc}", recoverable=False)
        import traceback
        traceback.print_exc()
        return 1

    emitter.job_started(config=str(config_path))

    # --- Generation loop ---
    generator = torch.Generator(device="cpu")
    if seed is None:
        seed = generator.seed()
    generator.manual_seed(seed)

    artifact_paths = []
    for i in range(count):
        t0 = time.time()

        def _step_callback(pipe_self, step_idx, timestep, callback_kwargs):
            emitter.progress(stage="step", step=step_idx + 1, total_steps=steps)
            return callback_kwargs

        try:
            gen_kwargs = dict(
                prompt=prompt,
                negative_prompt="blurry, low quality, still frame, watermark",
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=float(fps),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=_step_callback,
                output_type="pil",
            )
            if init_image is not None:
                gen_kwargs["image"] = init_image
            result = pipe(**gen_kwargs)
            video_frames = result.frames[0]
        except Exception as exc:
            emitter.error(
                "GENERATION_FAILED",
                f"Video generation failed: {exc}",
                recoverable=(i + 1 < count),
            )
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        # Export video
        from modl_worker.video_util import save_and_emit_video_artifact
        filepath = save_and_emit_video_artifact(
            video_frames, fps, output_dir, emitter,
            index=i, count=count,
            metadata={
                "generated_with": "modl.run",
                "prompt": prompt,
                "base_model_id": base_model_id,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "steps": steps,
                "guidance": guidance,
                "seed": seed + i if seed is not None else None,
            },
            elapsed=elapsed,
        )
        if filepath:
            artifact_paths.append(filepath)

        if seed is not None:
            generator.manual_seed(seed + i + 1)

    if artifact_paths:
        emitter.completed(f"Generated {len(artifact_paths)} video(s)")
    else:
        emitter.error("NO_OUTPUT_GENERATED", "All generation attempts failed", recoverable=False)
        return 1

    return 0


def _load_ltx2_hybrid(
    base_model_id: str,
    base_model_path: str | None,
    is_distilled: bool,
    mode: str,
    emitter,
):
    """Load LTX-2 pipeline with local transformer/text-encoder + HF for the rest.

    This is a hybrid: we provide the two large components from local store
    (GGUF transformer + dequantized fp4 text encoder) and let
    `from_pretrained()` download the smaller pieces (audio_vae, connectors,
    vocoder, scheduler, tokenizer, processor) from the HF repo on first run.
    """
    import time
    import torch
    from diffusers import (
        LTX2Pipeline,
        LTX2ImageToVideoPipeline,
        LTX2VideoTransformer3DModel,
        GGUFQuantizationConfig,
    )
    from transformers import Gemma3ForConditionalGeneration

    from modl_worker.adapters.arch_config import _get_installed_path
    from modl_worker.adapters.comfy_fp4 import (
        is_comfy_nvfp4,
        cached_dequant_path,
        dequant_comfy_nvfp4_to_file,
    )
    from modl_worker.adapters.pipeline_loader import _ensure_hf_layout

    PipelineClass = LTX2ImageToVideoPipeline if mode == "img2vid" else LTX2Pipeline

    CONFIGS = Path(__file__).parent.parent / "configs"
    dtype = torch.bfloat16

    # --- 1. Resolve transformer path + HF repo for remaining components ---
    transformer_path = base_model_path
    if not transformer_path:
        # Prefer 2.3 distilled, fall back to 19B Dev
        transformer_path = (
            _get_installed_path("ltx-video-2-3")
            or _get_installed_path("ltx-video-distilled")
            or _get_installed_path("ltx-video-dev")
        )
    if not transformer_path or not Path(transformer_path).exists():
        raise FileNotFoundError(
            "LTX Video transformer not found. Install with: modl pull ltx-video-2-3"
        )

    # Pick HF repo based on whether we're using 2.3 distilled or 19B Dev
    if is_distilled or "2.3" in Path(transformer_path).name.lower() or "2-3" in Path(transformer_path).name.lower():
        hf_repo = "dg845/LTX-2.3-Distilled-Diffusers"
        config_dir = CONFIGS / "ltx23-transformer"
    else:
        hf_repo = "Lightricks/LTX-2"
        config_dir = CONFIGS / "ltx2-transformer"

    emitter.info(f"Using LTX-2 base repo: {hf_repo}")

    # --- 2. Load GGUF transformer locally ---
    # For LTX-2.3 we need low_cpu_mem_usage=False + ignore_mismatched_sizes=True
    # to avoid meta-tensor copy errors with the new architecture. This loads
    # the full dequantized model into RAM (~22GB for 22B model) but is required
    # by the current diffusers LTX2 implementation.
    emitter.info(f"Loading GGUF transformer: {Path(transformer_path).name}")
    t0 = time.time()
    load_kwargs = dict(
        config=str(config_dir),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )
    is_23_transformer = (
        "2.3" in Path(transformer_path).name.lower()
        or "2-3" in Path(transformer_path).name.lower()
    )
    if is_23_transformer:
        load_kwargs["low_cpu_mem_usage"] = False
        load_kwargs["ignore_mismatched_sizes"] = True
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        transformer_path, **load_kwargs,
    )
    emitter.info(f"  → Transformer loaded in {time.time()-t0:.1f}s")

    # --- 3. Load text encoder from local fp4 cache, quantize to NF4 in RAM ---
    # The dequantized bf16 cache is 23GB; loading directly would OOM with the
    # GGUF transformer also resident. Apply bitsandbytes NF4 at load time →
    # ~8GB RAM/VRAM, fits alongside the transformer.
    text_encoder_path = _get_installed_path("ltx2-gemma3-12b")
    text_encoder = None
    if text_encoder_path and Path(text_encoder_path).exists() and is_comfy_nvfp4(text_encoder_path):
        cache_path = cached_dequant_path(text_encoder_path)
        if not cache_path.exists():
            emitter.info(f"  → Dequantizing fp4 text encoder (one-time, ~3min)...")
            dequant_comfy_nvfp4_to_file(text_encoder_path, str(cache_path), target_dtype=dtype)
        else:
            emitter.info(f"  → Using cached dequantized text encoder")
        te_layout = _ensure_hf_layout(str(cache_path), CONFIGS / "ltx2-text-encoder")

        # Quantize to NF4 at load time (8GB instead of 24GB)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        emitter.info(f"  → Loading text encoder with NF4 quantization...")
        t1 = time.time()
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            te_layout,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        emitter.info(f"  → Text encoder loaded in {time.time()-t1:.1f}s")
    else:
        emitter.info("  → No local fp4 text encoder, will download from HF")

    # --- 4. Load full pipeline from HF, override with local components ---
    emitter.info(f"Loading remaining components from {hf_repo} (audio_vae, connectors, vocoder, scheduler, tokenizer)...")
    t2 = time.time()
    overrides = {"transformer": transformer, "torch_dtype": dtype}
    if text_encoder is not None:
        overrides["text_encoder"] = text_encoder
    pipe = PipelineClass.from_pretrained(hf_repo, **overrides)
    emitter.info(f"  → Pipeline assembled in {time.time()-t2:.1f}s")

    return pipe


def _apply_distilled_lora(pipe, lora_info, emitter):
    """Apply distilled LoRA for 19B model (reduces steps from 30 to 8)."""
    from modl_worker.adapters.arch_config import _get_installed_path

    lora_path = None
    lora_strength = 0.6
    if lora_info:
        lora_path = lora_info.get("path")
        lora_strength = lora_info.get("weight", 0.6)
    if not lora_path:
        lora_path = _get_installed_path("ltx2-distilled-lora")
    if lora_path and Path(lora_path).exists():
        emitter.info(f"Applying distilled LoRA (strength={lora_strength})...")
        try:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_strength)
            emitter.info("Distilled LoRA applied")
        except Exception as exc:
            emitter.info(f"Warning: Failed to apply LoRA: {exc}")
