"""Shared utilities for vision-language adapters (ground, describe, vl_tag).

Provides model loading with configurable model ID and HuggingFace repo routing.
"""

import torch
from modl_worker.protocol import EventEmitter

# Map modl model IDs to HuggingFace repos
VL_MODEL_REPOS = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    # Legacy aliases
    "qwen25-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}

DEFAULT_VL_MODEL = "qwen3-vl-2b"


def load_qwen_vl(emitter: EventEmitter, model_id: str | None = None):
    """Load a Qwen VL model and processor.

    Args:
        emitter: EventEmitter for progress logging
        model_id: Model ID (qwen3-vl-2b or qwen3-vl-8b). Defaults to 2B.

    Returns:
        (model, processor) tuple
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = model_id or DEFAULT_VL_MODEL
    repo = VL_MODEL_REPOS.get(model_id)

    if repo is None:
        # Allow passing a HuggingFace repo directly
        repo = model_id

    emitter.info(f"Loading {repo}...")

    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(repo)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    emitter.info(f"Loaded {repo} ({param_count:.1f}B params)")

    return model, processor


def run_vl_inference(model, processor, image_path: str, prompt: str, max_tokens: int = 1024) -> str:
    """Run a single VL inference: image + text prompt → text response.

    Args:
        model: Loaded Qwen2.5-VL model
        processor: Model processor
        image_path: Path to image file
        prompt: Text prompt
        max_tokens: Maximum output tokens

    Returns:
        Model response text (stripped)
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return response
