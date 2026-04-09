"""Minimal test: Qwen-Image-Edit-2511 GGUF via diffusers.

Follows the official diffusers GGUF pattern:
  1. Load transformer via from_single_file + GGUFQuantizationConfig
  2. Assemble pipeline with from_pretrained components
  3. Run inference with official Qwen params

Usage:
  python scripts/test-qwen-edit-2511-gguf.py [--steps N] [--cfg N] [--lora PATH]
"""

import argparse
import time
import torch
from PIL import Image
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    GGUFQuantizationConfig,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, Qwen2VLProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default="on warm cream backdrop, soft studio lighting")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gguf", default="/home/pedro/modl/store/checkpoint/vantage-2511/Qwen-Image-Edit-2511-Q5_K_M.gguf")
    parser.add_argument("--lora", default=None, help="Optional LoRA safetensors path")
    parser.add_argument("--output", default="/tmp/qwen-edit-test.png")
    args = parser.parse_args()

    print(f"Loading GGUF transformer from {args.gguf}...")
    t0 = time.time()

    transformer = QwenImageTransformer2DModel.from_single_file(
        args.gguf,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config="python/modl_worker/configs/qwen-image-edit-transformer",
        torch_dtype=torch.bfloat16,
    )
    print(f"  Transformer loaded in {time.time()-t0:.1f}s")

    # Load other components from local configs/weights
    config_base = "python/modl_worker/configs"
    store = "/home/pedro/modl/store"

    print("Loading text encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{store}/text_encoder/7dc87a9c61db8168/hf_layout",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{config_base}/qwen-image-tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(f"{config_base}/qwen-image-processor")

    import safetensors.torch
    vae_config = AutoencoderKLQwenImage.load_config(f"{config_base}/qwen-image-vae")
    vae = AutoencoderKLQwenImage.from_config(vae_config)
    vae_sd = safetensors.torch.load_file(f"{store}/vae/a70580f0213e6796/qwen-image-vae.safetensors")
    vae.load_state_dict(vae_sd, strict=False)
    vae = vae.to(torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        f"{config_base}/qwen-image-edit-scheduler"
    )

    print("Assembling pipeline...")
    pipe = QwenImageEditPlusPipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        vae=vae,
        scheduler=scheduler,
    )

    # Optional LoRA
    if args.lora:
        import os
        print(f"Loading LoRA: {args.lora}")
        pipe.load_lora_weights(
            os.path.dirname(args.lora),
            weight_name=os.path.basename(args.lora),
            adapter_name="default",
        )
        pipe.set_adapters(["default"], adapter_weights=[1.0])

    pipe.enable_model_cpu_offload()
    print(f"Pipeline ready (total setup: {time.time()-t0:.1f}s)")

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Input: {args.image} ({image.size[0]}x{image.size[1]})")
    print(f"Params: steps={args.steps}, true_cfg={args.cfg}, seed={args.seed}")

    # Run inference — official Qwen params
    t1 = time.time()
    result = pipe(
        image=[image],
        prompt=args.prompt,
        true_cfg_scale=args.cfg,
        negative_prompt=" ",
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        generator=torch.manual_seed(args.seed),
    )

    output = result.images[0]
    output.save(args.output)
    print(f"Output: {args.output} ({output.size[0]}x{output.size[1]}) in {time.time()-t1:.1f}s")


if __name__ == "__main__":
    main()
