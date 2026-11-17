"""Qwen-Image-2.1 glue: prompt-embedding cache and RGBA output handling.

The pipeline itself is diffusers' native ``QwenImage21Pipeline``; this module
only adds what modl needs around it.
"""

from collections import OrderedDict

_EMBED_CACHE_SIZE = 8

# Opaque images still come back with a sprinkle of alpha in 250-254 (measured:
# 3-5% of pixels, <0.03% below 250). Transparent ones have most of the canvas
# at 0. Treat an image as transparent when more than this share of pixels has
# alpha < _ALPHA_CUTOFF.
_ALPHA_CUTOFF = 250
_TRANSPARENT_SHARE = 0.005


def prepare_qwen21_pipeline(pipe) -> None:
    """Wrap ``encode_prompt`` with a small LRU cache for text-only prompts.

    The 8B text encoder is parked on the CPU and swapped in per encode. A
    ``count > 1`` job, a seed sweep, or the CFG negative re-encode the same
    strings. Image-conditioned encodes (edits) see the pixels, so they are
    never cached.
    """
    if getattr(pipe, "_modl_embed_cache", None) is not None:
        return
    cache: OrderedDict = OrderedDict()
    original = pipe.encode_prompt

    def encode_prompt(prompt=None, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None, **kwargs):
        cacheable = (
            not image and prompt_embeds is None and prompt_embeds_mask is None and not kwargs
        )
        key = None
        if cacheable:
            key = (tuple([prompt] if isinstance(prompt, str) else prompt), str(device),
                   num_images_per_prompt)
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit
        result = original(prompt=prompt, image=image, device=device,
                          num_images_per_prompt=num_images_per_prompt,
                          prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
                          **kwargs)
        if key is not None:
            cache[key] = result
            while len(cache) > _EMBED_CACHE_SIZE:
                cache.popitem(last=False)
        return result

    pipe.encode_prompt = encode_prompt
    pipe._modl_embed_cache = cache


def finalize_rgba(image):
    """Keep RGBA only when the image really is transparent.

    Qwen-Image-2.1 always decodes four channels. For an ordinary prompt the
    alpha is ~opaque noise, so flatten over white, which is how the model was
    trained to see transparency. Real cut-outs ("This is an RGBA image with
    transparency...") keep their alpha.
    """
    if getattr(image, "mode", None) != "RGBA":
        return image
    import numpy as np
    from PIL import Image

    alpha = np.asarray(image.getchannel("A"))
    if (alpha < _ALPHA_CUTOFF).mean() > _TRANSPARENT_SHARE:
        return image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def is_qwen21_pipeline(pipeline) -> bool:
    return type(pipeline).__name__ == "QwenImage21Pipeline"
