import argparse
import os
import sys
from pathlib import Path

from modl_worker.adapters import (
    run_train, run_generate, run_edit, run_caption, run_resize, run_tag,
    run_score, run_detect, run_compare,
    run_segment, run_face_restore, run_upscale, run_remove_bg,
    run_face_crop, run_ground, run_describe, run_vl_tag,
    run_preprocess, run_lanpaint,
)
from modl_worker.protocol import EventEmitter, fatal


# All commands that take --config + --job-id and dispatch to a run_* function
_CONFIG_COMMANDS: dict[str, tuple[callable, str]] = {
    "train":        (run_train,        "Run training adapter"),
    "generate":     (run_generate,     "Run inference/generation adapter"),
    "edit":         (run_edit,         "Run image editing adapter"),
    "caption":      (run_caption,      "Run auto-captioning adapter"),
    "resize":       (run_resize,       "Run batch image resize"),
    "tag":          (run_tag,          "Run auto-tagging adapter"),
    "score":        (run_score,        "Run aesthetic scoring adapter"),
    "detect":       (run_detect,       "Run face detection adapter"),
    "compare":      (run_compare,      "Run image comparison adapter"),
    "segment":      (run_segment,      "Run image segmentation adapter"),
    "face-restore": (run_face_restore, "Run face restoration adapter"),
    "upscale":      (run_upscale,      "Run image upscaling adapter"),
    "remove-bg":    (run_remove_bg,    "Run background removal adapter"),
    "face-crop":    (run_face_crop,    "Detect faces and create close-up crops"),
    "ground":       (run_ground,       "Run text-grounded object detection"),
    "describe":     (run_describe,     "Run image captioning/description"),
    "vl-tag":       (run_vl_tag,       "Run VL-based image tagging"),
    "preprocess":   (run_preprocess,   "Run control image preprocessing"),
    "lanpaint":     (run_lanpaint,     "Run LanPaint training-free inpainting"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modl_worker")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd_name, (_fn, help_text) in _CONFIG_COMMANDS.items():
        p = sub.add_parser(cmd_name, help=help_text)
        p.add_argument("--config", required=True, help=f"Path to {cmd_name} spec yaml")
        p.add_argument("--job-id", default="", help="Job ID for event envelope")

    srv = sub.add_parser("serve", help="Start persistent worker daemon")
    srv.add_argument("--timeout", type=int, default=600, help="Idle timeout in seconds (default: 600)")
    srv.add_argument("--max-models", type=int,
                     default=int(os.environ.get("MODL_MAX_MODELS", "2")),
                     help="Max models to cache in VRAM (default: 2, env: MODL_MAX_MODELS)")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        from modl_worker.serve import run_serve
        return run_serve(timeout=args.timeout, max_models=args.max_models)

    fn = _CONFIG_COMMANDS.get(args.command)
    if fn is None:
        fatal(f"Unsupported command: {args.command}")
        return 1

    job_id = getattr(args, "job_id", "") or ""
    emitter = EventEmitter(source="modl_worker", job_id=job_id)
    emitter.job_accepted(worker_pid=os.getpid())
    return fn[0](Path(args.config), emitter)


if __name__ == "__main__":
    sys.exit(main())
