import argparse
import os
import sys
from pathlib import Path

from modl_worker.adapters import (
    run_train, run_generate, run_caption, run_resize, run_tag,
    run_score, run_detect, run_compare,
    run_segment, run_face_restore,
)
from modl_worker.protocol import EventEmitter, fatal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modl_worker")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training adapter")
    train.add_argument("--config", required=True, help="Path to training config/spec yaml")
    train.add_argument("--job-id", default="", help="Job ID for event envelope")

    gen = sub.add_parser("generate", help="Run inference/generation adapter")
    gen.add_argument("--config", required=True, help="Path to generate spec yaml")
    gen.add_argument("--job-id", default="", help="Job ID for event envelope")

    cap = sub.add_parser("caption", help="Run auto-captioning adapter")
    cap.add_argument("--config", required=True, help="Path to caption spec yaml")
    cap.add_argument("--job-id", default="", help="Job ID for event envelope")

    rsz = sub.add_parser("resize", help="Run batch image resize")
    rsz.add_argument("--config", required=True, help="Path to resize spec yaml")
    rsz.add_argument("--job-id", default="", help="Job ID for event envelope")

    tg = sub.add_parser("tag", help="Run auto-tagging adapter")
    tg.add_argument("--config", required=True, help="Path to tag spec yaml")
    tg.add_argument("--job-id", default="", help="Job ID for event envelope")

    sc = sub.add_parser("score", help="Run aesthetic scoring adapter")
    sc.add_argument("--config", required=True, help="Path to score spec yaml")
    sc.add_argument("--job-id", default="", help="Job ID for event envelope")

    det = sub.add_parser("detect", help="Run face detection adapter")
    det.add_argument("--config", required=True, help="Path to detect spec yaml")
    det.add_argument("--job-id", default="", help="Job ID for event envelope")

    cmp = sub.add_parser("compare", help="Run image comparison adapter")
    cmp.add_argument("--config", required=True, help="Path to compare spec yaml")
    cmp.add_argument("--job-id", default="", help="Job ID for event envelope")

    seg = sub.add_parser("segment", help="Run image segmentation adapter")
    seg.add_argument("--config", required=True, help="Path to segment spec yaml")
    seg.add_argument("--job-id", default="", help="Job ID for event envelope")

    fr = sub.add_parser("face-restore", help="Run face restoration adapter")
    fr.add_argument("--config", required=True, help="Path to face restore spec yaml")
    fr.add_argument("--job-id", default="", help="Job ID for event envelope")

    srv = sub.add_parser("serve", help="Start persistent worker daemon")
    srv.add_argument("--timeout", type=int, default=600, help="Idle timeout in seconds (default: 600)")
    srv.add_argument("--max-models", type=int, default=2, help="Max models to cache in VRAM (default: 2)")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    job_id = getattr(args, "job_id", "") or ""
    emitter = EventEmitter(source="modl_worker", job_id=job_id)

    if args.command == "train":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_train(config_path, emitter)

    if args.command == "generate":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_generate(config_path, emitter)

    if args.command == "caption":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_caption(config_path, emitter)

    if args.command == "resize":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_resize(config_path, emitter)

    if args.command == "tag":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_tag(config_path, emitter)

    if args.command == "score":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_score(config_path, emitter)

    if args.command == "detect":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_detect(config_path, emitter)

    if args.command == "compare":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_compare(config_path, emitter)

    if args.command == "segment":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_segment(config_path, emitter)

    if args.command == "face-restore":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_face_restore(config_path, emitter)

    if args.command == "serve":
        from modl_worker.serve import run_serve
        return run_serve(timeout=args.timeout, max_models=args.max_models)

    fatal(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
