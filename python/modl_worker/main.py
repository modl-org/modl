import argparse
import os
import sys
from pathlib import Path

from modl_worker.adapters.registry import get_adapters
from modl_worker.protocol import EventEmitter, fatal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modl_worker")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, entry in get_adapters().items():
        p = sub.add_parser(name, help=entry.description)
        p.add_argument("--config", required=True, help=f"Path to {name} spec yaml")
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

    adapters = get_adapters()
    entry = adapters.get(args.command)
    if entry is None:
        fatal(f"Unsupported command: {args.command}")
        return 1

    job_id = getattr(args, "job_id", "") or ""
    emitter = EventEmitter(source="modl_worker", job_id=job_id)
    emitter.job_accepted(worker_pid=os.getpid())
    return entry.run_fn(Path(args.config), emitter)


if __name__ == "__main__":
    sys.exit(main())
