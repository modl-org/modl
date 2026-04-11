"""Adapter interface protocols.

These ``typing.Protocol`` classes document the two calling conventions
used by modl adapters.  They are *not* enforced at import time — existing
adapters work without inheriting from them.  Their purpose is:

  1. Make the contract explicit for anyone writing a new adapter.
  2. Enable optional ``isinstance`` checks at registration time
     (via ``@runtime_checkable``).

One-shot adapters
-----------------
Signature: ``(config_path: Path, emitter: EventEmitter) -> int``

Receives a YAML spec file and an emitter.  Loads models from scratch,
runs the operation, emits events, and returns an exit code.

Cacheable adapters
------------------
Signature: ``(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int``

Same as one-shot, but accepts an optional ``model_cache`` dict that the
persistent daemon passes to allow model reuse across requests.  When
``model_cache`` is ``None``, behaves identically to a one-shot adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from modl_worker.protocol import EventEmitter


@runtime_checkable
class Adapter(Protocol):
    """One-shot adapter: loads config, runs operation, emits events."""

    def __call__(self, config_path: Path, emitter: EventEmitter) -> int: ...


@runtime_checkable
class CacheableAdapter(Protocol):
    """Adapter that can reuse models from a shared cache dict."""

    def __call__(
        self,
        config_path: Path,
        emitter: EventEmitter,
        model_cache: dict | None = None,
    ) -> int: ...
