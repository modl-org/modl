#!/usr/bin/env python3
"""Validate that models.toml arch_keys and Python ARCH_CONFIGS stay in sync.

Run from project root:
    python scripts/validate-arch-sync.py
"""
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Load arch_keys from models.toml
with open(root / "models.toml", "rb") as f:
    toml_data = tomllib.load(f)

toml_arch_keys = set()
for family in toml_data["families"].values():
    for model in family["models"].values():
        toml_arch_keys.add(model["arch_key"])

# Load arch_keys from Python ARCH_CONFIGS
sys.path.insert(0, str(root / "python"))
from modl_worker.adapters.arch_config import ARCH_CONFIGS

py_arch_keys = set(ARCH_CONFIGS.keys())

# Compare
missing_from_python = toml_arch_keys - py_arch_keys
extra_in_python = py_arch_keys - toml_arch_keys

ok = True

if missing_from_python:
    print(f"❌ In models.toml but not ARCH_CONFIGS: {missing_from_python}")
    ok = False

if extra_in_python:
    # These are Python-only arch variants (e.g. flux2_klein_base) that don't
    # have user-facing models in models.toml. Warn but don't fail.
    print(f"⚠  In ARCH_CONFIGS but not models.toml: {extra_in_python}")
    print("   (OK if these are internal/base variants without user-facing models)")

if ok:
    print(f"✅ Arch sync OK ({len(toml_arch_keys)} model arch_keys, {len(py_arch_keys)} Python configs)")
else:
    sys.exit(1)
