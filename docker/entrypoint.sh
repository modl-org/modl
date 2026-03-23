#!/bin/bash
set -e

# Use persistent volume if available (RunPod mounts /workspace)
if [ -d "/workspace" ]; then
    export MODL_ROOT=/workspace/modl
    mkdir -p "$MODL_ROOT"
    # Symlink config so modl finds existing state across restarts
    if [ ! -f "$HOME/.modl/config.yaml" ]; then
        modl init --defaults --root "$MODL_ROOT"
    fi
fi

# If MODEL env var is set and not already installed, pull it
if [ -n "$MODEL" ]; then
    if ! modl ls 2>/dev/null | grep -q "$MODEL"; then
        echo "Pulling model: $MODEL"
        modl pull "$MODEL"
    fi
fi

exec modl serve --foreground --port "${PORT:-3939}"
