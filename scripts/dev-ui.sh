#!/usr/bin/env bash
# Kill stale Vite processes, restart with hot-reload
# Binds to 0.0.0.0 so it works over Tailscale / SSH port forwarding.
# Usage: ./scripts/dev-ui.sh

set -e

echo "→ Killing stale Vite processes..."
pkill -f "node.*vite" 2>/dev/null || true
sleep 0.5

# Double-check port is free
lsof -sTCP:LISTEN -ti:5173 2>/dev/null | xargs kill 2>/dev/null || true
sleep 0.3

# Detect Tailscale IP for remote access
TS_IP=$(tailscale ip -4 2>/dev/null || true)

echo "→ Starting Vite dev server..."
if [[ -n "$TS_IP" ]]; then
    echo "  Local:     http://localhost:5173"
    echo "  Tailscale: http://${TS_IP}:5173"
else
    echo "  http://localhost:5173"
fi

cd "$(dirname "$0")/../src/ui/web"
exec npm run dev -- --host 0.0.0.0
