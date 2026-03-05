#!/usr/bin/env bash
# Kill stale Vite processes, restart on port 5173
# Usage: ./scripts/dev-ui.sh

set -e

echo "→ Killing stale Vite processes..."
pkill -f "node.*vite" 2>/dev/null || true
sleep 0.5

# Double-check port is free
lsof -ti:5173 2>/dev/null | xargs kill 2>/dev/null || true
sleep 0.3

echo "→ Starting Vite dev server on http://localhost:5173"
cd "$(dirname "$0")/../src/ui/web"
exec npm run dev
