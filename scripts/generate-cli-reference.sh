#!/bin/bash
# Generate CLI reference for README from `modl cli-schema`
# Usage: ./scripts/generate-cli-reference.sh
#
# Requires: modl binary in PATH (or cargo run), jq

set -euo pipefail

MODL="${MODL_BIN:-modl}"

# Get schema JSON
SCHEMA=$($MODL cli-schema 2>/dev/null)

echo "## Commands"
echo ""
echo "<!-- BEGIN AUTO-GENERATED (scripts/generate-cli-reference.sh) -->"
echo ""
echo "Run \`modl <command> --help\` for full usage details."
echo ""

# Extract and format commands as a table
echo "$SCHEMA" | jq -r '
  .commands[] |
  "| `\(.usage)` | \(.description) |"
' | sort | (
  echo "| Command | Description |"
  echo "|---------|-------------|"
  cat
)

echo ""
echo "<!-- END AUTO-GENERATED -->"
