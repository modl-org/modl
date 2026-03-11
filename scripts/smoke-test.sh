#!/usr/bin/env bash
# Smoke test: generate one image per installed model to verify
# pipeline loading, config resolution, and basic inference work.
# Uses each model's default steps/guidance to produce real images.
#
# Usage:
#   ./scripts/smoke-test.sh              # test all installed models
#   ./scripts/smoke-test.sh flux-schnell # test a specific model
#
# Requires: modl binary built, worker running (auto-starts if needed)
set -uo pipefail

MODL="${MODL:-./target/debug/modl}"
export MODL_MAX_MODELS=1
PROMPT="a orange cat sitting in a sunny window"
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# model — uses default steps/guidance from model_family.rs
GEN_MODELS=(
  flux-schnell
  flux-dev
  sdxl-base-1.0
  z-image-turbo
  qwen-image
  flux2-klein-4b
  flux2-klein-9b
)

# Edit models
EDIT_MODELS=(
  qwen-image-edit
)

# Models that support --fast (Lightning LoRA)
FAST_EDIT_MODELS=(
  qwen-image-edit
)

TEST_IMAGE="/tmp/modl-smoke-test-input.png"

# Filter to specific model if arg provided
FILTER="${1:-}"

create_test_image() {
  if [ ! -f "$TEST_IMAGE" ]; then
    python3 -c "
from PIL import Image
img = Image.new('RGB', (512, 512), 'white')
img.save('$TEST_IMAGE')
" 2>/dev/null || {
      echo "WARN: Could not create test image (PIL not found), edit tests will be skipped"
      return 1
    }
  fi
}

# Restart worker with MODL_MAX_MODELS=1 for clean VRAM between models
$MODL worker stop &>/dev/null
echo "→ Starting worker (max_models=$MODL_MAX_MODELS)..."
$MODL worker start
sleep 3

run_gen_test() {
  local model="$1"
  local label="generate/$model"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! $MODL info "$model" 2>/dev/null | grep -q "Installed"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  # No --steps: use model default from model_family.rs
  if output=$($MODL generate "$PROMPT" \
      --base "$model" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Generated"; then
    echo "OK"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL"
    echo "$output" | tail -3 | sed 's/^/        /'
    FAILED=$((FAILED + 1))
    FAILURES+=("$label")
  fi
}

run_edit_test() {
  local model="$1"
  local label="edit/$model"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! $MODL info "$model" 2>/dev/null | grep -q "Installed"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  if [ ! -f "$TEST_IMAGE" ]; then
    echo "  SKIP  $label (no test image)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  if output=$($MODL edit "add a small orange cat" \
      --image "$TEST_IMAGE" \
      --base "$model" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Edited\|Generated"; then
    echo "OK"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL"
    echo "$output" | tail -3 | sed 's/^/        /'
    FAILED=$((FAILED + 1))
    FAILURES+=("$label")
  fi
}

run_fast_edit_test() {
  local model="$1"
  local label="edit/$model --fast"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! $MODL info "$model" 2>/dev/null | grep -q "Installed"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  if [ ! -f "$TEST_IMAGE" ]; then
    echo "  SKIP  $label (no test image)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  # --fast requires LoRA which is incompatible with GGUF variants
  if $MODL info "$model" 2>/dev/null | grep "Variant:" | grep -q "gguf"; then
    echo "  SKIP  $label (GGUF variant, LoRA not supported)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  if output=$($MODL edit "add a small orange cat" \
      --image "$TEST_IMAGE" \
      --base "$model" \
      --fast \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Edited\|Generated"; then
    echo "OK"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL"
    echo "$output" | tail -3 | sed 's/^/        /'
    FAILED=$((FAILED + 1))
    FAILURES+=("$label")
  fi
}

echo "=== modl smoke test ==="
echo "  Prompt: $PROMPT"
echo ""

create_test_image

echo "--- txt2img (default steps) ---"
for model in "${GEN_MODELS[@]}"; do
  run_gen_test "$model"
done

echo ""
echo "--- edit (default steps) ---"
for model in "${EDIT_MODELS[@]}"; do
  run_edit_test "$model"
done

echo ""
echo "--- edit --fast (Lightning LoRA) ---"
for model in "${FAST_EDIT_MODELS[@]}"; do
  run_fast_edit_test "$model"
done

echo ""
echo "=== Results ==="
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"

if [ ${#FAILURES[@]} -gt 0 ]; then
  echo ""
  echo "  Failures:"
  for f in "${FAILURES[@]}"; do
    echo "    - $f"
  done
  exit 1
fi
