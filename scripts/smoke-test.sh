#!/usr/bin/env bash
# Smoke test: generate one image per installed model to verify
# pipeline loading, config resolution, and basic inference work.
# Uses each model's default steps/guidance to produce real images.
#
# The edit tests use a generated photo as input (not a blank image),
# so the edits are meaningful and the output quality is verifiable.
#
# Usage:
#   ./scripts/smoke-test.sh              # test all installed models
#   ./scripts/smoke-test.sh flux-schnell # test a specific model
#
# Requires: modl binary built, worker running (auto-starts if needed)
set -uo pipefail

MODL="${MODL:-./target/debug/modl}"
export MODL_MAX_MODELS=1
GEN_PROMPT="a orange cat sitting in a sunny window"
EDIT_PROMPT="add a tiny golden crown on the cat's head"
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# Minimum output file size (bytes) — catches blank/corrupt images
MIN_OUTPUT_SIZE=20000

# model — uses default steps/guidance from model_family.rs
GEN_MODELS=(
  flux-schnell
  flux-dev
  sdxl-base-1.0
  z-image-turbo
  z-image
  qwen-image
  chroma
  flux2-dev
  flux2-klein-4b
  flux2-klein-9b
)

# Edit models
EDIT_MODELS=(
  qwen-image-edit-2511
  flux2-klein-4b
  flux2-klein-9b
)

# Video models (txt2vid)
VIDEO_MODELS=(
  ltx-video-dev
)

# Models that support --fast (Lightning LoRA)
FAST_EDIT_MODELS=(
  qwen-image-edit-2511
)

# Models that support --fast for generation
FAST_GEN_MODELS=(
  qwen-image
)

TEST_IMAGE="/tmp/modl-smoke-test-input.png"

# Filter to specific model if arg provided
FILTER="${1:-}"

# Cache installed model list (avoids repeated modl ls calls)
INSTALLED_MODELS=$($MODL ls 2>/dev/null)

# Match on the ID column exactly (last column, between ┆ and │)
# Avoids "z-image" matching "z-image-turbo" or "qwen-image" matching "qwen-image-edit"
is_installed() {
  echo "$INSTALLED_MODELS" | grep -qP "┆ ${1}\s+│"
}



# Generate a real test image using the first available model.
# A real photo is much better for edit testing than a blank white image.
create_test_image() {
  if [ -f "$TEST_IMAGE" ]; then
    return 0
  fi

  # Pick the first installed generate model
  local gen_model=""
  for m in z-image-turbo flux-schnell flux2-klein-4b sdxl-base-1.0; do
    if is_installed "$m"; then
      gen_model="$m"
      break
    fi
  done

  if [ -z "$gen_model" ]; then
    echo "WARN: No generate model installed, edit tests will be skipped"
    return 1
  fi

  echo -n "  → Generating test image with $gen_model ... "
  local output
  if output=$($MODL generate "$GEN_PROMPT" \
      --base "$gen_model" \
      --count 1 \
      --seed 12345 \
      --size 512x512 \
      --no-worker \
      2>&1); then
    # Extract the output path (on its own line, indented)
    local img_path
    img_path=$(echo "$output" | grep -oP '\S+\.png' | tail -1)
    if [ -n "$img_path" ] && [ -f "$img_path" ]; then
      cp "$img_path" "$TEST_IMAGE"
      echo "OK ($(du -h "$TEST_IMAGE" | cut -f1))"
      return 0
    fi
  fi
  echo "FAIL — falling back to synthetic image"
  # Fallback: create a simple gradient image (better than blank white)
  python3 -c "
from PIL import Image
import random
img = Image.new('RGB', (512, 512))
for x in range(512):
    for y in range(512):
        img.putpixel((x,y), (x//2, y//2, 128))
img.save('$TEST_IMAGE')
" 2>/dev/null || return 1
}

check_output_size() {
  local output="$1"
  local label="$2"
  local file_path
  file_path=$(echo "$output" | grep -oP '\S+\.(png|mp4)' | tail -1)
  if [ -n "$file_path" ] && [ -f "$file_path" ]; then
    local size
    size=$(stat -c%s "$file_path" 2>/dev/null || echo 0)
    if [ "$size" -lt "$MIN_OUTPUT_SIZE" ]; then
      echo -n "WARN ($(( size / 1024 ))KB) "
    fi
  fi
}

# Generate test image BEFORE starting the worker (--no-worker, releases VRAM)
create_test_image

# Restart worker with MODL_MAX_MODELS=1 for clean VRAM between models
$MODL worker stop &>/dev/null
echo "→ Starting worker (max_models=$MODL_MAX_MODELS)..."
$MODL worker start
sleep 3

run_gen_test() {
  local model="$1"
  local label="generate/$model"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! is_installed "$model"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  # No --steps: use model default from model_family.rs
  if output=$($MODL generate "$GEN_PROMPT" \
      --base "$model" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Generated"; then
    check_output_size "$output" "$label"
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

  if ! is_installed "$model"; then
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
  if output=$($MODL edit "$EDIT_PROMPT" \
      --image "$TEST_IMAGE" \
      --base "$model" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Edited\|Generated"; then
    check_output_size "$output" "$label"
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
  local fast_steps="${2:-4}"
  local label="edit/$model --fast $fast_steps"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! is_installed "$model"; then
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
  if output=$($MODL edit "$EDIT_PROMPT" \
      --image "$TEST_IMAGE" \
      --base "$model" \
      --fast "$fast_steps" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Edited\|Generated"; then
    check_output_size "$output" "$label"
    echo "OK"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL"
    echo "$output" | tail -3 | sed 's/^/        /'
    FAILED=$((FAILED + 1))
    FAILURES+=("$label")
  fi
}

run_video_test() {
  local model="$1"
  local label="txt2vid/$model"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! is_installed "$model"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  # Use small frame count (25 = 8*3+1) for fast smoke test
  if output=$($MODL generate "$GEN_PROMPT" \
      --base "$model" \
      --frames 25 \
      --fps 24 \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Generated"; then
    check_output_size "$output" "$label"
    echo "OK"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL"
    echo "$output" | tail -3 | sed 's/^/        /'
    FAILED=$((FAILED + 1))
    FAILURES+=("$label")
  fi
}

run_fast_gen_test() {
  local model="$1"
  local fast_steps="${2:-4}"
  local label="generate/$model --fast $fast_steps"

  [[ -n "$FILTER" && "$model" != "$FILTER" ]] && return

  if ! is_installed "$model"; then
    echo "  SKIP  $label (not installed)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  echo -n "  TEST  $label ... "
  local output
  if output=$($MODL generate "$GEN_PROMPT" \
      --base "$model" \
      --fast "$fast_steps" \
      --count 1 \
      --seed 42 \
      2>&1) && echo "$output" | grep -q "Generated"; then
    check_output_size "$output" "$label"
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
echo "  Generate: $GEN_PROMPT"
echo "  Edit:     $EDIT_PROMPT"
echo ""

echo "--- txt2img (default steps) ---"
for model in "${GEN_MODELS[@]}"; do
  run_gen_test "$model"
done

echo ""
echo "--- txt2img --fast 4 (Lightning LoRA) ---"
for model in "${FAST_GEN_MODELS[@]}"; do
  run_fast_gen_test "$model" 4
done

echo ""
echo "--- txt2img --fast 8 (Lightning LoRA) ---"
for model in "${FAST_GEN_MODELS[@]}"; do
  run_fast_gen_test "$model" 8
done

echo ""
echo "--- txt2vid (25 frames) ---"
for model in "${VIDEO_MODELS[@]}"; do
  run_video_test "$model"
done

echo ""
echo "--- edit (default steps) ---"
for model in "${EDIT_MODELS[@]}"; do
  run_edit_test "$model"
done

echo ""
echo "--- edit --fast 4 (Lightning LoRA) ---"
for model in "${FAST_EDIT_MODELS[@]}"; do
  run_fast_edit_test "$model" 4
done

echo ""
echo "--- edit --fast 8 (Lightning LoRA) ---"
for model in "${FAST_EDIT_MODELS[@]}"; do
  run_fast_edit_test "$model" 8
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
