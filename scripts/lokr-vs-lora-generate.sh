#!/bin/bash
# LoRA vs LoKR — comprehensive generation comparison with timing
# 3 models x 3 conditions (none, lora, lokr) x 10 prompts x 3 seeds = 270 images
# Sequential execution (GPU safety)
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
SEEDS=(42 123 777)
SIZE="1:1"
LOG="/home/pedro/lokr-experiment-generate.log"
RESULTS="/home/pedro/lokr-experiment-generate-results.md"
OUTPUT_BASE="/home/pedro/lokr-experiment-images"

mkdir -p "$OUTPUT_BASE"

echo "=== LoRA vs LoKR Generation Comparison ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

cat > "$RESULTS" << 'HEADER'
# LoRA vs LoKR — Inference Results

## Inference Timing

| Model | Condition | Avg Time/Image | Images | Notes |
|-------|-----------|----------------|--------|-------|
HEADER

# 10 prompts — identity, context, accessories, artistic styles, edge cases
PROMPTS=(
  "a photo of maxi dog, close-up portrait, soft lighting, shallow depth of field"
  "a photo of maxi dog sitting in a sunny park, green grass, oak trees in background"
  "a photo of maxi dog running on a beach, ocean waves, golden hour light"
  "a photo of maxi dog wearing a small red bandana around its neck, studio portrait, white background"
  "a photo of maxi dog wearing tiny round sunglasses, sitting on a couch, cozy living room"
  "a photo of maxi dog in the snow, winter forest, snowflakes falling"
  "a photo of maxi dog next to a birthday cake with candles, party hat, celebration"
  "a watercolor painting of maxi dog in a garden, loose brushstrokes, soft pastel colors"
  "a pencil sketch of maxi dog, detailed graphite drawing on white paper"
  "a photo of maxi dog sitting in the cockpit of a spaceship, sci-fi, dramatic lighting"
)
PROMPT_NAMES=(portrait park beach bandana sunglasses snow birthday watercolor pencil spaceship)

# Baseline prompts (no trigger word — for no-lora runs)
BASELINE_PROMPTS=(
  "a photo of a pomeranian dog, close-up portrait, soft lighting, shallow depth of field"
  "a photo of a pomeranian dog sitting in a sunny park, green grass, oak trees in background"
  "a photo of a pomeranian dog running on a beach, ocean waves, golden hour light"
  "a photo of a pomeranian dog wearing a small red bandana around its neck, studio portrait, white background"
  "a photo of a pomeranian dog wearing tiny round sunglasses, sitting on a couch, cozy living room"
  "a photo of a pomeranian dog in the snow, winter forest, snowflakes falling"
  "a photo of a pomeranian dog next to a birthday cake with candles, party hat, celebration"
  "a watercolor painting of a pomeranian dog in a garden, loose brushstrokes, soft pastel colors"
  "a pencil sketch of a pomeranian dog, detailed graphite drawing on white paper"
  "a photo of a pomeranian dog sitting in the cockpit of a spaceship, sci-fi, dramatic lighting"
)

# Model configs
declare -A MODEL_STEPS MODEL_GUIDANCE
MODEL_STEPS[klein4b]=4
MODEL_STEPS[klein9b]=4
MODEL_STEPS[zimage]=6
MODEL_GUIDANCE[klein4b]=3.5
MODEL_GUIDANCE[klein9b]=3.5
MODEL_GUIDANCE[zimage]=3.5

# LoRA file paths
declare -A LORA_FILES
LORA_FILES[klein4b-lora]="$HOME/.modl/loras/maxi-klein4b-lora.safetensors"
LORA_FILES[klein4b-lokr]="$HOME/.modl/loras/maxi-klein4b-lokr.safetensors"
LORA_FILES[klein9b-lora]="$HOME/.modl/loras/maxi-klein9b-lora-v2.safetensors"
LORA_FILES[klein9b-lokr]="$HOME/.modl/loras/maxi-klein9b-lokr-v2.safetensors"
LORA_FILES[zimage-lora]="$HOME/.modl/loras/maxi-zimage-lora.safetensors"
LORA_FILES[zimage-lokr]="$HOME/.modl/loras/maxi-zimage-lokr.safetensors"

# Base model IDs
declare -A BASE_IDS
BASE_IDS[klein4b]="flux2-klein-4b"
BASE_IDS[klein9b]="flux2-klein-9b"
BASE_IDS[zimage]="z-image"

find_latest_output() {
  # modl organizes outputs by date subdirectory
  local today=$(date +%Y-%m-%d)
  ls -t ~/.modl/outputs/"$today"/*.png 2>/dev/null | head -1
}

generate_set() {
  local model_key=$1
  local condition=$2
  local base="${BASE_IDS[$model_key]}"
  local steps="${MODEL_STEPS[$model_key]}"
  local guidance="${MODEL_GUIDANCE[$model_key]}"
  local out_dir="$OUTPUT_BASE/${model_key}-${condition}"
  mkdir -p "$out_dir"

  local lora_arg=""
  local use_baseline=false
  if [ "$condition" = "none" ]; then
    use_baseline=true
  else
    local lora_file="${LORA_FILES[${model_key}-${condition}]}"
    if [ ! -f "$lora_file" ]; then
      echo "  SKIP: $lora_file not found" | tee -a "$LOG"
      return
    fi
    lora_arg="--lora $lora_file"
  fi

  echo "" | tee -a "$LOG"
  echo "--- $model_key + $condition ($base, ${steps} steps) ---" | tee -a "$LOG"

  local total_time=0
  local count=0

  for seed in "${SEEDS[@]}"; do
    for i in "${!PROMPTS[@]}"; do
      local pname="${PROMPT_NAMES[$i]}"
      local outfile="$out_dir/${pname}_seed${seed}.png"

      if [ -f "$outfile" ]; then
        echo "  EXISTS: $outfile" | tee -a "$LOG"
        count=$((count + 1))
        continue
      fi

      local prompt
      if $use_baseline; then
        prompt="${BASELINE_PROMPTS[$i]}"
      else
        prompt="${PROMPTS[$i]}"
      fi

      local start_time=$(date +%s%N)

      $MODL generate "$prompt" \
        --base "$base" \
        $lora_arg \
        --seed "$seed" \
        --size "$SIZE" \
        --steps "$steps" \
        --guidance "$guidance" \
        2>&1 | tee -a "$LOG"

      local end_time=$(date +%s%N)
      local elapsed_ms=$(( (end_time - start_time) / 1000000 ))
      total_time=$((total_time + elapsed_ms))
      count=$((count + 1))

      local latest=$(find_latest_output)
      if [ -n "$latest" ]; then
        cp "$latest" "$outfile"
        echo "  -> ${pname}_seed${seed} (${elapsed_ms}ms)" | tee -a "$LOG"
      else
        echo "  WARN: no output for ${pname}_seed${seed}" | tee -a "$LOG"
      fi
    done
  done

  if [ $count -gt 0 ]; then
    local avg=$((total_time / count))
    local avg_sec=$(echo "scale=1; $avg / 1000" | bc)
    echo "  Average: ${avg_sec}s per image ($count images, total ${total_time}ms)" | tee -a "$LOG"
    echo "| $base | $condition | ${avg_sec}s | $count | ${steps} steps |" >> "$RESULTS"
  fi
}

# Generate for each model: baseline, lora, lokr
for model_key in klein4b klein9b zimage; do
  echo "" | tee -a "$LOG"
  echo "========== ${model_key} ==========" | tee -a "$LOG"
  generate_set "$model_key" none
  generate_set "$model_key" lora
  generate_set "$model_key" lokr
done

# File size comparison
echo "" >> "$RESULTS"
echo "## File Size Comparison" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| Model | LoRA Size | LoKR Size | Reduction |" >> "$RESULTS"
echo "|-------|-----------|-----------|-----------|" >> "$RESULTS"

for model_key in klein4b klein9b zimage; do
  lora_file="${LORA_FILES[${model_key}-lora]}"
  lokr_file="${LORA_FILES[${model_key}-lokr]}"
  lora_size="N/A"; lokr_size="N/A"; reduction="N/A"
  if [ -f "$lora_file" ]; then
    lora_bytes=$(stat --format=%s "$lora_file" 2>/dev/null)
    lora_size=$(du -h "$lora_file" | cut -f1)
  fi
  if [ -f "$lokr_file" ]; then
    lokr_bytes=$(stat --format=%s "$lokr_file" 2>/dev/null)
    lokr_size=$(du -h "$lokr_file" | cut -f1)
  fi
  if [ -n "$lora_bytes" ] && [ -n "$lokr_bytes" ] && [ "$lora_bytes" -gt 0 ]; then
    pct=$(echo "scale=0; 100 - ($lokr_bytes * 100 / $lora_bytes)" | bc)
    reduction="${pct}%"
  fi
  echo "| ${BASE_IDS[$model_key]} | $lora_size | $lokr_size | $reduction |" >> "$RESULTS"
done

echo "" | tee -a "$LOG"
echo "=== All generation complete! ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Images: $OUTPUT_BASE/" | tee -a "$LOG"
echo "Results: $RESULTS" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Total images per directory:" | tee -a "$LOG"
for d in "$OUTPUT_BASE"/*/; do
  echo "  $(basename "$d"): $(ls "$d"/*.png 2>/dev/null | wc -l)" | tee -a "$LOG"
done
