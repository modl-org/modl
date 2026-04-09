#!/bin/bash
# LoRA vs LoKR experiment — Maxi (Pomeranian) character training
# 3 models x 2 adapter types = 6 sequential training runs
# RTX 4090 24GB — prodigy optimizer, 1000 steps each
# Controlled variables: same dataset, trigger, seed, optimizer, rank, resolution
#
# Don't set -e — if one run fails (e.g. OOM), continue to next
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
DATASET="maxi"
TRIGGER="maxi"
CLASS="dog"
TYPE="character"
SEED=42
STEPS=1000
RANK=16
OPTIMIZER="prodigy"
LR="1.0"
SAMPLE_EVERY=500
LOG="/home/pedro/lokr-experiment.log"
RESULTS="/home/pedro/lokr-experiment-results.md"

echo "=== LoRA vs LoKR Experiment ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Dataset: $DATASET | Trigger: $TRIGGER | Seed: $SEED" | tee -a "$LOG"
echo "Steps: $STEPS | Rank: $RANK | Optimizer: $OPTIMIZER" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Initialize results markdown
cat > "$RESULTS" << 'HEADER'
# LoRA vs LoKR Experiment Results

## Setup
- **Dataset:** Maxi (Pomeranian dog, 24 images)
- **Trigger:** `maxi`
- **Optimizer:** prodigy (LR=1.0)
- **Steps:** 1000
- **Rank:** 16
- **Seed:** 42
- **Samples at:** step 1, 500, 1000
- **GPU:** RTX 4090 24GB

## Training Results

| # | Model | Adapter | Time | File Size | Status |
|---|-------|---------|------|-----------|--------|
HEADER

run_train() {
  local num=$1
  local model=$2
  local adapter=$3
  local name=$4

  echo "[$num/6] $model + $adapter — started $(date)" | tee -a "$LOG"
  local start_time=$(date +%s)

  $MODL train \
    --dataset "$DATASET" \
    --base "$model" \
    --name "$name" \
    --trigger "$TRIGGER" \
    --class-word "$CLASS" \
    --lora-type "$TYPE" \
    --steps "$STEPS" \
    --rank "$RANK" \
    --optimizer "$OPTIMIZER" \
    --lr "$LR" \
    --seed "$SEED" \
    --sample-every "$SAMPLE_EVERY" \
    --network-type "$adapter" \
    --preset quick \
    2>&1 | tee -a "$LOG"

  local exit_code=$?
  local end_time=$(date +%s)
  local elapsed=$(( end_time - start_time ))
  local minutes=$(( elapsed / 60 ))
  local seconds=$(( elapsed % 60 ))
  local time_str="${minutes}m ${seconds}s"

  # Find the output safetensors file size
  local output_dir="$HOME/.modl/training_output/$name"
  local file_size="N/A"
  local final_file=$(find "$output_dir" -name "${name}.safetensors" 2>/dev/null | head -1)
  if [ -n "$final_file" ]; then
    file_size=$(du -h "$final_file" | cut -f1)
  fi

  local status="ok"
  if [ $exit_code -ne 0 ]; then
    status="FAILED (exit $exit_code)"
  fi

  echo "[$num/6] $model + $adapter — done in $time_str (size: $file_size) — $(date)" | tee -a "$LOG"
  echo "" | tee -a "$LOG"

  # Append to results table
  echo "| $num | $model | $adapter | $time_str | $file_size | $status |" >> "$RESULTS"
}

# --- 6 training runs ---

# 1. Klein 4B + LoRA
run_train 1 flux2-klein-4b lora maxi-klein4b-lora

# 2. Klein 4B + LoKR
run_train 2 flux2-klein-4b lokr maxi-klein4b-lokr

# 3. Klein 9B + LoRA
run_train 3 flux2-klein-9b lora maxi-klein9b-lora

# 4. Klein 9B + LoKR
run_train 4 flux2-klein-9b lokr maxi-klein9b-lokr

# 5. Z-Image + LoRA
run_train 5 z-image lora maxi-zimage-lora

# 6. Z-Image + LoKR
run_train 6 z-image lokr maxi-zimage-lokr

echo "" >> "$RESULTS"
echo "## Generation Comparison" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "Run the generation script next to produce side-by-side comparisons." >> "$RESULTS"

echo "" | tee -a "$LOG"
echo "=== All training complete! ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Results: cat $RESULTS" | tee -a "$LOG"
echo "Next: run generation comparison script" | tee -a "$LOG"
