#!/bin/bash
# Retry Klein 9B LoRA + LoKR after clearing GPU memory and fixing quantization
# Uses prodigy (same as other runs) — OOM was caused by missing quantization, not prodigy
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
LOG="/home/pedro/lokr-experiment.log"
RESULTS="/home/pedro/lokr-experiment-results.md"

echo "" | tee -a "$LOG"
echo "=== Klein 9B Retry (adamw8bit) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# Kill stale GPU processes to free memory
echo "Clearing stale GPU processes..." | tee -a "$LOG"
# ComfyUI leak (386MB GPU), stuck cloud trains from Mar 27
for pid in 453430 485542 485540 896240 896238; do
  kill "$pid" 2>/dev/null
done
sleep 5
nvidia-smi | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_train() {
  local num=$1
  local adapter=$2
  local name=$3

  echo "[$num] Klein 9B + $adapter — started $(date)" | tee -a "$LOG"
  local start_time=$(date +%s)

  $MODL train \
    --dataset maxi \
    --base flux2-klein-9b \
    --name "$name" \
    --trigger maxi \
    --class-word dog \
    --lora-type character \
    --steps 1000 \
    --rank 16 \
    --optimizer prodigy \
    --lr 1.0 \
    --seed 42 \
    --sample-every 500 \
    --network-type "$adapter" \
    --preset quick \
    2>&1 | tee -a "$LOG"

  local exit_code=$?
  local end_time=$(date +%s)
  local elapsed=$(( end_time - start_time ))
  local minutes=$(( elapsed / 60 ))
  local seconds=$(( elapsed % 60 ))
  local time_str="${minutes}m ${seconds}s"

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

  echo "[$num] Klein 9B + $adapter — done in $time_str (size: $file_size) — $(date)" | tee -a "$LOG"
  echo "| $num | flux2-klein-9b | $adapter | $time_str | $file_size | $status (adamw8bit) |" >> "$RESULTS"
}

# 3. Klein 9B + LoRA (retry)
run_train "3r" lora maxi-klein9b-lora-v2

# 4. Klein 9B + LoKR (retry)
run_train "4r" lokr maxi-klein9b-lokr-v2

echo "" | tee -a "$LOG"
echo "=== Klein 9B retry complete ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
