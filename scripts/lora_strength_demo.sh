#!/usr/bin/env bash
# lora_strength_demo.sh
#
# Runs inference across multiple prompts and LoRA strength values using modl,
# so you can visually compare how the style bleeds in at 0 → 1.
#
# Usage:
#   ./scripts/lora_strength_demo.sh [LORA_NAME] [BASE_MODEL] [TRIGGER_WORD]
#
# Defaults:
#   LORA_NAME  = kids-art-sdxl-v2
#   BASE_MODEL = sdxl-base-1.0
#   TRIGGER    = KIDSART

set -euo pipefail

LORA="${1:-kids-art-sdxl-v2}"
BASE="${2:-sdxl-base-1.0}"
TRIGGER="${3:-KIDSART}"
SEED=42

# ── Model-aware inference defaults ────────────────────────────────────────────
BASE_LC="$(echo "${BASE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${BASE_LC}" == *"schnell"* || "${BASE_LC}" == *"turbo"* || "${BASE_LC}" == *"lightning"* ]]; then
  STEPS=4
  GUIDANCE=0.0
elif [[ "${BASE_LC}" == *"sdxl"* ]]; then
  STEPS=30
  GUIDANCE=7.5
else
  STEPS=28
  GUIDANCE=3.5
fi

# ── Strength sweep ────────────────────────────────────────────────────────────
# Includes 1.0 to show over-application; practical sweet spot is often 0.4-0.8.
STRENGTHS=(0.0 0.3 0.5 0.7 1.0)

# ── Prompt bodies (no trigger token baked in) ────────────────────────────────
declare -a PROMPT_BODIES=(
  "a dog playing in the park"
  "a rocket ship flying through colorful space"
  "a princess in a magical forest"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD=$(tput bold 2>/dev/null || echo "")
RESET=$(tput sgr0 2>/dev/null || echo "")
CYAN=$(tput setaf 6 2>/dev/null || echo "")

total=$(( ${#PROMPT_BODIES[@]} * (2 + 2 * ${#STRENGTHS[@]}) ))
current=0

echo ""
echo "${BOLD}${CYAN}LoRA Strength Demo${RESET}"
echo "  LoRA:   ${LORA}"
echo "  Base:   ${BASE}"
echo "  Trigger:${TRIGGER}"
echo "  Seed:   ${SEED}"
echo "  Steps:  ${STEPS}"
echo "  CFG:    ${GUIDANCE}"
echo "  Combos: ${total}"
echo "  Modes:  base/no-trigger, base/with-trigger, lora/no-trigger, lora/with-trigger"
echo ""

run_case() {
  local mode="$1"
  local prompt="$2"
  local use_lora="$3"
  local strength="${4:-}"

  (( current++ )) || true
  if [[ "${use_lora}" == "yes" ]]; then
    echo "${BOLD}[${current}/${total}]${RESET} ${mode} | strength=${strength} | \"${prompt}\""
  else
    echo "${BOLD}[${current}/${total}]${RESET} ${mode} | \"${prompt}\""
  fi

  local -a cmd=(
    modl generate "${prompt}"
    --base "${BASE}"
    --seed "${SEED}"
    --size "1:1"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
  )

  if [[ "${use_lora}" == "yes" ]]; then
    cmd+=(--lora "${LORA}" --lora-strength "${strength}")
  fi

  "${cmd[@]}"
  echo ""
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for body in "${PROMPT_BODIES[@]}"; do
  prompt_plain="${body}"
  prompt_with_trigger="${TRIGGER} ${body}"

  # Controls (no LoRA): isolates trigger-word-only behavior.
  run_case "base/no-trigger" "${prompt_plain}" "no"
  run_case "base/with-trigger" "${prompt_with_trigger}" "no"

  # LoRA conditions at matched seeds/settings: isolates keyword + LoRA effects.
  for strength in "${STRENGTHS[@]}"; do
    run_case "lora/no-trigger" "${prompt_plain}" "yes" "${strength}"
    run_case "lora/with-trigger" "${prompt_with_trigger}" "yes" "${strength}"
  done
done

echo "${BOLD}Done!${RESET} Images saved to ~/.modl/outputs/$(date +%Y-%m-%d)/"
