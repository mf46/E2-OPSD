#!/usr/bin/env bash
# Training entry for Qwen3-1.7B Thinking: 360 optimizer updates for every method.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
METHOD=${1:?Usage: train_qwen3_1_7b_thinking.sh e2opsd|opsd|sdpo|rlsd|eopd|grpo [Hydra overrides...]}
shift
NUM_NEIGHBORS=15
NEIGHBORS=data/egt/qwen3_1_7b_thinking/neighbors.json
EXTRA=(trainer.resume_mode=disable)
case "$METHOD" in
  e2opsd|opsd|sdpo) ;;
  rlsd|eopd) EXTRA+=('actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]') ;;
  grpo)
    EXTRA+=('actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]' algorithm.norm_adv_by_std_in_grpo=True actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.actor.loss_agg_mode=token-mean)
    ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac
if [ "$METHOD" = e2opsd ]; then
  if [ ! -f "$NEIGHBORS" ]; then
    echo "Build the EGT neighbor map with scripts/release/build_qwen3_1_7b_neighbors.sh first." >&2
    exit 1
  fi
  "${PYTHON:-python3}" - "$NEIGHBORS" "$NUM_NEIGHBORS" <<'CHECK_NEIGHBORS'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
if not rows or any(len(v) != expected or len(set(v)) != expected or k in v for k, v in rows.items()):
    raise SystemExit(f"Expected exactly {expected} distinct non-self neighbors per question; rebuild the map.")
print(f"Verified {expected} non-self neighbors for {len(rows)} questions.")
CHECK_NEIGHBORS
fi
export N_GPUS_PER_NODE_OVERRIDE=8
export VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE=0.4
export FREE_CACHE_ENGINE_OVERRIDE=false ENABLE_SLEEP_MODE_OVERRIDE=false
export WANDB_MODE=offline
# Release settings are appended last so a caller cannot silently change the budget.
exec bash "$REPO/scripts/_run_verl.sh" "$REPO/configs/release/qwen3_1_7b_thinking/$METHOD.yaml" \
  "$@" "${EXTRA[@]}" \
  trainer.training_step_unit=optimizer trainer.total_training_steps=360 \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  +data.val_apply_chat_template_kwargs.enable_thinking=True
