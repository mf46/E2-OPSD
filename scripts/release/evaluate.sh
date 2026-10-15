#!/usr/bin/env bash
# Evaluate one checkpoint: n=16 samples per question with seeds 17, 42 and 1234.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
MODEL=${1:?Usage: evaluate.sh HF_MODEL_OR_ACTOR_DIR OUTPUT_DIR}
OUTPUT=${2:?Usage: evaluate.sh HF_MODEL_OR_ACTOR_DIR OUTPUT_DIR}
if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite evaluation output: $OUTPUT" >&2
  exit 1
fi
export PYTHONPATH="$REPO/third_party/verl:$REPO:${PYTHONPATH:-}"
if [ -f "$MODEL/fsdp_config.json" ]; then
  python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$MODEL" --target_dir "$OUTPUT/model"
  MODEL="$OUTPUT/model"
fi
export MODEL_PATH_OVERRIDE="$MODEL"
export WANDB_MODE=offline N_GPUS_PER_NODE_OVERRIDE=8
export VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE=0.4
export FREE_CACHE_ENGINE_OVERRIDE=false ENABLE_SLEEP_MODE_OVERRIDE=false
# Validation-only runs through the plain veRL entry (the GRPO config), so no
# training and no teacher are involved. The student sees only the question.
for seed in 17 42 1234; do
  OUTPUT_DIR_OVERRIDE="$OUTPUT" RUN_TAG_OVERRIDE="seed_$seed" \
    bash scripts/_run_verl.sh configs/release/qwen3_1_7b_thinking/grpo.yaml \
      trainer.resume_mode=disable trainer.val_before_train=True trainer.val_only=True \
      data.val_batch_size=4 \
      actor_rollout_ref.rollout.val_kwargs.n=16 actor_rollout_ref.rollout.val_kwargs.seed="$seed"
done
python3 scripts/summarize_eval.py "$OUTPUT/seed_17" "$OUTPUT/seed_42" "$OUTPUT/seed_1234" \
  --n 16 --pass-at-ks 1,2,4,8
