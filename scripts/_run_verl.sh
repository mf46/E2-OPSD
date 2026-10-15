#!/bin/bash
set -eo pipefail
set -x

# Base script: reads all params from a YAML config and launches veRL training.
# Usage: bash scripts/_run_verl.sh <config.yaml> [extra hydra overrides...]
# Run it from the Python environment set up in the README.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CONFIG=${1:?Usage: $0 <config.yaml> [extra overrides...]}
shift  # remaining args are extra hydra overrides

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-""}
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.8"}
export PATH="$CUDA_HOME/bin:$PATH"
if [ -n "${HF_ENDPOINT}" ]; then export HF_ENDPOINT; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}
# Note: do NOT set TORCH_DISTRIBUTED_DEBUG=DETAIL — its debug PG wrapper
# lacks `allgather_into_tensor_coalesced`, which FSDP2/DTensor need when
# resharding weights to vLLM. NCCL flight recorder above is enough to
# capture stack traces on collective desync.

cd "${REPO_ROOT}"
export PYTHONPATH="$(pwd)/third_party/verl:$(pwd):${PYTHONPATH}"

# --- Read YAML helper ---
Y() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('$1','${2:-}'))"; }

METHOD=$(Y method grpo)
MODEL_PATH="${MODEL_PATH_OVERRIDE:-$(Y model_path Qwen/Qwen3-1.7B)}"
USE_LORA=$(Y use_lora true)
SEED=$(Y seed 42)
FULL_DETERMINISM=$(Y full_determinism false)
LORA_R=$(Y lora_r 64)
LORA_ALPHA=$(Y lora_alpha 128)
LR=$(Y learning_rate 1e-5)
NUM_EPOCHS=$(Y num_epochs 30)
TOTAL_TRAINING_STEPS=$(Y total_training_steps "")
TRAINING_STEP_UNIT=$(Y training_step_unit rollout)
BS=$(Y per_device_batch_size 4)
GROUP_SIZE=$(Y group_size 4)
MAX_GRAD_NORM=$(Y max_grad_norm 1.0)
WARMUP="${WARMUP_OVERRIDE:-$(Y warmup_steps 50)}"
WEIGHT_DECAY=$(Y weight_decay 0.01)
MAX_TRAIN=$(Y max_train_samples 17000)
SAVE_STEPS="${SAVE_STEPS_OVERRIDE:-$(Y save_steps 200)}"
EVAL_STEPS="${EVAL_STEPS_OVERRIDE:-$(Y eval_steps 100)}"
SAVE_BEST_ONLY=$(Y save_best_only false)
SAVE_BEST_ONLY_KEEP_LATEST=$(Y save_best_only_keep_latest true)
CHECKPOINT_RETENTION_MODE=$(Y checkpoint_retention_mode full)
MAX_PROMPT=$(Y max_prompt_length 2048)
MAX_RESP=$(Y max_completion_length 2048)
VAL_MAX_RESP=$(Y val_max_completion_length 38912)
ACTOR_MAX_TOKEN_LEN=$(Y actor_max_token_len_per_gpu "")
TEMP=$(Y temperature 1.0)
TOP_P=$(Y top_p 0.95)
TOP_K=$(Y top_k_sampling 20)
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE:-$(Y vllm_gpu_memory_utilization 0.6)}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE_OVERRIDE:-true}"
ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE_OVERRIDE:-true}"
TP=$(Y vllm_tensor_parallel_size 2)
STUDENT_ENABLE_THINKING=$(Y student_enable_thinking false)
VAL_ENABLE_THINKING=$(Y val_enable_thinking true)
VAL_DO_SAMPLE=$(Y val_do_sample true)
VAL_N=$(Y val_n 12)
VAL_TEMP=$(Y val_temperature 0.6)
VAL_TOP_P=$(Y val_top_p ${TOP_P})
VAL_TOP_K=$(Y val_top_k ${TOP_K})
VAL_BATCH_SIZE=$(Y val_batch_size 16)
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU_OVERRIDE:-4}"
EVAL_COMPUTE_ENTROPY="${EVAL_COMPUTE_ENTROPY_OVERRIDE:-$(Y eval_compute_entropy false)}"
CALCULATE_ENTROPY=$(Y calculate_entropy true)
ENTROPY_FROM_LOGITS_WITH_CHUNKING=$(Y entropy_from_logits_with_chunking false)
ENTROPY_CHUNK_SIZE=$(Y entropy_chunk_size 2048)
TEACHER_ENABLE_THINKING=$(Y teacher_enable_thinking true)
THINKING_SYSTEM_PROMPT=$(Y thinking_system_prompt false)
PROJECT=$(Y project_name e2opsd)
EXPERIMENT_NAME=$(Y experiment_name ${METHOD})
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$(Y output_dir "")}"
CONFIG_RUN_TAG=$(Y run_tag "")
CONFIG_RUN_TAG="${RUN_TAG_OVERRIDE:-$CONFIG_RUN_TAG}"
VAL_BEFORE=$(Y val_before_train false)
VAL_ONLY=$(Y val_only false)
USE_SWANLAB=$(Y use_swanlab true)
PRIVILEGED_TEXT_MODE=$(Y privileged_text_mode solution_answer)
# --- EGT: teacher context = a retrieved solved neighboring problem ---
EGT=$(Y egt false)
EGT_NEIGHBORS_PATH=$(Y egt_neighbors_path "")
EGT_MAX_SOLUTIONS=$(Y egt_max_solutions_per_question 8)
EGT_MIN_HIT_FRACTION=$(Y egt_min_hit_fraction 0.5)
EGT_PROBLEM_CHAR_BUDGET=$(Y egt_problem_char_budget 0)
EGT_SOLUTION_CHAR_BUDGET=$(Y egt_solution_char_budget 0)
SAVE_TRAIN_ROLLOUTS=$(Y save_train_rollouts false)
USE_WANDB=$(Y use_wandb false)
N_GPUS_PER_NODE=${N_GPUS_PER_NODE_OVERRIDE:-$(Y n_gpus_per_node 8)}
DATA_ROOT=$(Y data_dir ./data)
TRAIN_DATASET=$(Y train_dataset deepmath_filtered_level5_7)
VAL_DATASET=$(Y val_dataset amc23,aime24,aime25,hmmt25)
TRAIN_FILES=$(Y train_files "")
VAL_FILES=$(Y val_files "")
REWARD_MANAGER_NAME=$(Y reward_manager_name "")
USE_CUSTOM_REWARD_FUNCTION=$(Y use_custom_reward_function true)
CUSTOM_REWARD_FUNCTION_PATH=$(Y custom_reward_function_path src/verl_reward.py)
CUSTOM_REWARD_FUNCTION_NAME=$(Y custom_reward_function_name compute_score)
NCCL_TIMEOUT=$(Y nccl_timeout 7200)
ROLLOUT_IS=$(Y rollout_is "")
ROLLOUT_IS_THRESHOLD=$(Y rollout_is_threshold "")
ROLLOUT_IS_BATCH_NORMALIZE=$(Y rollout_is_batch_normalize "")
ROLLOUT_RS=$(Y rollout_rs "")
ROLLOUT_RS_THRESHOLD=$(Y rollout_rs_threshold "")
OVERRIDE_MAX_POSITION_EMBEDDINGS=$(Y override_max_position_embeddings "")
ROPE_SCALING_TYPE=$(Y rope_scaling_type "")
ROPE_SCALING_FACTOR=$(Y rope_scaling_factor "")
ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS=$(Y rope_scaling_original_max_position_embeddings "")
KL_LOSS_COEF=$(Y kl_loss_coef "")
RESUME_DIR=""
SWANLAB_RUN_ID_OVERRIDE=""

if [ "${N_GPUS_PER_NODE}" -le 0 ] || [ "${TP}" -le 0 ] || [ $((N_GPUS_PER_NODE % TP)) -ne 0 ]; then
    echo "n_gpus_per_node (${N_GPUS_PER_NODE}) must be positive and divisible by vllm_tensor_parallel_size (${TP})" >&2
    exit 1
fi
NUM_DP=$((N_GPUS_PER_NODE / TP))
ROLLOUT_BATCH=$((BS * NUM_DP))                    # prompts per rollout
ROLLOUT_SAMPLES=$((ROLLOUT_BATCH * GROUP_SIZE))    # samples per rollout

# ppo_mini_batch_size is expressed in prompt groups.  The launcher/worker
# multiplies it by group_size before splitting response trajectories.
PPO_MINI_BS=$(Y ppo_mini_batch_size ${ROLLOUT_BATCH})

case "$TRAINING_STEP_UNIT" in
    rollout|optimizer) ;;
    *)
        echo "training_step_unit must be 'rollout' or 'optimizer', got '${TRAINING_STEP_UNIT}' in ${CONFIG}" >&2
        exit 1
        ;;
esac
if [ "$TRAINING_STEP_UNIT" = "optimizer" ]; then
    if [ $((ROLLOUT_BATCH % PPO_MINI_BS)) -ne 0 ]; then
        echo "optimizer-step mode requires rollout prompt batch (${ROLLOUT_BATCH}) to be divisible by PPO mini-batch (${PPO_MINI_BS})" >&2
        exit 1
    fi
    OPTIMIZER_STEPS_PER_ROLLOUT=$((ROLLOUT_BATCH / PPO_MINI_BS))
    if [ -n "$TOTAL_TRAINING_STEPS" ] && [ "$TOTAL_TRAINING_STEPS" != "None" ] \
        && [ $((TOTAL_TRAINING_STEPS % OPTIMIZER_STEPS_PER_ROLLOUT)) -ne 0 ]; then
        echo "total_training_steps (${TOTAL_TRAINING_STEPS}) must be divisible by optimizer steps per rollout (${OPTIMIZER_STEPS_PER_ROLLOUT})" >&2
        exit 1
    fi
fi
# These token budgets are for full sequence length, so they must include prompt + response.
TRAIN_MAX_SEQ_LEN=$((MAX_PROMPT + MAX_RESP))
VAL_MAX_SEQ_LEN=$((MAX_PROMPT + VAL_MAX_RESP))
ROLLOUT_MAX_MODEL_LEN=${TRAIN_MAX_SEQ_LEN}
if [ "${VAL_MAX_SEQ_LEN}" -gt "${ROLLOUT_MAX_MODEL_LEN}" ]; then
    ROLLOUT_MAX_MODEL_LEN=${VAL_MAX_SEQ_LEN}
fi
if [ -z "${ACTOR_MAX_TOKEN_LEN}" ] || [ "${ACTOR_MAX_TOKEN_LEN}" = "None" ]; then
    ACTOR_MAX_TOKEN_LEN=${TRAIN_MAX_SEQ_LEN}
fi
LOGPROB_MAX_RESP=${MAX_RESP}
if [ "${VAL_MAX_RESP}" -gt "${LOGPROB_MAX_RESP}" ]; then
    LOGPROB_MAX_RESP=${VAL_MAX_RESP}
fi
LOGPROB_MAX_SEQ_LEN=$((MAX_PROMPT + LOGPROB_MAX_RESP))
ROLLOUT_MAX_BATCHED_TOKENS=${VAL_MAX_SEQ_LEN}

USE_SWANLAB_LOWER=$(echo "$USE_SWANLAB" | tr '[:upper:]' '[:lower:]')
USE_WANDB_LOWER=$(echo "$USE_WANDB" | tr '[:upper:]' '[:lower:]')
USE_LORA_LOWER=$(echo "$USE_LORA" | tr '[:upper:]' '[:lower:]')
USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "seed must be a non-negative integer, got '${SEED}' in ${CONFIG}" >&2
    exit 1
fi

# PYTHONHASHSEED must be present before Python/Ray workers start.  The remaining
# variables select deterministic CUDA/FlashAttention implementations when the
# experiment requests full determinism.
export PYTHONHASHSEED=${SEED}
FULL_DETERMINISM_LOWER=$(echo "$FULL_DETERMINISM" | tr '[:upper:]' '[:lower:]')
if [ "$FULL_DETERMINISM_LOWER" = "true" ]; then
    export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-":16:8"}
    export FLASH_ATTENTION_DETERMINISTIC=1
fi

case "$METHOD" in
    grpo|e2opsd|eopd|opsd|sdpo|rlsd)
        ;;
    *)
        echo "Invalid method '${METHOD}' in ${CONFIG}. Allowed values: e2opsd, eopd, grpo, opsd, sdpo, rlsd" >&2
        exit 1
        ;;
esac

if [ -z "$EXPERIMENT_NAME" ] || [ "$EXPERIMENT_NAME" = "None" ]; then
    EXPERIMENT_NAME="${METHOD}"
fi

if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" = "None" ]; then
    OUTPUT_DIR="./outputs/${PROJECT}/${EXPERIMENT_NAME}"
fi

LOGGER_ITEMS='"console","tensorboard","file"'
if [ "$USE_SWANLAB_LOWER" = "true" ]; then
    LOGGER_ITEMS="${LOGGER_ITEMS},\"swanlab\""
fi
if [ "$USE_WANDB_LOWER" = "true" ]; then
    LOGGER_ITEMS="${LOGGER_ITEMS},\"wandb\""
fi
LOGGER="[${LOGGER_ITEMS}]"

REWARD_ARGS=()
if [ -n "$REWARD_MANAGER_NAME" ] && [ "$REWARD_MANAGER_NAME" != "None" ]; then
    REWARD_ARGS+=(reward.reward_manager.name=${REWARD_MANAGER_NAME})
fi
if [ "$USE_CUSTOM_REWARD_FUNCTION_LOWER" = "true" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_PATH" ] && [ "$CUSTOM_REWARD_FUNCTION_PATH" != "None" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_NAME" ] && [ "$CUSTOM_REWARD_FUNCTION_NAME" != "None" ]; then
    REWARD_ARGS+=(
        reward.custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}
        reward.custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME}
    )
fi

if [ "$USE_SWANLAB_LOWER" = "true" ] && [ -z "${SWANLAB_API_KEY}" ]; then
    echo "Warning: SWANLAB_API_KEY is empty; swanlab logging requires existing login state or an exported key" >&2
fi

LORA_ARGS=(
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.lora_rank=0
)
if [ "$USE_LORA_LOWER" = "true" ]; then
    LORA_ARGS=(
        actor_rollout_ref.model.lora.merge=True
        actor_rollout_ref.model.lora_rank=${LORA_R}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    )
fi

HYDRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        resume_dir=*|+resume_dir=*)
            RESUME_DIR="${arg#*=}"
            ;;
        swanlab_run_id=*|+swanlab_run_id=*)
            SWANLAB_RUN_ID_OVERRIDE="${arg#*=}"
            ;;
        train_dataset=*|+train_dataset=*)
            TRAIN_DATASET="${arg#*=}"
            ;;
        val_dataset=*|+val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        data_dir=*|+data_dir=*)
            DATA_ROOT="${arg#*=}"
            ;;
        train_files=*|+train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        val_files=*|+val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        privileged_text_mode=*|+privileged_text_mode=*)
            PRIVILEGED_TEXT_MODE="${arg#*=}"
            ;;
        val_enable_thinking=*|+val_enable_thinking=*)
            VAL_ENABLE_THINKING="${arg#*=}"
            ;;
        reward_manager_name=*|+reward_manager_name=*)
            REWARD_MANAGER_NAME="${arg#*=}"
            ;;
        use_custom_reward_function=*|+use_custom_reward_function=*)
            USE_CUSTOM_REWARD_FUNCTION="${arg#*=}"
            USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')
            ;;
        custom_reward_function_path=*|+custom_reward_function_path=*)
            CUSTOM_REWARD_FUNCTION_PATH="${arg#*=}"
            ;;
        custom_reward_function_name=*|+custom_reward_function_name=*)
            CUSTOM_REWARD_FUNCTION_NAME="${arg#*=}"
            ;;
        data.train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        data.val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        *)
            HYDRA_ARGS+=("$arg")
            ;;
    esac
done

PARQUET_ROOT="${DATA_ROOT%/}/verl"

normalize_dataset_spec() {
    local spec="$1"
    spec="$(echo "$spec" | tr -d '[:space:]')"
    spec="${spec#,}"
    spec="${spec%,}"
    while [[ "$spec" == *",,"* ]]; do
        spec="${spec//,,/,}"
    done
    printf '%s' "$spec"
}

dataset_spec_key() {
    local spec
    spec="$(normalize_dataset_spec "$1")"
    printf '%s' "${spec//,/+}"
}

normalize_privileged_text_mode() {
    local mode="$1"
    mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$mode" in
        solution+answer|solution_and_answer)
            mode="solution_answer"
            ;;
        answer|answer_only)
            mode="answer_only"
            ;;
    esac
    printf '%s' "$mode"
}

ensure_dataset_parquet() {
    local split="$1"
    local dataset_name="$2"
    local target_path="$3"

    if [ -f "$target_path" ]; then
        return 0
    fi
    echo "Missing ${split} parquet: ${target_path}" >&2
    echo "Run: python scripts/download_data.py --dataset ${dataset_name} --split ${split}" >&2
    exit 1
}

if [ -z "$TRAIN_FILES" ] || [ "$TRAIN_FILES" = "None" ]; then
    TRAIN_FILES="${PARQUET_ROOT}/${TRAIN_DATASET}/train.parquet"
    ensure_dataset_parquet train "$TRAIN_DATASET" "$TRAIN_FILES"
fi

if [ -z "$VAL_FILES" ] || [ "$VAL_FILES" = "None" ]; then
    VAL_DATASET="$(normalize_dataset_spec "$VAL_DATASET")"
    VAL_FILES="${PARQUET_ROOT}/$(dataset_spec_key "$VAL_DATASET")/val.parquet"
    ensure_dataset_parquet val "$VAL_DATASET" "$VAL_FILES"
fi

PRIVILEGED_TEXT_MODE="$(normalize_privileged_text_mode "$PRIVILEGED_TEXT_MODE")"

resolve_abs_path() {
    python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

discover_swanlab_run_id() {
    python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
experiment_name = sys.argv[2]

record_path = run_dir / "swanlab_run_id.txt"
if record_path.exists():
    run_id = record_path.read_text(encoding="utf-8").strip()
    if run_id:
        print(run_id)
        raise SystemExit(0)

swanlog_root = Path("swanlog")
if not swanlog_root.exists():
    raise SystemExit(0)

candidates = sorted(swanlog_root.glob("run-*/backup.swanlab"), key=lambda path: path.stat().st_mtime, reverse=True)
for backup_path in candidates:
    try:
        with backup_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(32):
                line = handle.readline()
                if not line:
                    break
                json_start = line.find("{")
                if json_start < 0:
                    continue
                payload = json.loads(line[json_start:])
                if payload.get("model_type") != "Experiment":
                    continue
                data = payload.get("data", {})
                if data.get("name") == experiment_name and data.get("id"):
                    print(data["id"])
                    raise SystemExit(0)
                break
    except Exception:
        continue
PY
}

TEE_ARGS=()
RESUME_ARGS=()

if [ -n "$RESUME_DIR" ] && [ "$RESUME_DIR" != "None" ]; then
    DIR="$(resolve_abs_path "${RESUME_DIR%/}")"
    if [ ! -d "$DIR" ]; then
        echo "Resume directory does not exist: $DIR" >&2
        exit 1
    fi
    if [ ! -f "$DIR/latest_checkpointed_iteration.txt" ]; then
        echo "Resume directory is missing latest_checkpointed_iteration.txt: $DIR" >&2
        exit 1
    fi

    RUN_TAG="$(basename "$DIR")"
    if [ -n "$CONFIG_RUN_TAG" ] && [ "$CONFIG_RUN_TAG" != "None" ] && [ "$RUN_TAG" != "$CONFIG_RUN_TAG" ]; then
        echo "Resume directory tag '${RUN_TAG}' does not match config run_tag '${CONFIG_RUN_TAG}': $DIR" >&2
        exit 1
    fi
    EXP="${EXPERIMENT_NAME}-${RUN_TAG}"
    TEE_ARGS=(-a)
    RESUME_ARGS+=(trainer.resume_mode=auto)
    export VERL_FILE_LOGGER_APPEND=1

    if [ "$USE_SWANLAB_LOWER" = "true" ]; then
        export SWANLAB_RUN_ID_RECORD_FILE="${DIR}/swanlab_run_id.txt"
        if [ -n "$SWANLAB_RUN_ID_OVERRIDE" ] && [ "$SWANLAB_RUN_ID_OVERRIDE" != "None" ]; then
            export SWANLAB_RUN_ID="$SWANLAB_RUN_ID_OVERRIDE"
        else
            SWANLAB_RUN_ID_FOUND="$(discover_swanlab_run_id "$DIR" "$EXP")"
            if [ -z "$SWANLAB_RUN_ID_FOUND" ]; then
                echo "Failed to discover SwanLab run id for ${EXP}. Pass swanlab_run_id=<run_id>." >&2
                exit 1
            fi
            export SWANLAB_RUN_ID="$SWANLAB_RUN_ID_FOUND"
            printf '%s\n' "$SWANLAB_RUN_ID_FOUND" > "${DIR}/swanlab_run_id.txt"
        fi
        export SWANLAB_RESUME=must
    else
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        unset SWANLAB_RUN_ID_RECORD_FILE
    fi
else
    if [ -n "$CONFIG_RUN_TAG" ] && [ "$CONFIG_RUN_TAG" != "None" ]; then
        if [[ ! "$CONFIG_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            echo "Invalid run_tag '${CONFIG_RUN_TAG}'. Use letters, digits, '.', '_' or '-', starting with a letter or digit." >&2
            exit 1
        fi
        RUN_TAG="$CONFIG_RUN_TAG"
    else
        RUN_TAG=$(date +%Y%m%d.%H%M%S)
    fi
    EXP="${EXPERIMENT_NAME}-${RUN_TAG}"
    DIR="$(resolve_abs_path "${OUTPUT_DIR%/}/${RUN_TAG}")"
    if [ -e "$DIR" ]; then
        echo "Run directory already exists; refusing to overwrite: $DIR" >&2
        echo "To continue that run, pass resume_dir=$DIR explicitly." >&2
        exit 1
    fi
    mkdir -p "$DIR"
    cp "$CONFIG" "$DIR/config.yaml"
    unset VERL_FILE_LOGGER_APPEND
    if [ "$USE_SWANLAB_LOWER" = "true" ]; then
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        export SWANLAB_RUN_ID_RECORD_FILE="${DIR}/swanlab_run_id.txt"
    else
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        unset SWANLAB_RUN_ID_RECORD_FILE
    fi
fi

mkdir -p "$DIR"
if [ "$(resolve_abs_path "$CONFIG")" != "$(resolve_abs_path "$DIR/config.yaml")" ]; then
    cp "$CONFIG" "$DIR/config.yaml"
fi
export TENSORBOARD_DIR="${DIR}/tensorboard_log/"
export VERL_FILE_LOGGER_PATH="${DIR}/metrics.jsonl"

if [ "$USE_WANDB_LOWER" = "true" ]; then
    # Save tracking data locally.
    export WANDB_MODE=offline
    export WANDB_DIR="$DIR"
    export WANDB_PROJECT="$PROJECT"
    export WANDB_NAME="$EXP"
    export WANDB_DISABLE_CODE=true
    unset WANDB_API_KEY

    WANDB_RUN_ID_FILE="${DIR}/wandb_run_id.txt"
    if [ -n "$RESUME_DIR" ] && [ "$RESUME_DIR" != "None" ]; then
        if [ ! -s "$WANDB_RUN_ID_FILE" ]; then
            echo "Resume directory is missing a non-empty W&B run id file: $WANDB_RUN_ID_FILE" >&2
            exit 1
        fi
        WANDB_RUN_ID=$(tr -d '[:space:]' < "$WANDB_RUN_ID_FILE")
    else
        WANDB_RUN_ID=$(python3 -c 'import wandb; print(wandb.util.generate_id())')
        if [ -z "$WANDB_RUN_ID" ]; then
            echo "Failed to generate a W&B run id." >&2
            exit 1
        fi
        printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
    fi
    if [[ ! "$WANDB_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid W&B run id in $WANDB_RUN_ID_FILE" >&2
        exit 1
    fi
    export WANDB_RUN_ID
    export WANDB_RESUME=allow
else
    unset WANDB_MODE WANDB_DIR WANDB_PROJECT WANDB_NAME WANDB_RUN_ID WANDB_RESUME
fi

# --- Build algorithm-specific overrides from YAML ---
ALGO=""
for key in eopd_entropy_threshold eopd_fkl_coef beta jsd_token_clip teacher_mode teacher_temperature \
           full_logit_distill distill_add_tail alpha top_k_distill is_clip ema_decay \
           epsilon epsilon_w tau lam lam_decay_steps teacher_sync_interval; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ALGO="$ALGO +actor_rollout_ref.actor.policy_loss.${key}=${val}"
done
if [ "$METHOD" != "grpo" ]; then
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.privileged_text_mode=${PRIVILEGED_TEXT_MODE}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.teacher_enable_thinking=${TEACHER_ENABLE_THINKING}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.thinking_system_prompt=${THINKING_SYSTEM_PROMPT}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt=${EGT}"
    if [ "$METHOD" = "sdpo" ]; then
        for key in privileged_solution_source sdpo_success_reward_threshold \
                   sdpo_exclude_self_success sdpo_strip_sibling_thinking; do
            val=$(Y "$key" "")
            [ -n "$val" ] && [ "$val" != "None" ] \
                && ALGO="$ALGO +actor_rollout_ref.rollout.custom.${key}=${val}"
        done
    fi
    if [ "$(printf '%s' "$EGT" | tr 'A-Z' 'a-z')" = "true" ]; then
        ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt_neighbors_path=${EGT_NEIGHBORS_PATH}"
        ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt_max_solutions_per_question=${EGT_MAX_SOLUTIONS}"
        ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt_min_hit_fraction=${EGT_MIN_HIT_FRACTION}"
        ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt_problem_char_budget=${EGT_PROBLEM_CHAR_BUDGET}"
        ALGO="$ALGO +actor_rollout_ref.rollout.custom.egt_solution_char_budget=${EGT_SOLUTION_CHAR_BUDGET}"
    fi
fi

ROLLOUT_CORR=""
for key in rollout_is rollout_is_threshold rollout_is_batch_normalize rollout_rs rollout_rs_threshold; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ROLLOUT_CORR="$ROLLOUT_CORR algorithm.rollout_correction.${key}=${val}"
done

TRAINING_STEP_ARGS=()
if [ -n "$TOTAL_TRAINING_STEPS" ] && [ "$TOTAL_TRAINING_STEPS" != "None" ]; then
    TRAINING_STEP_ARGS+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

MODEL_OVERRIDE=""
if [ -n "$OVERRIDE_MAX_POSITION_EMBEDDINGS" ] && [ "$OVERRIDE_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.max_position_embeddings=${OVERRIDE_MAX_POSITION_EMBEDDINGS}"
fi
if [ -n "$ROPE_SCALING_TYPE" ] && [ "$ROPE_SCALING_TYPE" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.type=${ROPE_SCALING_TYPE}"
fi
if [ -n "$ROPE_SCALING_FACTOR" ] && [ "$ROPE_SCALING_FACTOR" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.factor=${ROPE_SCALING_FACTOR}"
fi
if [ -n "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" ] && [ "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=${ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
fi

# Select the training entry and policy loss for each method.
LOSS_MODE=$METHOD
if [ "$METHOD" = "grpo" ]; then
    LOSS_MODE=vanilla
fi
MAIN=src.self_distill_main
case "$METHOD" in
    grpo) MAIN=verl.trainer.main_ppo ;;
    eopd) MAIN=src.eopd_main ;;
    rlsd) MAIN=src.rlsd_baselines_main ;;
esac

WORKER_IMPL=$(Y use_legacy_worker_impl "")
if [ -z "$WORKER_IMPL" ] || [ "$WORKER_IMPL" = "None" ]; then
    case "$METHOD" in
        grpo) WORKER_IMPL=disable ;;
        *)        WORKER_IMPL=enable ;;
    esac
fi

# Preserve each baseline's KL setting; E²-OPSD inherits OPSD's zero coefficient.
if [ -n "$KL_LOSS_COEF" ] && [ "$KL_LOSS_COEF" != "None" ]; then
    EFFECTIVE_KL_LOSS_COEF=$KL_LOSS_COEF
else
    case "$METHOD" in
        e2opsd|eopd|rlsd|sdpo|opsd) EFFECTIVE_KL_LOSS_COEF=0 ;;
        *) EFFECTIVE_KL_LOSS_COEF=0.001 ;;
    esac
fi

python3 -m $MAIN \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${RESUME_ARGS[@]}" \
    trainer.val_before_train=${VAL_BEFORE} \
    trainer.val_only=${VAL_ONLY} \
    trainer.training_step_unit=${TRAINING_STEP_UNIT} \
    +trainer.seed=${SEED} \
    trainer.use_legacy_worker_impl=${WORKER_IMPL} \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    trainer.project_name=${PROJECT} \
    trainer.experiment_name=${EXP} \
    trainer.default_local_dir=${DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_STEPS} \
    trainer.test_freq=${EVAL_STEPS} \
    trainer.save_best_only=${SAVE_BEST_ONLY} \
    trainer.save_best_only_keep_latest=${SAVE_BEST_ONLY_KEEP_LATEST} \
    trainer.save_train_rollouts=${SAVE_TRAIN_ROLLOUTS} \
    trainer.checkpoint_retention_mode=${CHECKPOINT_RETENTION_MODE} \
    trainer.eval_compute_entropy=${EVAL_COMPUTE_ENTROPY} \
    trainer.total_epochs=${NUM_EPOCHS} \
    "${TRAINING_STEP_ARGS[@]}" \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${ROLLOUT_BATCH} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.train_max_samples=${MAX_TRAIN} \
    data.seed=${SEED} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    +data.apply_chat_template_kwargs.enable_thinking=${STUDENT_ENABLE_THINKING} \
    +data.val_apply_chat_template_kwargs.enable_thinking=${VAL_ENABLE_THINKING} \
    +data.thinking_system_prompt=${THINKING_SYSTEM_PROMPT} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP} \
    actor_rollout_ref.actor.grad_clip=${MAX_GRAD_NORM} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${EFFECTIVE_KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=${ENTROPY_FROM_LOGITS_WITH_CHUNKING} \
    actor_rollout_ref.actor.entropy_chunk_size=${ENTROPY_CHUNK_SIZE} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.seed=${SEED} \
    actor_rollout_ref.actor.fsdp_config.full_determinism=${FULL_DETERMINISM} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    +actor_rollout_ref.rollout.seed=${SEED} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE} \
    +actor_rollout_ref.rollout.enable_sleep_mode=${ENABLE_SLEEP_MODE} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS} \
    +actor_rollout_ref.rollout.custom.val_response_length=${VAL_MAX_RESP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    +actor_rollout_ref.rollout.val_kwargs.seed=${SEED} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.seed=${SEED} \
    actor_rollout_ref.ref.fsdp_config.full_determinism=${FULL_DETERMINISM} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    critic.data_loader_seed=${SEED} \
    critic.fsdp.seed=${SEED} \
    critic.fsdp.full_determinism=${FULL_DETERMINISM} \
    "${REWARD_ARGS[@]}" \
    ${MODEL_OVERRIDE} \
    ${ALGO} \
    ${ROLLOUT_CORR} \
    "${HYDRA_ARGS[@]}" \
    2>&1 | tee "${TEE_ARGS[@]}" "${DIR}/train_log.txt"
