#!/usr/bin/env bash
set -euo pipefail

# Single-machine launcher for Stage A / System 2.  The upstream
# train_system2.sh remains the 8-node Slurm reference configuration.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${PROJECT_ROOT}/.venv/bin/torchrun}"

VLN_DATA_ROOT="${VLN_DATA_ROOT:-/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data}"
BASE_MODEL="${BASE_MODEL:-/mnt/cpfs/zbl-cpfs-new/Models/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
DATASETS="${VLN_DATASETS:-r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30}"
SCENE_IDS="${VLN_SCENE_IDS:-}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-tensorboard}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/tensorboard}"
SKIP_FINAL_MODEL_SAVE="${SKIP_FINAL_MODEL_SAVE:-False}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/scripts/train/qwenvl_train/zero3.json}"

for required_path in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${VLN_DATA_ROOT}" "${BASE_MODEL}" "${DEEPSPEED_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -c 'import importlib.util; names=("deepspeed", "flash_attn", "pyarrow", "transformers"); assert all(importlib.util.find_spec(name) for name in names)' >/dev/null; then
    echo "Training dependencies are incomplete. Run scripts/train/qwenvl_train/setup_system2_training.sh first." >&2
    exit 1
fi

if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
    CUDA_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
    if (( CUDA_COUNT < NPROC_PER_NODE )); then
        echo "Need ${NPROC_PER_NODE} visible CUDA devices, but PyTorch sees ${CUDA_COUNT}." >&2
        echo "Run this launcher inside a GPU instance and check CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${SCENE_IDS}" ]]; then
    EXTRA_ARGS+=(--vln_scene_ids "${SCENE_IDS}")
fi
if [[ "${MAX_STEPS}" != "-1" ]]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${PROJECT_ROOT}/internnav/trainer/internvla_n1_trainer.py" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${BASE_MODEL}" \
    --vln_data_root "${VLN_DATA_ROOT}" \
    --vln_dataset_use "${DATASETS}" \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 none \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_pixels 313600 \
    --min_pixels 3136 \
    --eval_strategy no \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --vision_tower_lr 5e-6 \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --run_name InternVLA-N1-System2 \
    --report_to "${REPORT_TO}" \
    --logging_dir "${LOGGING_DIR}" \
    --skip_final_model_save "${SKIP_FINAL_MODEL_SAVE}" \
    "${EXTRA_ARGS[@]}"
