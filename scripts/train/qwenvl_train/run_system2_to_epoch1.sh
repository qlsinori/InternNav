#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspace/flow/work_space/InternNav
RUN_DIR=/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2-r2r-rxr-8cfg-4gpu-zero3-20260813_085909_UTC
LOG_DIR=${RUN_DIR}/logs
CONTROL_DIR=${RUN_DIR}/control
TARGET_STEP=24488
SAVE_STEPS=500
GLOBAL_BATCH=128
COMBINED_LOG=${LOG_DIR}/train_resume_to_epoch1.log
BOOTSTRAP_LOG=${LOG_DIR}/nohup_resume_to_epoch1.log
TENSORBOARD_DIR=${RUN_DIR}/tensorboard/live_2913_to_24488
RUNNING_MARKER=${CONTROL_DIR}/EPOCH1_RUNNING
COMPLETE_MARKER=${CONTROL_DIR}/EPOCH1_COMPLETE
FAILED_MARKER=${CONTROL_DIR}/EPOCH1_FAILED
STOP_FILE=${CONTROL_DIR}/STOP_AFTER_CURRENT_STEP
RANK_LOG_DIR=${LOG_DIR}/torchrun_ranks_resume_to_epoch1

mkdir -p "${LOG_DIR}" "${CONTROL_DIR}" "${TENSORBOARD_DIR}" "${RANK_LOG_DIR}"
exec > >(tee -a "${COMBINED_LOG}") 2>&1

latest_complete_checkpoint() {
    local checkpoint
    while IFS= read -r checkpoint; do
        if [[ -f "${checkpoint}/trainer_state.json" && -f "${checkpoint}/model.safetensors.index.json" ]]; then
            printf '%s\n' "${checkpoint}"
        fi
    done < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V)
}

SOURCE_CHECKPOINT=$(latest_complete_checkpoint | tail -n 1)
if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    echo "No complete checkpoint found in ${RUN_DIR}" >&2
    exit 1
fi
START_STEP=$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/trainer_state.json")
START_EPOCH=$(jq -r '.epoch' "${SOURCE_CHECKPOINT}/trainer_state.json")
if (( START_STEP >= TARGET_STEP )); then
    echo "Latest checkpoint is already at step ${START_STEP}; target is ${TARGET_STEP}."
    rm -f "${RUNNING_MARKER}" "${FAILED_MARKER}"
    touch "${COMPLETE_MARKER}"
    exit 0
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | rg -q '[0-9]'; then
    echo "A GPU compute process is already active; refusing to overlap training." >&2
    nvidia-smi >&2
    exit 1
fi

rm -f "${FAILED_MARKER}" "${COMPLETE_MARKER}" "${STOP_FILE}"
touch "${RUNNING_MARKER}"
date -u +%s >"${CONTROL_DIR}/epoch1_start_epoch_seconds"

echo "RESUME_START_UTC=$(date -u --iso-8601=seconds)"
echo "RUN_ID=$(basename "${RUN_DIR}")"
echo "OUTPUT_DIR=${RUN_DIR}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
echo "START_STEP=${START_STEP}"
echo "START_EPOCH=${START_EPOCH}"
echo "TARGET_STEP=${TARGET_STEP}"
echo "TARGET_EPOCH=1.0"
echo "REMAINING_STEPS=$((TARGET_STEP - START_STEP))"
echo "GLOBAL_BATCH=${GLOBAL_BATCH} (micro=1 x GPUs=4 x grad_accum=32)"
echo "REMAINING_DECISION_SAMPLES=$(((TARGET_STEP - START_STEP) * GLOBAL_BATCH))"
echo "DEEPSPEED=ZeRO-3, no CPU optimizer/parameter offload"
echo "SAVE_STEPS=${SAVE_STEPS} (forced after trainer_state resume)"
echo "FORCE_FINAL_CHECKPOINT=1"
echo "SAVE_TOTAL_LIMIT=5"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "LR_SCHEDULER=resume scheduler state at step ${START_STEP} against the one-epoch target ${TARGET_STEP}"
hostname
id
nvidia-smi
df -h "${RUN_DIR}"

cd "${PROJECT_ROOT}"
git rev-parse HEAD >"${RUN_DIR}/git_commit_resume_to_epoch1.txt"
git status --short >"${RUN_DIR}/git_status_resume_to_epoch1.txt"
git diff --output="${RUN_DIR}/source_diff_resume_to_epoch1.patch"
cp scripts/train/qwenvl_train/train_system2_local.sh "${RUN_DIR}/train_system2_local_resume_to_epoch1.sh"
cp scripts/train/qwenvl_train/run_system2_to_epoch1.sh "${RUN_DIR}/run_system2_to_epoch1.sh"
cp internnav/trainer/internvla_n1_trainer.py "${RUN_DIR}/internvla_n1_trainer_resume_to_epoch1.py"

GPU_TELEMETRY=${LOG_DIR}/gpu_telemetry_resume_to_epoch1.csv
echo "timestamp_utc,index,memory_used_mib,memory_free_mib,utilization_gpu_pct,power_w,temperature_c" >"${GPU_TELEMETRY}"
monitor_gpu() {
    while true; do
        timestamp=$(date -u --iso-8601=seconds)
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | sed "s/^/${timestamp},/" >>"${GPU_TELEMETRY}" || true
        sleep 30
    done
}
monitor_gpu &
MONITOR_PID=$!
cleanup_monitor() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    SKIP_CUDA_CHECK=1 \
    NUM_TRAIN_EPOCHS=1.0 \
    MAX_STEPS="${TARGET_STEP}" \
    GRAD_ACCUM_STEPS=32 \
    SAVE_STRATEGY=steps \
    SAVE_STEPS="${SAVE_STEPS}" \
    INTERNNAV_RESUME_SAVE_STEPS="${SAVE_STEPS}" \
    INTERNNAV_FORCE_FINAL_CHECKPOINT=1 \
    DATALOADER_NUM_WORKERS=8 \
    REPORT_TO=tensorboard \
    LOGGING_DIR="${TENSORBOARD_DIR}" \
    SKIP_FINAL_MODEL_SAVE=False \
    OUTPUT_DIR="${RUN_DIR}" \
    DEEPSPEED_CONFIG="${RUN_DIR}/zero3_gpu.json" \
    TORCHRUN_BIN="${RUN_DIR}/torchrun_logged_resume_728_to_2912.sh" \
    TORCHRUN_LOG_DIR="${RANK_LOG_DIR}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    NCCL_DEBUG=WARN \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    bash scripts/train/qwenvl_train/train_system2_local.sh
TRAIN_RC=$?
set -e

cleanup_monitor
trap - EXIT

FINAL_STEP=$(jq -r '.global_step // -1' "${RUN_DIR}/trainer_state.json" 2>/dev/null || echo -1)
FINAL_EPOCH=$(jq -r '.epoch // -1' "${RUN_DIR}/trainer_state.json" 2>/dev/null || echo -1)
echo "RESUME_END_UTC=$(date -u --iso-8601=seconds)"
echo "TRAIN_EXIT_CODE=${TRAIN_RC}"
echo "FINAL_STEP=${FINAL_STEP}"
echo "FINAL_EPOCH=${FINAL_EPOCH}"
echo "FINAL_DISK_USAGE=$(du -sh "${RUN_DIR}" | cut -f1)"

if [[ ${TRAIN_RC} -eq 0 && "${FINAL_STEP}" == "${TARGET_STEP}" ]]; then
    mv "${RUNNING_MARKER}" "${COMPLETE_MARKER}"
    exit 0
fi

mv "${RUNNING_MARKER}" "${FAILED_MARKER}"
if [[ ${TRAIN_RC} -eq 0 ]]; then
    exit 97
fi
exit "${TRAIN_RC}"
