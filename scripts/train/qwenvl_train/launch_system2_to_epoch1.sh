#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspace/flow/work_space/InternNav
RUN_DIR=/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2-r2r-rxr-8cfg-4gpu-zero3-20260813_085909_UTC
BOOTSTRAP_LOG=${RUN_DIR}/logs/nohup_resume_to_epoch1.log

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/control"
nohup setsid "${PROJECT_ROOT}/scripts/train/qwenvl_train/run_system2_to_epoch1.sh" \
    </dev/null >>"${BOOTSTRAP_LOG}" 2>&1 &
LAUNCHER_PID=$!
printf '%s\n' "${LAUNCHER_PID}" >"${RUN_DIR}/control/launcher_resume_to_epoch1.pid"
printf 'detached epoch-1 launcher pid=%s\n' "${LAUNCHER_PID}"
