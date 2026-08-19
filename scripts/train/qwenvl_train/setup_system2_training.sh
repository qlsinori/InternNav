#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

# Torch, Transformers and the ABI-matched FlashAttention wheel are already in
# the project environment.  VLN JPG/PNG data do not require decord/torchcodec.
uv pip install --python "${PYTHON_BIN}" \
    'deepspeed==0.16.4'

uv pip check --python "${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import deepspeed, flash_attn, pyarrow, torch, transformers; print("System 2 training imports OK")'
