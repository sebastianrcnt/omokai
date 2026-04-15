#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv-rocm/bin/python"
CONFIG="${ROOT_DIR}/configs/rocm_unlimited.yaml"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/rocm_unlimited"
STATE_PATH="${CHECKPOINT_DIR}/trainer_state.pt"

mkdir -p "${ROOT_DIR}/logs" "${CHECKPOINT_DIR}"

if [[ -f "${STATE_PATH}" ]]; then
  exec "${PYTHON}" -u -m omokai.train --config "${CONFIG}" --resume "${STATE_PATH}"
else
  exec "${PYTHON}" -u -m omokai.train --config "${CONFIG}"
fi
