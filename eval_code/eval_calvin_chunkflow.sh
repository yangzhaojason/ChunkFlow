#!/usr/bin/env bash
set -euo pipefail

: "${CALVIN_ROOT:?set CALVIN_ROOT to the CALVIN checkout}"
: "${CHUNKFLOW_CHECKPOINT:?set CHUNKFLOW_CHECKPOINT to a trained checkpoint}"
: "${CALVIN_DATASET:?set CALVIN_DATASET to the CALVIN dataset root}"
: "${CHUNKFLOW_CONFIG:?set CHUNKFLOW_CONFIG to the openpi training config name}"

CALVIN_ROOT="$(cd -- "${CALVIN_ROOT}" && pwd)"
CALVIN_DATASET="$(cd -- "${CALVIN_DATASET}" && pwd)"
CHUNKFLOW_CHECKPOINT="$(cd -- "${CHUNKFLOW_CHECKPOINT}" && pwd)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CALVIN_OUTPUT_DIR="${CALVIN_OUTPUT_DIR:-outputs/calvin}"
CALVIN_NUM_SEQUENCES="${CALVIN_NUM_SEQUENCES:-1000}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m eval_code.calvin_evaluate \
  --calvin-root "${CALVIN_ROOT}" \
  --dataset-path "${CALVIN_DATASET}" \
  --checkpoint-dir "${CHUNKFLOW_CHECKPOINT}" \
  --config-name "${CHUNKFLOW_CONFIG}" \
  --output-dir "${CALVIN_OUTPUT_DIR}" \
  --num-sequences "${CALVIN_NUM_SEQUENCES}" \
  "$@"
