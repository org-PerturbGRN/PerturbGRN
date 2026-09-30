#!/usr/bin/env bash
# Run Mode 3: Biolord pseudo Perturb-seq generation followed by PerturbGRN
# driver inference.
#
# This script starts from a user-prepared merged AnnData, a cell/source
# embedding dictionary, and target-state scRNA-seq, then runs end-to-end:
#   1. train Biolord and generate pseudo Perturb-seq profiles
#   2. merge generated profiles with target-state scRNA-seq into PerturbGRN form
#   3. write a driver-mode config and run main/main.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Python executables. By default, resolve the documented conda environment
# names. Explicit paths can still be supplied for schedulers/non-interactive
# shells where `conda` is not on PATH.
resolve_conda_python() {
  local env_name="$1"
  conda run -n "${env_name}" python -c 'import sys; print(sys.executable)'
}

CPA_PYTHON="${CPA_PYTHON:-}"
PERTURBGRN_PYTHON="${PERTURBGRN_PYTHON:-}"

if [[ -z "${CPA_PYTHON}" || -z "${PERTURBGRN_PYTHON}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH. Activate conda or set CPA_PYTHON and PERTURBGRN_PYTHON explicitly." >&2
    exit 1
  fi
  CPA_PYTHON="${CPA_PYTHON:-$(resolve_conda_python perturbgrn_mode3)}"
  PERTURBGRN_PYTHON="${PERTURBGRN_PYTHON:-$(resolve_conda_python perturbgrn)}"
fi

if [[ ! -x "${CPA_PYTHON}" ]]; then
  echo "CPA_PYTHON is not executable: ${CPA_PYTHON}" >&2
  echo "Activate perturbgrn_mode3 or set CPA_PYTHON=/path/to/perturbgrn_mode3/bin/python and rerun." >&2
  exit 1
fi

if [[ ! -x "${PERTURBGRN_PYTHON}" ]]; then
  echo "PERTURBGRN_PYTHON is not executable: ${PERTURBGRN_PYTHON}" >&2
  echo "Set PERTURBGRN_PYTHON=/path/to/perturbgrn/bin/python and rerun." >&2
  exit 1
fi

# User-provided Mode 3 inputs.
MODE3_DIR="${MODE3_DIR:-Data/mode3/t2ctl}"
MERGED_ADATA="${MERGED_ADATA:-${MODE3_DIR}/adata_decoupled_generation_input.h5ad}"
CELL_EMBEDDING="${CELL_EMBEDDING:-${MODE3_DIR}/CellTypeEmbedding.pkl}"
TARGET_SCRNA="${TARGET_SCRNA:-${MODE3_DIR}/target_scrna.h5ad}"

# Target and output layout.
TARGET="${TARGET:-T}"
SCENARIO="${SCENARIO:-mode3_${TARGET}_biolord_driver}"
BIOLORD_OUTPUT_DIR="${BIOLORD_OUTPUT_DIR:-${MODE3_DIR}/biolord_outputs}"
BIOLORD_OUTPUT_TEMPLATE="${BIOLORD_OUTPUT_TEMPLATE:-}"
if [[ -z "${BIOLORD_OUTPUT_TEMPLATE}" ]]; then
  BIOLORD_OUTPUT_TEMPLATE='{target}/result_biolord.h5ad'
fi
GENERATED_ADATA="${GENERATED_ADATA:-${BIOLORD_OUTPUT_DIR}/${TARGET}/result_biolord.h5ad}"
PERTURBGRN_ADATA="${PERTURBGRN_ADATA:-${MODE3_DIR}/perturbgrn_input.h5ad}"
CELL_ORDER_JSON="${CELL_ORDER_JSON:-${MODE3_DIR}/cell_order.json}"
PERTURBGRN_OUTPUT_DIR="${PERTURBGRN_OUTPUT_DIR:-${MODE3_DIR}/perturbgrn_driver_output}"

# Training/runtime controls. Keep defaults close to the original scripts.
CUDA_VISIBLE_DEVICES_FOR_BIOLORD="${CUDA_VISIBLE_DEVICES_FOR_BIOLORD:-}"
BIOLORD_MAX_EPOCHS="${BIOLORD_MAX_EPOCHS:-200}"
BIOLORD_BATCH_SIZE="${BIOLORD_BATCH_SIZE:-128}"
TARGET_GENE_SOURCE="${TARGET_GENE_SOURCE:-var_names}"
INCLUDE_GROUPS="${INCLUDE_GROUPS:-}"
MARKER_FILE="${MARKER_FILE:-}"
SKIP_BIOLORD="${SKIP_BIOLORD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_PERTURBGRN="${SKIP_PERTURBGRN:-0}"

require_file() {
  local input_path="$1"
  if [[ ! -f "${input_path}" ]]; then
    echo "Required Mode 3 input not found: ${input_path}" >&2
    exit 1
  fi
}

if [[ "${SKIP_BIOLORD}" != "1" ]]; then
  require_file "${MERGED_ADATA}"
  require_file "${CELL_EMBEDDING}"
elif [[ "${SKIP_CONVERT}" != "1" ]]; then
  require_file "${GENERATED_ADATA}"
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  require_file "${TARGET_SCRNA}"
  if [[ -n "${MARKER_FILE}" ]]; then
    require_file "${MARKER_FILE}"
  fi
elif [[ "${SKIP_PERTURBGRN}" != "1" ]]; then
  require_file "${PERTURBGRN_ADATA}"
  require_file "${CELL_ORDER_JSON}"
fi

MAX_ITER="${MAX_ITER:-10000}"
K="${K:-1}"
POOL_SIZE="${POOL_SIZE:-30}"
DOWN_SAMPLE_SIZE="${DOWN_SAMPLE_SIZE:-0}"
SAVE_EVERY="${SAVE_EVERY:-100}"
FLUSH_EVERY="${FLUSH_EVERY:-10}"
RANDOM_STATE="${RANDOM_STATE:-0}"
DRIVER_STOP_THRESHOLD="${DRIVER_STOP_THRESHOLD:-0.005}"
DRIVER_STOP_WINDOW="${DRIVER_STOP_WINDOW:-3}"

mkdir -p "${MODE3_DIR}" "${BIOLORD_OUTPUT_DIR}" "${PERTURBGRN_OUTPUT_DIR}"

echo "[mode3] repo root: ${REPO_ROOT}"
echo "[mode3] merged AnnData: ${MERGED_ADATA}"
echo "[mode3] embedding: ${CELL_EMBEDDING}"
echo "[mode3] target scRNA-seq: ${TARGET_SCRNA}"
echo "[mode3] target: ${TARGET}"
if [[ -n "${MARKER_FILE}" ]]; then
  echo "[mode3] pseudotime root markers: ${MARKER_FILE}"
else
  echo "[mode3] pseudotime root markers: built-in T-cell markers"
fi

if [[ "${SKIP_BIOLORD}" != "1" ]]; then
  echo "[mode3:1] training Biolord and generating pseudo Perturb-seq"
  BIOLORD_CMD=(
    "${CPA_PYTHON}" main/train_biolord_pseudo_perturbseq.py
    --adata "${MERGED_ADATA}"
    --embedding "${CELL_EMBEDDING}"
    --targets "${TARGET}"
    --output-dir "${BIOLORD_OUTPUT_DIR}"
    --output-template "${BIOLORD_OUTPUT_TEMPLATE}"
    --max-epochs "${BIOLORD_MAX_EPOCHS}"
    --batch-size "${BIOLORD_BATCH_SIZE}"
  )
  if [[ -n "${CUDA_VISIBLE_DEVICES_FOR_BIOLORD}" ]]; then
    BIOLORD_CMD+=(--cuda-visible-devices "${CUDA_VISIBLE_DEVICES_FOR_BIOLORD}")
  fi
  "${BIOLORD_CMD[@]}"
else
  echo "[mode3:1] SKIP_BIOLORD=1; using existing ${GENERATED_ADATA}"
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  echo "[mode3:2] converting Biolord output into PerturbGRN input with the PerturbGRN environment"
  CONVERT_CMD=(
    "${PERTURBGRN_PYTHON}" main/prepare_perturbgrn_input_from_biolord.py
    --generated-adata "${GENERATED_ADATA}"
    --target-scrna "${TARGET_SCRNA}"
    --output-adata "${PERTURBGRN_ADATA}"
    --output-cell-order "${CELL_ORDER_JSON}"
    --target-gene-source "${TARGET_GENE_SOURCE}"
  )
  if [[ -n "${INCLUDE_GROUPS}" ]]; then
    CONVERT_CMD+=(--include-groups "${INCLUDE_GROUPS}")
  fi
  if [[ -n "${MARKER_FILE}" ]]; then
    CONVERT_CMD+=(--marker-file "${MARKER_FILE}")
  fi
  "${CONVERT_CMD[@]}"
else
  echo "[mode3:2] SKIP_CONVERT=1; using existing ${PERTURBGRN_ADATA} and ${CELL_ORDER_JSON}"
fi

echo "[mode3:3] writing driver-mode config"
CONFIG_PATH="main/config.py"
CONFIG_SNAPSHOT="${PERTURBGRN_OUTPUT_DIR}/config_mode3_driver.py"

"${PERTURBGRN_PYTHON}" - \
  "${SCENARIO}" \
  "${PERTURBGRN_ADATA}" \
  "${CELL_ORDER_JSON}" \
  "${PERTURBGRN_OUTPUT_DIR}" \
  "${DRIVER_STOP_THRESHOLD}" \
  "${DRIVER_STOP_WINDOW}" \
  "${MAX_ITER}" \
  "${K}" \
  "${POOL_SIZE}" \
  "${DOWN_SAMPLE_SIZE}" \
  "${SAVE_EVERY}" \
  "${FLUSH_EVERY}" \
  "${RANDOM_STATE}" \
  "${CONFIG_PATH}" \
  "${CONFIG_SNAPSHOT}" <<'PY'
from pathlib import Path
import sys

(
    scenario,
    adata_path,
    cell_order_file,
    output_dir,
    driver_stop_threshold,
    driver_stop_window,
    max_iter,
    k,
    pool_size,
    down_sample_size,
    save_every,
    flush_every,
    random_state,
    config_path,
    config_snapshot,
) = sys.argv[1:]

config_text = f'''"""Auto-generated Mode 3 driver config.

Generated by main/run_mode3_biolord_driver.sh.
"""

MODE = "driver"

SCENARIO = {scenario!r}
ADATA_PATH = {adata_path!r}
CELLORDER_FILE = {cell_order_file!r}
OUTPUT_DIR = {output_dir!r}

TARGET_PERT_LABEL = "unknown"
DRIVER_EARLY_STOP = True
DRIVER_STOP_THRESHOLD = {driver_stop_threshold}
DRIVER_STOP_WINDOW = {driver_stop_window}

SELECTED_TF_JSON = None
GROUND_TRUTH_PATH = None

MAX_ITER = {max_iter}
K = {k}
POOL_SIZE = {pool_size}
DOWN_SAMPLE_SIZE = {down_sample_size}
SAVE_EVERY = {save_every}
FLUSH_EVERY = {flush_every}
RANDOM_STATE = {random_state}
'''

Path(config_snapshot).parent.mkdir(parents=True, exist_ok=True)
Path(config_snapshot).write_text(config_text)
Path(config_path).write_text(config_text)
print(f"Wrote {config_path} and {config_snapshot}")
PY

if [[ "${SKIP_PERTURBGRN}" != "1" ]]; then
  echo "[mode3:3] running PerturbGRN driver mode"
  "${PERTURBGRN_PYTHON}" main/main.py
else
  echo "[mode3:3] SKIP_PERTURBGRN=1; config written but main/main.py not run"
fi

echo "[mode3] done"
