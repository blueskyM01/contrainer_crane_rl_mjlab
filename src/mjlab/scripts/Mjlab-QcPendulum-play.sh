#!/usr/bin/env bash
set -euo pipefail

# Batch play/evaluate all checkpoints under a directory tree.
# Usage:
#   src/mjlab/scripts/Mjlab-QcPendulum-play.sh <PLAY_ROOT_DIR> <TROLLEY_TARGET>
# Example:
#   src/mjlab/scripts/Mjlab-QcPendulum-play.sh \
#     logs/rsl_rl/qc_pendulum/phase3/track_pos \
#     logs/rsl_rl/qc_pendulum/phase3/sine_curve_amp0p65_offset0p65_t10_dt0p05.csv

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <PLAY_ROOT_DIR> <TROLLEY_TARGET>"
  exit 1
fi

PLAY_ROOT_DIR="$1"
TROLLEY_TARGET="$2"

if [[ ! -d "${PLAY_ROOT_DIR}" ]]; then
  echo "[ERROR] PLAY_ROOT_DIR does not exist: ${PLAY_ROOT_DIR}"
  exit 1
fi
if [[ ! -f "${TROLLEY_TARGET}" ]]; then
  echo "[ERROR] TROLLEY_TARGET file does not exist: ${TROLLEY_TARGET}"
  exit 1
fi

VIEWER="${VIEWER:-viser}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
PLOT_WINDOW="${PLOT_WINDOW:-400}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
PYOPENGL_BACKEND="${PYOPENGL_BACKEND:-egl}"

mapfile -t CHECKPOINTS < <(find "${PLAY_ROOT_DIR}" -type f -name 'model_*.pt' | sort -V)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint files matching model_*.pt found under: ${PLAY_ROOT_DIR}"
  exit 1
fi

summary_csv="${PLAY_ROOT_DIR%/}/track_pos_eval_summary.csv"
echo "checkpoint_file,results_dir,state_action_csv,samples,mae,rmse,max_abs_error,bias,nrmse_range,play_status,eval_status" > "${summary_csv}"

echo "[INFO] Found ${#CHECKPOINTS[@]} checkpoints under: ${PLAY_ROOT_DIR}"

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_dir="$(dirname "${ckpt}")"
  ckpt_base="$(basename "${ckpt}")"
  model_id=""
  if [[ "${ckpt_base}" =~ model_([0-9]+)\.pt ]]; then
    model_id="${BASH_REMATCH[1]}"
  else
    echo "[WARN] Skip unrecognized checkpoint name: ${ckpt}"
    continue
  fi

  results_dir="${ckpt_dir}/results-${model_id}"
  mkdir -p "${results_dir}"

  echo "[INFO] Play checkpoint: ${ckpt}"
  set +e
  MUJOCO_GL="${MUJOCO_GL_BACKEND}" PYOPENGL_PLATFORM="${PYOPENGL_BACKEND}" \
    uv run play Mjlab-QcPendulum \
      --checkpoint-file "${ckpt}" \
      --trolley-target "${TROLLEY_TARGET}" \
      --viewer "${VIEWER}" \
      --video True \
      --video-length "${VIDEO_LENGTH}" \
      --video-height "${VIDEO_HEIGHT}" \
      --video-width "${VIDEO_WIDTH}" \
      --plot-window "${PLOT_WINDOW}" \
      --plot-state-action-curve False \
      --save-state-action-curve True \
      --video-output-dir "${results_dir}" \
      --trolley-output-dir "${results_dir}"
  play_status=$?
  set -e

  state_action_csv="${results_dir}/state_action_curves.csv"
  samples=""
  mae=""
  rmse=""
  max_abs_error=""
  bias=""
  nrmse_range=""
  eval_status=1

  if [[ ${play_status} -eq 0 && -f "${state_action_csv}" ]]; then
    set +e
    eval_output="$(uv run python -m mjlab.scripts.sine_curve_track_pos_eval --csv-path "${state_action_csv}" 2>&1)"
    eval_status=$?
    set -e

    echo "${eval_output}" > "${results_dir}/track_pos_eval.txt"

    if [[ ${eval_status} -eq 0 ]]; then
      samples="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] Samples/ {print $2; exit}')"
      mae="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] MAE/ {print $2; exit}')"
      rmse="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] RMSE/ {print $2; exit}')"
      max_abs_error="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] MaxAbsError/ {print $2; exit}')"
      bias="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] Bias/ {print $2; exit}')"
      nrmse_range="$(echo "${eval_output}" | awk -F': ' '/\[INFO\] NRMSE\(range\)/ {print $2; exit}')"
    fi
  else
    echo "[WARN] Play failed or CSV missing: ${ckpt}"
  fi

  echo "${ckpt},${results_dir},${state_action_csv},${samples},${mae},${rmse},${max_abs_error},${bias},${nrmse_range},${play_status},${eval_status}" >> "${summary_csv}"
  echo "[INFO] Finished model_${model_id}: play_status=${play_status}, eval_status=${eval_status}"
done

summary_png="${summary_csv%.csv}.png"
set +e
uv run python - "${summary_csv}" "${summary_png}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(value: str) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


summary_csv = Path(sys.argv[1])
summary_png = Path(sys.argv[2])

if not summary_csv.is_file():
  raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

labels: list[str] = []
mae: list[float] = []
rmse: list[float] = []
max_abs_error: list[float] = []
bias: list[float] = []
nrmse_range: list[float] = []

with summary_csv.open("r", encoding="utf-8", newline="") as f:
  reader = csv.DictReader(f)
  for row in reader:
    if row.get("eval_status", "") != "0":
      continue

    m = re.search(r"model_(\d+)\.pt$", row.get("checkpoint_file", ""))
    label = f"model_{m.group(1)}" if m else f"#{len(labels) + 1}"

    row_mae = _to_float(row.get("mae", ""))
    row_rmse = _to_float(row.get("rmse", ""))
    row_max = _to_float(row.get("max_abs_error", ""))
    row_bias = _to_float(row.get("bias", ""))
    row_nrmse = _to_float(row.get("nrmse_range", ""))
    if None in (row_mae, row_rmse, row_max, row_bias, row_nrmse):
      continue

    labels.append(label)
    mae.append(row_mae)
    rmse.append(row_rmse)
    max_abs_error.append(row_max)
    bias.append(row_bias)
    nrmse_range.append(row_nrmse)

if not labels:
  raise ValueError("No valid eval_status=0 rows with numeric metrics in summary CSV.")

x = list(range(len(labels)))
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(x, mae, marker="o", label="MAE")
axes[0].plot(x, rmse, marker="o", label="RMSE")
axes[0].plot(x, max_abs_error, marker="o", label="MaxAbsError")
axes[0].set_ylabel("Position Error")
axes[0].set_title("Track Position Evaluation Summary")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].plot(x, bias, marker="o", label="Bias")
axes[1].plot(x, nrmse_range, marker="o", label="NRMSE(range)")
axes[1].set_ylabel("Bias / Normalized Error")
axes[1].set_xlabel("Checkpoint")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper right")

axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=30, ha="right")

fig.tight_layout()
fig.savefig(summary_png, dpi=180)
plt.close(fig)

print(f"[INFO] Saved summary curve figure: {summary_png}")
PY
plot_status=$?
set -e

if [[ ${plot_status} -ne 0 ]]; then
  echo "[WARN] Failed to generate summary plot: ${summary_png}"
else
  echo "[INFO] Summary PNG: ${summary_png}"
fi

echo "[INFO] Batch play + evaluation finished."
echo "[INFO] Summary CSV: ${summary_csv}"
