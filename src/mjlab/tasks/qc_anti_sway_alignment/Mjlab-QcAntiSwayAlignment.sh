#!/usr/bin/env bash
set -euo pipefail

# Batch play/evaluate all checkpoints under a directory tree.
#
# Required args:
#   --batch-eval-dir  Root dir to scan for model_*.pt
#   --actual-csv-path Target CSV used as --trolley-target during play
#
# Example:
#   src/mjlab/tasks/qc_anti_sway_alignment/Mjlab-QcAntiSwayAlignment.sh \
#     --batch-eval-dir logs/rsl_rl/qc_anti_sway_alignment/trolley_pos_cmd_track \
#     --actual-csv-path logs/rsl_rl/qc_pendulum/phase3/sine_curve_amp0p65_offset0p65_t10_dt0p05.csv

BATCH_EVAL_DIR=""
ACTUAL_CSV_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-eval-dir)
      BATCH_EVAL_DIR="${2:-}"
      shift 2
      ;;
    --actual-csv-path)
      ACTUAL_CSV_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --batch-eval-dir <DIR> --actual-csv-path <CSV>"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 --batch-eval-dir <DIR> --actual-csv-path <CSV>"
      exit 1
      ;;
  esac
done

if [[ -z "${BATCH_EVAL_DIR}" || -z "${ACTUAL_CSV_PATH}" ]]; then
  echo "[ERROR] Missing required args."
  echo "Usage: $0 --batch-eval-dir <DIR> --actual-csv-path <CSV>"
  exit 1
fi

if [[ ! -d "${BATCH_EVAL_DIR}" ]]; then
  echo "[ERROR] --batch-eval-dir does not exist: ${BATCH_EVAL_DIR}"
  exit 1
fi

if [[ ! -f "${ACTUAL_CSV_PATH}" ]]; then
  echo "[ERROR] --actual-csv-path does not exist: ${ACTUAL_CSV_PATH}"
  exit 1
fi

VIEWER="${VIEWER:-viser}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
PLOT_WINDOW="${PLOT_WINDOW:-400}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
PYOPENGL_BACKEND="${PYOPENGL_BACKEND:-egl}"

mapfile -t CHECKPOINTS < <(find "${BATCH_EVAL_DIR}" -type f -name 'model_*.pt' | sort -V)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint files matching model_*.pt found under: ${BATCH_EVAL_DIR}"
  exit 1
fi

summary_csv="${BATCH_EVAL_DIR%/}/batch_eval_summary.csv"
echo "checkpoint_file,results_dir,state_action_csv,samples,mae,rmse,max_abs_error,bias,nrmse_range,play_status,eval_status" > "${summary_csv}"

echo "[INFO] Found ${#CHECKPOINTS[@]} checkpoints under: ${BATCH_EVAL_DIR}"

prev_ckpt_dir=""

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_dir="$(dirname "${ckpt}")"
  ckpt_base="$(basename "${ckpt}")"

  # Detect group boundary: the previous group just finished.
  if [[ -n "${prev_ckpt_dir}" && "${ckpt_dir}" != "${prev_ckpt_dir}" ]]; then
    echo "[INFO] Group '${prev_ckpt_dir##*/}' complete, generating intermediate plots..."
    set +e
    uv run python -m mjlab.tasks.qc_anti_sway_alignment.batch_eval_plot \
      --summary-csv "${summary_csv}"
    set -e
  fi
  prev_ckpt_dir="${ckpt_dir}"

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
    uv run play Mjlab-QcAntiSwayAlignment \
      --checkpoint-file "${ckpt}" \
      --trolley-target "${ACTUAL_CSV_PATH}" \
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
    eval_output="$(uv run python -m mjlab.tasks.qc_anti_sway_alignment.sine_curve_track_pos_eval --play-csv-path "${state_action_csv}" 2>&1)"
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
      echo "[INFO] Eval done: ${results_dir}/track_pos_eval.txt"
    else
      echo "[WARN] Eval failed for ${ckpt}. See ${results_dir}/track_pos_eval.txt"
    fi
  else
    echo "[WARN] Play failed or state_action_curves.csv missing for ${ckpt}"
  fi

  echo "\"${ckpt}\",\"${results_dir}\",\"${state_action_csv}\",\"${samples}\",\"${mae}\",\"${rmse}\",\"${max_abs_error}\",\"${bias}\",\"${nrmse_range}\",${play_status},${eval_status}" >> "${summary_csv}"
done

echo "[INFO] Batch evaluation completed."
echo "[INFO] Summary CSV: ${summary_csv}"

echo "[INFO] Generating plots..."
uv run python -m mjlab.tasks.qc_anti_sway_alignment.batch_eval_plot \
  --summary-csv "${summary_csv}"
