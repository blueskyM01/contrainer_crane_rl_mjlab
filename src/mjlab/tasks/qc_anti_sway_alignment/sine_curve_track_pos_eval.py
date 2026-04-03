"""Evaluate trolley tracking and sway-angle metrics from play output CSV.

Expected play CSV columns include:
  - trolley_pos
  - target_pos
  - spreader_sway_angle_deg (preferred) or spreader_sway_angle (radians)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab


@dataclass(frozen=True)
class TrackPosEvalConfig:
  play_csv_path: str
  """Path to play output CSV (e.g., state_action_curves.csv)."""


def _get_col(data: np.ndarray, candidates: tuple[str, ...], label: str) -> np.ndarray:
  if data.dtype.names is None:
    raise ValueError(f"{label} CSV has no named columns.")
  for name in candidates:
    if name in data.dtype.names:
      return np.atleast_1d(np.asarray(data[name], dtype=np.float64))
  raise ValueError(
    f"{label} CSV is missing required column candidates {candidates}, got: {data.dtype.names}"
  )


def evaluate_track_pos(cfg: TrackPosEvalConfig) -> None:
  play_csv_path = Path(cfg.play_csv_path)
  if not play_csv_path.is_file():
    raise FileNotFoundError(f"Play CSV file not found: {play_csv_path}")

  play_data = np.genfromtxt(
    str(play_csv_path), delimiter=",", names=True, dtype=np.float64
  )
  if play_data.size == 0:
    raise ValueError(f"Play CSV has no rows: {play_csv_path}")

  # Play output should contain both actual and target positions.
  trolley_pos = _get_col(play_data, ("trolley_pos",), "Play")
  target_pos = _get_col(play_data, ("target_pos",), "Play")

  # Prefer degree column; fallback to radians and convert for consistent reporting.
  sway_deg = None
  if play_data.dtype.names is not None and "spreader_sway_angle_deg" in play_data.dtype.names:
    sway_deg = _get_col(play_data, ("spreader_sway_angle_deg",), "Play")
  elif play_data.dtype.names is not None and "spreader_sway_angle" in play_data.dtype.names:
    sway_rad = _get_col(play_data, ("spreader_sway_angle",), "Play")
    sway_deg = np.degrees(sway_rad)

  if trolley_pos.shape != target_pos.shape:
    raise ValueError(
      "trolley_pos and target_pos shapes do not match: "
      f"{trolley_pos.shape} vs {target_pos.shape}"
    )

  err = trolley_pos - target_pos
  abs_err = np.abs(err)

  sample_count = int(err.size)
  mae = float(np.mean(abs_err))
  rmse = float(np.sqrt(np.mean(err ** 2)))
  max_abs_err = float(np.max(abs_err))
  bias = float(np.mean(err))

  target_range = float(np.max(target_pos) - np.min(target_pos))
  nrmse = rmse / target_range if target_range > 0 else float("nan")

  sway_abs_mae = float("nan")
  sway_rms = float("nan")
  sway_max_abs = float("nan")
  if sway_deg is not None:
    if sway_deg.shape != trolley_pos.shape:
      raise ValueError(
        "spreader_sway_angle shape does not match trolley_pos shape: "
        f"{sway_deg.shape} vs {trolley_pos.shape}"
      )
    sway_abs = np.abs(sway_deg)
    sway_abs_mae = float(np.mean(sway_abs))
    sway_rms = float(np.sqrt(np.mean(sway_deg ** 2)))
    sway_max_abs = float(np.max(sway_abs))

  metrics_output_path = play_csv_path.parent / "track_pos_eval.txt"
  lines = [
    f"[INFO] CSV: {play_csv_path}",
    f"[INFO] Samples: {sample_count}",
    "[INFO] Position error defined as: trolley_pos - target_pos",
    f"[INFO] MAE: {mae:.8f}",
    f"[INFO] RMSE: {rmse:.8f}",
    f"[INFO] MaxAbsError: {max_abs_err:.8f}",
    f"[INFO] Bias: {bias:.8f}",
    f"[INFO] NRMSE(range): {nrmse:.8f}",
    f"[INFO] SwayAbsMAE(deg): {sway_abs_mae:.8f}",
    f"[INFO] SwayRMS(deg): {sway_rms:.8f}",
    f"[INFO] SwayMaxAbs(deg): {sway_max_abs:.8f}",
  ]
  metrics_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  for line in lines:
    print(line)
  print(f"[INFO] Saved eval text: {metrics_output_path}")


def main() -> None:
  cfg = tyro.cli(TrackPosEvalConfig, config=mjlab.TYRO_FLAGS)
  evaluate_track_pos(cfg)


if __name__ == "__main__":
  main()
