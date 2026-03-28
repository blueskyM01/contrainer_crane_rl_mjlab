"""Evaluate trolley position tracking error against target position from a CSV.

Expected CSV columns include:
  - trolley_pos
  - target_pos
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab


@dataclass(frozen=True)
class TrackPosEvalConfig:
  csv_path: str
  """Path to state_action_curves.csv."""


def evaluate_track_pos(cfg: TrackPosEvalConfig) -> None:
  csv_path = Path(cfg.csv_path)
  if not csv_path.is_file():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

  data = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=np.float64)
  if data.size == 0:
    raise ValueError(f"CSV has no rows: {csv_path}")

  required_cols = ("trolley_pos", "target_pos")
  if data.dtype.names is None or any(col not in data.dtype.names for col in required_cols):
    raise ValueError(
      f"CSV must contain columns {required_cols}, got: {data.dtype.names}"
    )

  trolley_pos = np.atleast_1d(np.asarray(data["trolley_pos"], dtype=np.float64))
  target_pos = np.atleast_1d(np.asarray(data["target_pos"], dtype=np.float64))

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

  print(f"[INFO] CSV: {csv_path}")
  print(f"[INFO] Samples: {sample_count}")
  print("[INFO] Error defined as: trolley_pos - target_pos")
  print(f"[INFO] MAE: {mae:.8f}")
  print(f"[INFO] RMSE: {rmse:.8f}")
  print(f"[INFO] MaxAbsError: {max_abs_err:.8f}")
  print(f"[INFO] Bias: {bias:.8f}")
  print(f"[INFO] NRMSE(range): {nrmse:.8f}")


def main() -> None:
  cfg = tyro.cli(TrackPosEvalConfig, config=mjlab.TYRO_FLAGS)
  evaluate_track_pos(cfg)


if __name__ == "__main__":
  main()
