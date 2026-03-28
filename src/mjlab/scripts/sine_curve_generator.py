"""Generate a single-cycle sine curve and save to CSV.

The generated curve maps time in [0, duration] linearly to phase in [0, 2*pi].
Output CSV has two columns:
  1) step  : integer step index starting from 0
  2) value : sine value at this step
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

import mjlab


@dataclass(frozen=True)
class SineCurveConfig:
  amplitude: float
  """Sine amplitude."""

  duration: float
  """Total time length. This maps to phase range [0, 2*pi]."""

  step_size: float
  """Time step size used to sample the curve."""

  output_csv: str
  """Output CSV file path."""

  offset: float = 0.0
  """Constant offset added to sine values."""

  phase: float = -0.5 * float(np.pi)
  """Phase shift in radians. Applied as sin(phase(t) + phase)."""


def generate_sine_curve(cfg: SineCurveConfig) -> None:
  if cfg.duration <= 0:
    raise ValueError(f"duration must be > 0, got {cfg.duration}")
  if cfg.step_size <= 0:
    raise ValueError(f"step_size must be > 0, got {cfg.step_size}")

  # Include both endpoints 0 and duration when possible.
  times = np.arange(0.0, cfg.duration + 0.5 * cfg.step_size, cfg.step_size)
  phases = 2.0 * np.pi * (times / cfg.duration)
  values = cfg.amplitude * np.sin(phases + cfg.phase) + cfg.offset

  steps = np.arange(times.shape[0], dtype=np.int32)
  output = np.column_stack((steps, values.astype(np.float64, copy=False)))

  output_path = Path(cfg.output_csv)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(
    output_path,
    output,
    delimiter=",",
    header="step,value",
    comments="",
    fmt=["%d", "%.10f"],
  )

  # Save a plot next to the CSV using the same stem name.
  png_path = output_path.with_suffix(".png")
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(times, values, color="tab:blue", linewidth=1.8)
  ax.set_title("Sine Curve")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Value")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(png_path, dpi=160)
  plt.close(fig)

  print(f"[INFO] Saved sine curve to: {output_path}")
  print(f"[INFO] Saved sine curve plot to: {png_path}")
  print(f"[INFO] Number of samples: {output.shape[0]}")


def main() -> None:
  cfg = tyro.cli(SineCurveConfig, config=mjlab.TYRO_FLAGS)
  generate_sine_curve(cfg)


if __name__ == "__main__":
  main()
