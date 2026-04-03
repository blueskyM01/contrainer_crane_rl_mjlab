"""Generate a linear step curve and save to CSV.

The generated curve stays at 0 until ``step_time``, then ramps linearly to the
target amplitude over ``step_duration`` seconds, and stays at the amplitude
after the ramp finishes.
Output CSV has two columns:
  1) step  : integer step index starting from 0
  2) value : curve value at this step
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro

import mjlab


@dataclass(frozen=True)
class StepCurveConfig:
  amplitude: float
  """Step function amplitude."""

  duration: float
  """Total time length in seconds."""

  step_size: float
  """Time step size used to sample the curve."""

  output_csv: str
  """Output CSV file path."""

  step_time: Optional[float] = None
  """Time at which the linear step starts. If None, defaults to duration/2."""

  step_duration: float = 0.0
  """Duration of the linear ramp from 0 to amplitude. 0 means instant step."""


def generate_step_curve(cfg: StepCurveConfig) -> None:
  if cfg.duration <= 0:
    raise ValueError(f"duration must be > 0, got {cfg.duration}")
  if cfg.step_size <= 0:
    raise ValueError(f"step_size must be > 0, got {cfg.step_size}")
  if cfg.step_duration < 0:
    raise ValueError(f"step_duration must be >= 0, got {cfg.step_duration}")

  # Determine step time (default to halfway through duration).
  step_time = cfg.step_time if cfg.step_time is not None else cfg.duration / 2.0
  ramp_end_time = step_time + cfg.step_duration

  # Generate time samples: include both endpoints when possible.
  times = np.arange(0.0, cfg.duration + 0.5 * cfg.step_size, cfg.step_size)

  # Create linear step function: 0 before step_time, linear ramp during
  # step_duration, amplitude after ramp_end_time.
  if cfg.step_duration == 0.0:
    values = np.where(times >= step_time, cfg.amplitude, 0.0)
  else:
    ramp = np.clip((times - step_time) / cfg.step_duration, 0.0, 1.0)
    values = cfg.amplitude * ramp

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
  ax.set_title("Linear Step Curve")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Value")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(png_path, dpi=160)
  plt.close(fig)

  print(f"[INFO] Saved step curve to: {output_path}")
  print(f"[INFO] Saved step curve plot to: {png_path}")
  print(f"[INFO] Number of samples: {output.shape[0]}")
  print(
    f"[INFO] Step starts at t={step_time}s, ramp ends at t={ramp_end_time}s, "
    f"amplitude={cfg.amplitude}"
  )


def main() -> None:
  cfg = tyro.cli(StepCurveConfig, config=mjlab.TYRO_FLAGS)
  generate_step_curve(cfg)


if __name__ == "__main__":
  main()
