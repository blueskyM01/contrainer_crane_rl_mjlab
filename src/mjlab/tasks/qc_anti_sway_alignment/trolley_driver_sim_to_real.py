from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media

# Prefer headless offscreen rendering on machines without DISPLAY.
if not os.environ.get("DISPLAY"):
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["PYOPENGL_PLATFORM"] = "egl"
else:
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.qc_anti_sway_alignment.qc_anti_sway_alignment_constants import (
  QC_ANTI_SWAY_ALIGNMENT_XML,
)


def _parse_command_dt_from_filename(path: Path) -> float | None:
  """Parse command dt from file names like '*_dt0p05.csv' or '*_dt0.05.csv'."""
  match = re.search(r"_dt([0-9]+(?:[p.][0-9]+)?)", path.stem)
  if match is None:
    return None
  token = match.group(1).replace("p", ".")
  try:
    value = float(token)
  except ValueError:
    return None
  return value if value > 0.0 else None


def _load_sine_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
  data = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=np.float64)
  if data.size == 0:
    raise ValueError(f"Input sine curve has no data rows: {csv_path}")
  if data.dtype.names is None or "step" not in data.dtype.names or "value" not in data.dtype.names:
    raise ValueError("Input sine curve CSV must contain columns: step,value")

  steps = np.atleast_1d(np.asarray(data["step"], dtype=np.int64))
  values = np.atleast_1d(np.asarray(data["value"], dtype=np.float64))
  if steps.shape != values.shape:
    raise ValueError(f"Shape mismatch: step{steps.shape} vs value{values.shape}")
  return steps, values


def _build_video_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
  """Create a camera that matches qc_anti_sway_alignment viewer style.

  Target style from env config:
  - origin_type: ASSET_BODY
  - body_name: gantry
  - distance: 4.0
  - elevation: -5.0
  - azimuth: 0.0
  """
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)

  gantry_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gantry")
  if gantry_body_id >= 0:
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    camera.trackbodyid = int(gantry_body_id)
    camera.fixedcamid = -1
    camera.lookat[:] = data.xpos[gantry_body_id]
  else:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    camera.trackbodyid = -1
    camera.fixedcamid = -1

  camera.distance = 4.0
  camera.elevation = -5.0
  camera.azimuth = 0.0
  return camera


def run_trolley_driver_sim_to_real(input_sine_curve: Path, output_dir: Path) -> tuple[Path, Path, Path]:
  if not input_sine_curve.is_file():
    raise FileNotFoundError(f"Input sine curve not found: {input_sine_curve}")

  output_dir.mkdir(parents=True, exist_ok=True)
  steps, commands = _load_sine_curve(input_sine_curve)

  model = mujoco.MjModel.from_xml_path(str(QC_ANTI_SWAY_ALIGNMENT_XML))
  data = mujoco.MjData(model)

  trolley_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "trolley_driver")
  if trolley_act_id < 0:
    raise ValueError("Actuator 'trolley_driver' not found in XML")

  trolley_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trolley_joint")
  if trolley_joint_id < 0:
    raise ValueError("Joint 'trolley_joint' not found in XML")

  hoist_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hoist_driver")
  if hoist_act_id >= 0:
    data.ctrl[hoist_act_id] = 0.0

  sim_dt = float(model.opt.timestep)
  command_dt = _parse_command_dt_from_filename(input_sine_curve) or 0.05
  steps_per_command = max(1, int(round(command_dt / sim_dt)))

  qpos_adr = int(model.jnt_qposadr[trolley_joint_id])
  qvel_adr = int(model.jnt_dofadr[trolley_joint_id])

  # Render one frame per command sample.
  render_fps = max(1, int(round(1.0 / command_dt)))
  max_off_width = int(getattr(model.vis.global_, "offwidth", 640))
  max_off_height = int(getattr(model.vis.global_, "offheight", 480))
  render_width = min(1280, max(1, max_off_width))
  render_height = min(720, max(1, max_off_height))
  renderer = mujoco.Renderer(model, width=render_width, height=render_height)
  mujoco.mj_forward(model, data)
  video_camera = _build_video_camera(model, data)
  frames: list[np.ndarray] = []

  trolley_pos = np.zeros_like(commands)
  trolley_vel = np.zeros_like(commands)
  sim_time = np.zeros_like(commands)

  for i, cmd in enumerate(commands):
    data.ctrl[trolley_act_id] = float(cmd)
    for _ in range(steps_per_command):
      mujoco.mj_step(model, data)

    trolley_pos[i] = float(data.qpos[qpos_adr])
    trolley_vel[i] = float(data.qvel[qvel_adr])
    sim_time[i] = float(data.time)

    renderer.update_scene(data, camera=video_camera)
    frames.append(renderer.render().copy())

  renderer.close()

  output_csv = output_dir / "trolley_driver_trolley_joint_curve.csv"
  output_png = output_dir / "trolley_driver_trolley_joint_curve.png"
  output_mp4 = output_dir / "trolley_driver_trolley_joint_curve.mp4"

  out = np.column_stack((steps, sim_time, commands, trolley_pos, trolley_vel))
  np.savetxt(
    str(output_csv),
    out,
    delimiter=",",
    header="step,sim_time_s,trolley_driver_cmd,trolley_joint_pos,trolley_joint_vel",
    comments="",
    fmt=["%d", "%.8f", "%.10f", "%.10f", "%.10f"],
  )

  fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
  axes[0].plot(steps, commands, label="trolley_driver_cmd", color="tab:orange", linewidth=1.8)
  axes[0].plot(steps, trolley_pos, label="trolley_joint_pos", color="tab:blue", linewidth=1.8)
  axes[0].set_ylabel("Position")
  axes[0].set_title("Trolley Driver Command vs Trolley Joint Position")
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc="best")

  axes[1].plot(steps, trolley_vel, label="trolley_joint_vel", color="tab:green", linewidth=1.6)
  axes[1].set_xlabel("Step")
  axes[1].set_ylabel("Velocity")
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc="best")

  fig.tight_layout()
  fig.savefig(output_png, dpi=180)
  plt.close(fig)

  media.write_video(str(output_mp4), frames, fps=render_fps)

  return output_csv, output_png, output_mp4


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Simulate qc_anti_sway_alignment trolley_driver with an input sine curve and "
      "export trolley_joint response as CSV, PNG, and MP4."
    )
  )
  parser.add_argument(
    "--input_sine_curve",
    type=Path,
    required=True,
    help="Path to input sine curve CSV (columns: step,value).",
  )
  parser.add_argument(
    "--output_dir",
    type=Path,
    required=True,
    help="Directory for output CSV, PNG, and MP4.",
  )
  args = parser.parse_args()

  output_csv, output_png, output_mp4 = run_trolley_driver_sim_to_real(
    input_sine_curve=args.input_sine_curve,
    output_dir=args.output_dir,
  )
  print(f"[INFO] Saved trolley curve CSV: {output_csv}")
  print(f"[INFO] Saved trolley curve PNG: {output_png}")
  print(f"[INFO] Saved trolley simulation MP4: {output_mp4}")


if __name__ == "__main__":
  main()