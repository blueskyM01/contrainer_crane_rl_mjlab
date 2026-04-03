"""Script to play RL agent with RSL-RL."""

import math
import os
import sys
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.lines import Line2D

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  video_output_dir: str | None = None
  """Optional output directory for recorded videos. Defaults under run log dir or local outputs/."""
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Task-specific parameters
  trolley_target: float | str | None = None
  """Target for trolley tasks: fixed value or CSV path (step,value)."""

  plot_state_action_curve: bool = False
  """Display trolley/pendulum state and action curves during play (env 0)."""
  save_state_action_curve: bool = False
  """Save trolley/pendulum state and action curves as .png and .csv after play (env 0)."""
  plot_window: int = 400
  """Number of recent steps to keep in the live plot window."""
  trolley_output_dir: str | None = None
  """Optional output directory for state/action curves. Defaults under run log dir or local outputs/."""
  action_ema_alpha: float = 0.0
  """EMA smoothing coefficient for actions during play (0=disabled, 0→1 = more smoothing).

  Applied as: a_smooth[t] = (1-alpha)*a_raw[t] + alpha*a_smooth[t-1].
  Typical values: 0.5 (mild), 0.8 (strong).
  """

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  supports_trolley_curves = task_id in {
    "Mjlab-QcPendulum",
    "Mjlab-QcAntiSwayAlignment",
  }
  if (
    (cfg.plot_state_action_curve or cfg.save_state_action_curve)
    and not supports_trolley_curves
  ):
    raise ValueError(
      "State/action curve plotting/saving is only supported for trolley tasks: "
      "Mjlab-QcPendulum and Mjlab-QcAntiSwayAlignment."
    )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  trolley_curve_values_np: np.ndarray | None = None
  # Both trolley tasks use the same command key in env_cfg.
  trolley_command_name = "trolley_target"

  # Handle task-specific parameters
  if supports_trolley_curves and cfg.trolley_target is not None:
    if trolley_command_name in env_cfg.commands:
      trolley_command_cfg: Any = env_cfg.commands[trolley_command_name]
      parsed_target_value: float | None = None
      parsed_target_curve_path: Path | None = None

      if isinstance(cfg.trolley_target, str):
        target_arg = cfg.trolley_target.strip()
        target_path = Path(target_arg)
        if target_path.suffix.lower() == ".csv" or target_path.exists():
          parsed_target_curve_path = target_path
        else:
          try:
            parsed_target_value = float(target_arg)
          except ValueError as err:
            raise ValueError(
              "For --trolley-target, provide either a numeric value or a CSV file "
              f"path. Got: {cfg.trolley_target}"
            ) from err
      else:
        parsed_target_value = float(cfg.trolley_target)

      if parsed_target_curve_path is not None:
        if not parsed_target_curve_path.exists():
          raise FileNotFoundError(
            f"Trolley target CSV not found: {parsed_target_curve_path}"
          )
        curve_data = np.loadtxt(
          parsed_target_curve_path, delimiter=",", skiprows=1
        )
        if curve_data.ndim == 1:
          curve_data = curve_data.reshape(1, -1)
        if curve_data.shape[1] < 2:
          raise ValueError(
            "Trolley target CSV must have at least 2 columns: step,value"
          )

        trolley_curve_values_np = np.asarray(curve_data[:, 1], dtype=np.float32)
        if trolley_curve_values_np.size == 0:
          raise ValueError(
            f"Trolley target CSV has no data rows: {parsed_target_curve_path}"
          )

        first_target = float(trolley_curve_values_np[0])
        setattr(trolley_command_cfg, "initial_target", first_target)
        setattr(trolley_command_cfg, "mode", "fixed")
        print(
          "[INFO]: Set trolley target from curve: "
          f"{parsed_target_curve_path} ({trolley_curve_values_np.size} steps)"
        )
      else:
        assert parsed_target_value is not None
        setattr(trolley_command_cfg, "initial_target", parsed_target_value)
        setattr(trolley_command_cfg, "mode", "fixed")
        print(f"[INFO]: Set trolley target to {parsed_target_value} (fixed mode)")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    video_dir = (
      Path(cfg.video_output_dir)
      if cfg.video_output_dir is not None
      else (
        log_dir / "videos" / "play"
        if log_dir is not None
        else Path("outputs") / "videos" / "play"
      )
    )
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
  num_actions = action_shape[-1]
  plot_window = max(10, cfg.plot_window)
  action_history: deque[list[float]] = deque(maxlen=plot_window)
  recorded_actions: list[list[float]] = []
  playback_plot_fig = None
  action_plot_ax = None
  action_plot_lines: list[Line2D] = []
  trolley_plot_axes = None
  trolley_plot_lines: tuple[Line2D, Line2D, Line2D, Line2D] | None = None
  if cfg.plot_state_action_curve:
    plt.ion()
    playback_plot_fig, playback_axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    trolley_plot_axes = playback_axes[:4]
    action_plot_ax = playback_axes[4]
    pos_line = trolley_plot_axes[0].plot([], [], label="trolley_pos", color="tab:blue")[0]
    vel_line = trolley_plot_axes[1].plot([], [], label="trolley_vel", color="tab:orange")[0]
    acc_line = trolley_plot_axes[2].plot([], [], label="trolley_acc", color="tab:green")[0]
    sway_line = trolley_plot_axes[3].plot([], [], label="sway_angle", color="tab:purple")[0]
    trolley_plot_lines = (pos_line, vel_line, acc_line, sway_line)
    action_plot_lines = [
      action_plot_ax.plot([], [], label=f"a{i}")[0] for i in range(num_actions)
    ]

    trolley_plot_axes[0].set_title(f"Playback Curves ({task_id}, env 0)")
    trolley_plot_axes[0].set_ylabel("Position")
    trolley_plot_axes[1].set_ylabel("Velocity")
    trolley_plot_axes[2].set_ylabel("Acceleration")
    trolley_plot_axes[3].set_ylabel("Sway Angle (deg)")
    action_plot_ax.set_ylabel("Action")
    action_plot_ax.set_xlabel("Step")
    action_plot_ax.legend(loc="upper right")
    for axis in trolley_plot_axes:
      axis.grid(True, alpha=0.3)
      axis.legend(loc="upper right")
    action_plot_ax.grid(True, alpha=0.3)
    playback_plot_fig.tight_layout()

  trolley_history: list[tuple[float, float, float, float]] = []

  trolley_asset = None
  trolley_joint_idx = None
  trolley_target_term: Any | None = None
  trolley_curve_values: torch.Tensor | None = None
  trolley_curve_step_count: int | None = None
  if supports_trolley_curves:
    base_env = env.unwrapped
    trolley_entity_name = (
      "qc_pendulum" if task_id == "Mjlab-QcPendulum" else "qc_anti_sway_alignment"
    )
    trolley_asset = base_env.scene[trolley_entity_name]
    trolley_joint_idx = trolley_asset.find_joints("trolley_joint")[0][0]
    sway_trolley_site_idx: int | None = None
    sway_spreader_site_idx: int | None = None
    if task_id == "Mjlab-QcAntiSwayAlignment":
      _site_names = list(trolley_asset.site_names)
      if "trolley_center" in _site_names and "spreader_center" in _site_names:
        sway_trolley_site_idx = _site_names.index("trolley_center")
        sway_spreader_site_idx = _site_names.index("spreader_center")
    if (
      trolley_curve_values_np is not None
      and base_env.command_manager is not None
      and trolley_command_name in base_env.command_manager.active_terms
    ):
      trolley_target_term = base_env.command_manager.get_term(trolley_command_name)
      trolley_curve_values = torch.as_tensor(
        trolley_curve_values_np, device=base_env.device, dtype=torch.float32
      )
      trolley_curve_step_count = int(trolley_curve_values.shape[0])
      # Ensure the first command is in place before stepping.
      target_pos = getattr(trolley_target_term, "target_pos", None)
      if target_pos is not None:
        target_pos[:] = trolley_curve_values[0]
      desired_target_pos = getattr(trolley_target_term, "desired_target_pos", None)
      if desired_target_pos is not None:
        desired_target_pos[:] = trolley_curve_values[0]

  class PolicyWithPlaybackTracking:
    def __init__(self, wrapped_policy):
      self.wrapped_policy = wrapped_policy
      self.update_every = 5
      self.call_count = 0
      self.curve_step_idx = 0
      self._ema_actions: torch.Tensor | None = None

    def __call__(self, obs) -> torch.Tensor:
      if (
        trolley_target_term is not None
        and trolley_curve_values is not None
        and trolley_curve_step_count is not None
        and self.curve_step_idx < trolley_curve_step_count
      ):
        target_pos = getattr(trolley_target_term, "target_pos", None)
        if target_pos is not None:
          target_pos[:] = trolley_curve_values[self.curve_step_idx]
        desired_target_pos = getattr(trolley_target_term, "desired_target_pos", None)
        if desired_target_pos is not None:
          desired_target_pos[:] = trolley_curve_values[self.curve_step_idx]
        self.curve_step_idx += 1

      raw_actions = self.wrapped_policy(obs)
      if cfg.action_ema_alpha > 0.0:
        if self._ema_actions is None:
          self._ema_actions = raw_actions.clone()
        else:
          self._ema_actions = (
            (1.0 - cfg.action_ema_alpha) * raw_actions
            + cfg.action_ema_alpha * self._ema_actions
          )
        actions = self._ema_actions
      else:
        actions = raw_actions
      action_env0 = actions[0].detach().float().cpu().flatten().tolist()
      recorded_actions.append(action_env0)

      if cfg.plot_state_action_curve:
        action_history.append(action_env0)

      if trolley_asset is not None and trolley_joint_idx is not None:
        trolley_pos = float(trolley_asset.data.joint_pos[0, trolley_joint_idx].item())
        trolley_vel = float(trolley_asset.data.joint_vel[0, trolley_joint_idx].item())
        trolley_acc = float(trolley_asset.data.joint_acc[0, trolley_joint_idx].item())
        if sway_trolley_site_idx is not None and sway_spreader_site_idx is not None:
          t_s = trolley_asset.data.site_pos_w[0, sway_trolley_site_idx]
          sp_s = trolley_asset.data.site_pos_w[0, sway_spreader_site_idx]
          sway_angle = math.degrees(math.atan2(
            float((sp_s[1] - t_s[1]).item()),
            float((t_s[2] - sp_s[2]).item()),
          ))
        else:
          sway_angle = 0.0
        trolley_history.append((trolley_pos, trolley_vel, trolley_acc, sway_angle))

      self.call_count += 1
      if (
        cfg.plot_state_action_curve
        and self.call_count % self.update_every == 0
        and len(action_history) > 1
      ):
        ys = torch.tensor(list(action_history), dtype=torch.float32)
        xs = list(range(len(action_history)))
        for i, line in enumerate(action_plot_lines):
          line.set_data(xs, ys[:, i].tolist())
        assert action_plot_ax is not None
        assert playback_plot_fig is not None
        action_plot_ax.set_xlim(0, max(1, len(action_history) - 1))
        y_min = float(torch.min(ys))
        y_max = float(torch.max(ys))
        if y_min == y_max:
          y_min -= 0.1
          y_max += 0.1
        margin = 0.1 * (y_max - y_min)
        action_plot_ax.set_ylim(y_min - margin, y_max + margin)
        playback_plot_fig.canvas.draw_idle()
        playback_plot_fig.canvas.flush_events()
        plt.pause(0.001)

      if (
        cfg.plot_state_action_curve
        and trolley_plot_axes is not None
        and trolley_plot_lines is not None
        and len(trolley_history) > 1
        and self.call_count % self.update_every == 0
      ):
        xs = list(range(len(trolley_history)))
        ys = np.asarray(trolley_history, dtype=np.float32)
        for axis, line, values in zip(
          trolley_plot_axes,
          trolley_plot_lines,
          (ys[:, 0], ys[:, 1], ys[:, 2], ys[:, 3]),
          strict=True,
        ):
          line.set_data(xs, values.tolist())
          axis.set_xlim(0, max(1, len(xs) - 1))
          y_min = float(values.min())
          y_max = float(values.max())
          if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1
          margin = 0.1 * (y_max - y_min)
          axis.set_ylim(y_min - margin, y_max + margin)
        assert playback_plot_fig is not None
        playback_plot_fig.canvas.draw_idle()
        playback_plot_fig.canvas.flush_events()
        plt.pause(0.001)

      return actions

  policy = PolicyWithPlaybackTracking(policy)
  viewer_num_steps = trolley_curve_step_count

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run(num_steps=viewer_num_steps)
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run(num_steps=viewer_num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()
  if cfg.plot_state_action_curve:
    plt.ioff()
    plt.show(block=False)
  if (
    cfg.save_state_action_curve
    and supports_trolley_curves
    and trolley_history
    and recorded_actions
  ):
    output_dir = (
      Path(cfg.trolley_output_dir)
      if cfg.trolley_output_dir is not None
      else (log_dir / "play" if log_dir is not None else Path("outputs") / "play")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    action_array = np.asarray(recorded_actions, dtype=np.float32)
    trolley_array = np.asarray(trolley_history, dtype=np.float32)
    n = len(trolley_array)

    # Align reference curve to recorded length (trim or pad with last value).
    if trolley_curve_values_np is not None:
      ref = trolley_curve_values_np
      if len(ref) >= n:
        ref_aligned: np.ndarray | None = ref[:n]
      else:
        pad = np.full(n - len(ref), ref[-1], dtype=np.float32)
        ref_aligned = np.concatenate([ref, pad])
    else:
      ref_aligned = None

    data_path = output_dir / "state_action_curves.csv"
    action_header = ",".join(f"action_{i}" for i in range(action_array.shape[1]))
    if ref_aligned is not None:
      csv_data = np.column_stack(
        (np.arange(n, dtype=np.int32), trolley_array, action_array, ref_aligned)
      )
      header = (
        f"step,trolley_pos,trolley_vel,trolley_acc,spreader_sway_angle_deg,"
        f"{action_header},target_pos"
      )
    else:
      csv_data = np.column_stack(
        (np.arange(n, dtype=np.int32), trolley_array, action_array)
      )
      header = (
        f"step,trolley_pos,trolley_vel,trolley_acc,spreader_sway_angle_deg,"
        f"{action_header}"
      )
    # Save with fixed-width 4-decimal formatting for easier visual alignment.
    csv_fmt = ["%10.4f"] * int(csv_data.shape[1])
    np.savetxt(
      str(data_path),
      csv_data,
      delimiter=",",
      fmt=csv_fmt,
      header=header,
      comments="",
    )

    save_fig, save_axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    xs = np.arange(n, dtype=np.int32)
    save_axes[0].plot(xs, trolley_array[:, 0], color="tab:blue", label="actual")
    if ref_aligned is not None:
      save_axes[0].plot(xs, ref_aligned, color="tab:red", linestyle="--", label="target", alpha=0.8)
    save_axes[0].legend(loc="upper right")
    save_axes[1].plot(xs, trolley_array[:, 1], color="tab:orange")
    save_axes[2].plot(xs, trolley_array[:, 2], color="tab:green")
    save_axes[3].plot(xs, trolley_array[:, 3], color="tab:purple")
    for i in range(action_array.shape[1]):
      save_axes[4].plot(xs, action_array[:, i], label=f"a{i}")
    save_axes[0].set_title(f"Playback Curves ({task_id}, env 0)")
    save_axes[0].set_ylabel("Position")
    save_axes[1].set_ylabel("Velocity")
    save_axes[2].set_ylabel("Acceleration")
    save_axes[3].set_ylabel("Sway Angle (deg)")
    save_axes[4].set_ylabel("Action")
    save_axes[4].set_xlabel("Step")
    for axis in save_axes:
      axis.grid(True, alpha=0.3)
    save_axes[4].legend(loc="upper right")
    save_fig.tight_layout()
    figure_path = output_dir / "state_action_curves.png"
    save_fig.savefig(str(figure_path), dpi=160)
    plt.close(save_fig)
    print(f"[INFO]: Saved state/action curve data to {data_path}")
    print(f"[INFO]: Saved state/action curve figure to {figure_path}")


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
