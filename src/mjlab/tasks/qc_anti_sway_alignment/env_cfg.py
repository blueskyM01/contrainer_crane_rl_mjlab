"""My custom cartpole balance environment configuration."""
import math
from dataclasses import dataclass
from typing import Literal
from mjlab.tasks.velocity.mdp.rewards import upright
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.asset_zoo.robots.qc_anti_sway_alignment.qc_anti_sway_alignment_constants import (
  get_qc_anti_sway_alignment_robot_cfg,
)
from mjlab.envs import mdp
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  nan_detection,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.rewards import action_acc_l2, action_rate_l2, joint_acc_l2, joint_vel_l2

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab.envs import ManagerBasedRlEnv



# Rewards.
class TrolleyTargetCommand(CommandTerm):
  cfg: "TrolleyTargetCommandCfg"

  def __init__(self, cfg: "TrolleyTargetCommandCfg", env):
    super().__init__(cfg, env)
    self.target_pos = torch.full(
      (self.num_envs,), cfg.initial_target, device=self.device, dtype=torch.float32
    )
    self.desired_target_pos = self.target_pos.clone()
    self.target_vel = torch.zeros_like(self.target_pos)
    self.target_acc = torch.zeros_like(self.target_pos)
    self.metrics["target_pos"] = self.target_pos
    self.metrics["target_vel"] = self.target_vel
    self.metrics["target_acc"] = self.target_acc

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    self.metrics["target_pos"] = self.target_pos
    self.metrics["target_vel"] = self.target_vel
    self.metrics["target_acc"] = self.target_acc

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.mode == "fixed":
      self.desired_target_pos[env_ids] = self.cfg.initial_target
    else:
      self.desired_target_pos[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(*self.cfg.target_range)

  def _update_command(self) -> None:
    prev_pos = self.target_pos.clone()
    prev_vel = self.target_vel.clone()

    dt = float(self._env.step_dt)
    if self.cfg.traverse_speed > 0.0:
      # Linear traversal at constant speed: farther targets take proportionally longer.
      delta = self.desired_target_pos - self.target_pos
      max_step = self.cfg.traverse_speed * dt
      self.target_pos.add_(delta.clamp(-max_step, max_step))
    elif self.cfg.smoothing_tau <= 0.0:
      self.target_pos.copy_(self.desired_target_pos)
    else:
      alpha = min(dt / self.cfg.smoothing_tau, 1.0)
      self.target_pos += alpha * (self.desired_target_pos - self.target_pos)

    if dt > 0.0:
      self.target_vel.copy_((self.target_pos - prev_pos) / dt)
      self.target_acc.copy_((self.target_vel - prev_vel) / dt)
    else:
      self.target_vel.zero_()
      self.target_acc.zero_()


@dataclass(kw_only=True)
class TrolleyTargetCommandCfg(CommandTermCfg):
  entity_name: str
  target_range: tuple[float, float] = (0.0, 1.4)
  initial_target: float = 0.7
  mode: Literal["fixed", "random"] = "random"
  smoothing_tau: float = 0.0
  traverse_speed: float = 0.0
  """Constant traversal speed (m/s) for linear interpolation toward the target.
  When > 0, target_pos moves at this speed each step; travel time is
  proportional to distance (e.g. 0.5 m/s → 1.5 m in 3 s, 0.5 m in 1 s).
  Set to 0.0 to fall back to smoothing_tau or instant-jump behaviour."""

  def build(self, env):
    return TrolleyTargetCommand(self, env)


def _track_lin_pos_exp(base_pos: torch.Tensor, cmd_pos: float | torch.Tensor, std: float) -> torch.Tensor:
  """Exponential tracking reward for linear position."""
  if isinstance(cmd_pos, torch.Tensor):
    cmd = cmd_pos
  else:
    cmd = torch.full_like(base_pos, fill_value=cmd_pos)
  cmd = cmd.to(base_pos)
  if cmd.dim() == 0:
    cmd = cmd.expand_as(base_pos)
  return torch.exp(-0.5 * ((base_pos - cmd) / std)**2)


def generated_commands_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Generate command observation with explicit channel dimension."""
  cmd = mdp.generated_commands(env, command_name)
  if cmd.ndim == 1:
    cmd = cmd.unsqueeze(-1)
  return cmd


def generated_command_acc_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Generate command acceleration observation with explicit channel dimension."""
  term = env.command_manager.get_term(command_name)
  cmd_acc = getattr(term, "target_acc", None)
  if cmd_acc is None:
    cmd = mdp.generated_commands(env, command_name)
    cmd_acc = torch.zeros_like(cmd)
  if cmd_acc.ndim == 1:
    cmd_acc = cmd_acc.unsqueeze(-1)
  return cmd_acc


def trolley_acc_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current trolley joint acceleration."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  return torch.nan_to_num(joint_acc, nan=0.0, posinf=0.0, neginf=0.0)


def previous_trolley_pos_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Approximate previous-step trolley position using current state and dt."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  dt = env.step_dt
  return pos - vel * dt


def previous_trolley_vel_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Approximate previous-step trolley velocity using velocity and acceleration."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  acc = torch.nan_to_num(
    asset.data.joint_acc[:, asset_cfg.joint_ids],
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
  )
  dt = env.step_dt
  return vel - acc * dt


def previous_trolley_acc_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Approximate previous-step trolley acceleration (jerk not available; returns current acc)."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  return torch.nan_to_num(joint_acc, nan=0.0, posinf=0.0, neginf=0.0)


def spreader_sway_angle_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_site_name: str = "trolley_center",
  spreader_site_name: str = "spreader_center",
) -> torch.Tensor:
  """Angle between the trolley-spreader line and vertical downward in the Y-Z plane."""
  asset: Entity = env.scene[asset_cfg.name]

  trolley_site_idx = asset.site_names.index(trolley_site_name)
  spreader_site_idx = asset.site_names.index(spreader_site_name)

  trolley_site_pos = asset.data.site_pos_w[:, trolley_site_idx]
  spreader_site_pos = asset.data.site_pos_w[:, spreader_site_idx]

  dy = (spreader_site_pos[:, 1] - trolley_site_pos[:, 1]).unsqueeze(-1)
  dz = (trolley_site_pos[:, 2] - spreader_site_pos[:, 2]).unsqueeze(-1)
  return torch.atan2(dy, dz)


def rope_length_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_site_name: str = "trolley_center",
  spreader_site_name: str = "spreader_center",
) -> torch.Tensor:
  """Euclidean distance between trolley and spreader center sites in world frame."""
  asset: Entity = env.scene[asset_cfg.name]

  trolley_site_idx = asset.site_names.index(trolley_site_name)
  spreader_site_idx = asset.site_names.index(spreader_site_name)

  trolley_site_pos = asset.data.site_pos_w[:, trolley_site_idx]
  spreader_site_pos = asset.data.site_pos_w[:, spreader_site_idx]
  return torch.norm(spreader_site_pos - trolley_site_pos, dim=-1, keepdim=True)


def previous_spreader_sway_angle_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_site_name: str = "trolley_center",
  spreader_site_name: str = "spreader_center",
) -> torch.Tensor:
  """Approximate previous-step sway angle from site positions and site linear velocities."""
  asset: Entity = env.scene[asset_cfg.name]

  trolley_site_idx = asset.site_names.index(trolley_site_name)
  spreader_site_idx = asset.site_names.index(spreader_site_name)

  trolley_site_pos = asset.data.site_pos_w[:, trolley_site_idx]
  spreader_site_pos = asset.data.site_pos_w[:, spreader_site_idx]
  trolley_site_vel = asset.data.site_lin_vel_w[:, trolley_site_idx]
  spreader_site_vel = asset.data.site_lin_vel_w[:, spreader_site_idx]

  dt = env.step_dt
  prev_trolley_site_pos = trolley_site_pos - trolley_site_vel * dt
  prev_spreader_site_pos = spreader_site_pos - spreader_site_vel * dt

  dy = (prev_spreader_site_pos[:, 1] - prev_trolley_site_pos[:, 1]).unsqueeze(-1)
  dz = (prev_trolley_site_pos[:, 2] - prev_spreader_site_pos[:, 2]).unsqueeze(-1)
  return torch.atan2(dy, dz)


def spreader_sway_angle_vel_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_site_name: str = "trolley_center",
  spreader_site_name: str = "spreader_center",
) -> torch.Tensor:
  """Sway angle velocity in the Y-Z plane (rad/s)."""
  asset: Entity = env.scene[asset_cfg.name]

  trolley_site_idx = asset.site_names.index(trolley_site_name)
  spreader_site_idx = asset.site_names.index(spreader_site_name)

  trolley_site_pos = asset.data.site_pos_w[:, trolley_site_idx]
  spreader_site_pos = asset.data.site_pos_w[:, spreader_site_idx]
  trolley_site_vel = asset.data.site_lin_vel_w[:, trolley_site_idx]
  spreader_site_vel = asset.data.site_lin_vel_w[:, spreader_site_idx]

  dy = spreader_site_pos[:, 1] - trolley_site_pos[:, 1]
  dz = trolley_site_pos[:, 2] - spreader_site_pos[:, 2]
  dy_dot = spreader_site_vel[:, 1] - trolley_site_vel[:, 1]
  dz_dot = trolley_site_vel[:, 2] - spreader_site_vel[:, 2]

  denom = dy * dy + dz * dz
  denom = torch.clamp(denom, min=1e-8)
  theta_dot = (dz * dy_dot - dy * dz_dot) / denom
  return theta_dot.unsqueeze(-1)


def previous_spreader_sway_angle_vel_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_site_name: str = "trolley_center",
  spreader_site_name: str = "spreader_center",
) -> torch.Tensor:
  """Approximate previous-step sway angle velocity from angle finite difference."""
  dt = float(env.step_dt)
  if dt <= 0.0:
    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

  angle = spreader_sway_angle_obs(
    env,
    asset_cfg,
    trolley_site_name=trolley_site_name,
    spreader_site_name=spreader_site_name,
  )
  prev_angle = previous_spreader_sway_angle_obs(
    env,
    asset_cfg,
    trolley_site_name=trolley_site_name,
    spreader_site_name=spreader_site_name,
  )
  return (angle - prev_angle) / dt


def previous_action_obs(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """Get previous-step action from action manager with stable shape."""
  action_term = env.action_manager.get_term(action_name)
  pa: torch.Tensor | None = getattr(action_term, "prev_action", None) or getattr(
    action_term, "_prev_action", None
  )
  action: torch.Tensor = (
    pa if pa is not None
    else torch.zeros((env.num_envs, action_term.action_dim), device=env.device)
  )
  if action.ndim == 1:
    action = action.unsqueeze(-1)
  return action


def joint_limit_violation_termination(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  tolerance: float = 1e-3,
) -> torch.Tensor:
  """Terminate envs whose selected joints leave their soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  below = joint_pos < (joint_limits[..., 0] - tolerance)
  above = joint_pos > (joint_limits[..., 1] + tolerance)
  return torch.any(below | above, dim=1)


def spreader_anti_sway_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_cfg: SceneEntityCfg,
  angle_std: float = 0.2,
  settle_pos_std: float = 0.08,
  settle_vel_std: float = 0.12,
  settle_angle_vel_std: float = 0.35,
  crossing_angle_std: float = 0.06,
  global_outward_vel_std: float = 0.45,
  global_crossing_vel_std: float = 0.25,
) -> torch.Tensor:
  """Encourage one-sided damped sway during motion and zero sway at stop.

  Transit phase: allow non-zero sway but discourage outward growth and fast
  centerline crossings (back-and-forth oscillation).
  Near-stop phase: strongly encourage both sway angle and sway angular velocity
  to converge to zero.
  """
  angle = spreader_sway_angle_obs(env, asset_cfg).squeeze(-1)
  angle_vel = spreader_sway_angle_vel_obs(env, asset_cfg).squeeze(-1)
  prev_angle = previous_spreader_sway_angle_obs(env, asset_cfg).squeeze(-1)
  trolley: Entity = env.scene[trolley_cfg.name]
  trolley_pos = trolley.data.joint_pos[:, trolley_cfg.joint_ids].squeeze(-1)
  trolley_vel = trolley.data.joint_vel[:, trolley_cfg.joint_ids].squeeze(-1)

  if env.command_manager is not None and "trolley_target" in env.command_manager.active_terms:
    target_cmd = env.command_manager.get_command("trolley_target")
    if target_cmd is None:
      target_cmd = torch.full_like(trolley_pos, 0.7)
    else:
      target_cmd = target_cmd.squeeze(-1) if target_cmd.ndim > 1 else target_cmd
  else:
    target_cmd = torch.full_like(trolley_pos, 0.7)

  pos_error = trolley_pos - target_cmd
  settled_gate = torch.exp(
    -0.5 * (pos_error / settle_pos_std) ** 2
    -0.5 * (trolley_vel / settle_vel_std) ** 2
  )
  transit_gate = 1.0 - settled_gate

  outward_vel = torch.relu(torch.sign(angle) * angle_vel)
  center_crossing_vel = torch.where(prev_angle * angle < 0.0, angle_vel.abs(), 0.0)
  near_center_gate = torch.exp(-0.5 * (angle / crossing_angle_std) ** 2)
  oscillation_damping = torch.exp(
    -0.5 * (outward_vel / global_outward_vel_std) ** 2
    -0.5 * ((center_crossing_vel * near_center_gate) / global_crossing_vel_std) ** 2
  )

  settle_term = torch.exp(
    -0.5 * (angle / angle_std) ** 2
    -0.5 * (angle_vel / settle_angle_vel_std) ** 2
  )
  reward = transit_gate * oscillation_damping + settled_gate * settle_term
  return torch.clamp(reward, 0.0, 1.0)


def stopped_sway_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  trolley_cfg: SceneEntityCfg,
  vel_threshold: float = 0.05,
  angle_std: float = 0.05,
) -> torch.Tensor:
  """Penalize sway angle when the trolley is stopped.

  Returns a value in [0, 1] that is gated by a soft velocity threshold and
  proportional to how far the sway angle is from zero. Intended to be used
  with a negative weight so it acts as a pure penalty, decoupled from
  spreader_anti_sway_reward which handles the transit phase.
  """
  angle = spreader_sway_angle_obs(env, asset_cfg).squeeze(-1)
  trolley: Entity = env.scene[trolley_cfg.name]
  trolley_vel = trolley.data.joint_vel[:, trolley_cfg.joint_ids].squeeze(-1)

  stopped_gate = torch.exp(-0.5 * (trolley_vel / vel_threshold) ** 2)
  sway_penalty = 1.0 - torch.exp(-0.5 * (angle / angle_std) ** 2)
  return stopped_gate * sway_penalty


def trolley_acc_l2_reward(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  std: float = 2.0,
) -> torch.Tensor:
  """Return normalized trolley acceleration penalty (larger is worse)."""
  acc_l2_sq = joint_acc_l2(env, trolley_cfg)
  return acc_l2_sq / (std ** 2)


def trolley_vel_l2_reward(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  std: float = 0.5,
) -> torch.Tensor:
  """Return normalized trolley velocity penalty (larger is worse)."""
  vel_l2_sq = joint_vel_l2(env, trolley_cfg)
  return vel_l2_sq / (std ** 2)


def action_rate_l2_reward(
  env: ManagerBasedRlEnv,
  std: float = 0.2,
) -> torch.Tensor:
  """Return normalized action-rate penalty to suppress command chattering."""
  return action_rate_l2(env) / (std ** 2)


def action_acc_l2_reward(
  env: ManagerBasedRlEnv,
  std: float = 1.0,
) -> torch.Tensor:
  """Return normalized action-acceleration penalty to smooth action transitions."""
  return action_acc_l2(env) / (std ** 2)


def action_l2_reward(
  env: ManagerBasedRlEnv,
  std: float = 1.0,
) -> torch.Tensor:
  """Return normalized action-magnitude penalty to suppress large bursts."""
  return torch.sum(torch.square(env.action_manager.action), dim=1) / (std ** 2)


def trolley_cmd_track_reward_func(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  std: float = 0.5,
) -> torch.Tensor:
  """Return trolley target-tracking reward for QC anti-sway alignment.

  The reward is computed from the trolley position error relative to the
  current ``trolley_target`` command using an exponential tracking kernel.

  Args:
    env: RL environment that provides scene state and command manager.
    trolley_cfg: Entity/joint selection for trolley state lookup.
      Expected to reference ``qc_anti_sway_alignment`` and ``trolley_joint``.

  Returns:
    Per-environment reward tensor with shape ``(num_envs,)``. Values are in
    ``(0, 1]``, and approach ``1.0`` as trolley position approaches command target.
  """

  asset: Entity = env.scene[trolley_cfg.name]
  
  # trolley position stop.
  trolley_pos = asset.data.joint_pos[:, trolley_cfg.joint_ids].squeeze(-1)
  # Use command target if available; otherwise default to 0.7.
  if env.command_manager is not None and "trolley_target" in env.command_manager.active_terms:
    target_cmd = env.command_manager.get_command("trolley_target")
    if target_cmd is None:
      target_cmd = torch.full_like(trolley_pos, 0.7)
    else:
      target_cmd = target_cmd.squeeze(-1) if target_cmd.ndim > 1 else target_cmd
  else:
    target_cmd = torch.full_like(trolley_pos, 0.7)
  pos_stop = _track_lin_pos_exp(trolley_pos, cmd_pos=target_cmd, std=std)
  return pos_stop



def qc_anti_sway_alignment_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  trolley_cfg = SceneEntityCfg("qc_anti_sway_alignment", joint_names=("trolley_joint",))

  actor_terms = {
    "trolley_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": trolley_cfg},
    ),
    "trolley_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": trolley_cfg},
    ),
    "trolley_acc": ObservationTermCfg(
      func=trolley_acc_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "prev_trolley_pos": ObservationTermCfg(
      func=previous_trolley_pos_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "prev_trolley_vel": ObservationTermCfg(
      func=previous_trolley_vel_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "prev_trolley_acc": ObservationTermCfg(
      func=previous_trolley_acc_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "prev_trolley_driver_action": ObservationTermCfg(
      func=previous_action_obs,
      params={"action_name": "position"},
    ),
    "spreader_sway_angle": ObservationTermCfg(
      func=spreader_sway_angle_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "spreader_sway_angle_vel": ObservationTermCfg(
      func=spreader_sway_angle_vel_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "pre_spreader_sway_angle": ObservationTermCfg(
      func=previous_spreader_sway_angle_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "pre_spreader_sway_angle_vel": ObservationTermCfg(
      func=previous_spreader_sway_angle_vel_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "rope_length": ObservationTermCfg(
      func=rope_length_obs,
      params={"asset_cfg": trolley_cfg},
    ),
    "command": ObservationTermCfg(
      func=generated_commands_obs,
      params={"command_name": "trolley_target"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "position": JointPositionActionCfg(
      entity_name="qc_anti_sway_alignment",
      actuator_names=("trolley_joint", "hoist_joint"),
      scale={"trolley_joint": 1.0, "hoist_joint": 0.0},
    ),
  }

  commands = {
    "trolley_target": TrolleyTargetCommandCfg(
      resampling_time_range=(5.0, 5.0),
      entity_name="qc_anti_sway_alignment",
      target_range=(-0.08, 1.48),
      initial_target=0.7,
      mode="random",
      smoothing_tau=0.0,
      traverse_speed=0.5,  # 恒速线性插值：距离1.5m需3s，距离0.5m需1s
      debug_vis=False,
    ),
  }

  events = {
    "reset_trolley": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.08, 1.48),
        "velocity_range": (-0.4, 0.4),
        "asset_cfg": SceneEntityCfg("qc_anti_sway_alignment", joint_names=("trolley_joint",)),
      },
    ),
    "reset_hoist": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("qc_anti_sway_alignment", joint_names=("hoist_joint",)),
      },
    ),
  }

  rewards = {
    "trolley_cmd_track_reward": RewardTermCfg(
      func=trolley_cmd_track_reward_func,
      weight=1.1,
      params={"trolley_cfg": trolley_cfg, "std": 0.5},
    ),
    "spreader_anti_sway_reward": RewardTermCfg(
      func=spreader_anti_sway_reward,
      weight=0.28,
      params={
        "asset_cfg": SceneEntityCfg("qc_anti_sway_alignment"),
        "trolley_cfg": SceneEntityCfg(
          "qc_anti_sway_alignment", joint_names=("trolley_joint",)
        ),
        "angle_std": 0.07,
        "settle_pos_std": 0.12,
        "settle_vel_std": 0.15,
        "settle_angle_vel_std": 0.10,
        "crossing_angle_std": 0.04,
        "global_outward_vel_std": 0.35,
        "global_crossing_vel_std": 0.12,
      },
    ),
    "stopped_sway_penalty": RewardTermCfg(
      func=stopped_sway_penalty,
      weight=-0.08,
      params={
        "asset_cfg": SceneEntityCfg("qc_anti_sway_alignment"),
        "trolley_cfg": SceneEntityCfg(
          "qc_anti_sway_alignment", joint_names=("trolley_joint",)
        ),
        "vel_threshold": 0.02,
        "angle_std": 0.12,
      },
    ),
    "trolley_vel_l2_reward": RewardTermCfg(
      func=trolley_vel_l2_reward,
      weight=-0.00012,
      params={
        "trolley_cfg": SceneEntityCfg("qc_anti_sway_alignment", joint_names=("trolley_joint",)),
        "std": 0.5,
      },
    ),
    "trolley_acc_l2_reward": RewardTermCfg(
      func=trolley_acc_l2_reward,
      weight=-0.00015,
      params={
        "trolley_cfg": SceneEntityCfg("qc_anti_sway_alignment", joint_names=("trolley_joint",)),
        "std": 1.8,
      },
    ),
    "action_rate_l2_reward": RewardTermCfg(
      func=action_rate_l2_reward,
      weight=-0.00006,
      params={"std": 0.5},
    ),
    "action_acc_l2_reward": RewardTermCfg(
      func=action_acc_l2_reward,
      weight=-0.00008,
      params={"std": 1.0},
    ),
    "action_l2_reward": RewardTermCfg(
      func=action_l2_reward,
      weight=-0.00003,
      params={"std": 1.0},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=nan_detection, time_out=False),
    "joint_limit_violation": TerminationTermCfg(
      func=joint_limit_violation_termination,
      time_out=False,
      params={
        "asset_cfg": SceneEntityCfg(
          "qc_anti_sway_alignment",
          joint_names=("trolley_joint",),
        ),
        "tolerance": 1e-3,
      },
    ),
  }
  
  if play:
    episode_length_s = 1e10
    observations["actor"].enable_corruption = False
    events["reset_trolley"] = EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("qc_anti_sway_alignment", joint_names=("trolley_joint",)),
      },
    )
  else:
    episode_length_s = 50.0

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"qc_anti_sway_alignment": get_qc_anti_sway_alignment_robot_cfg()},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    commands=commands,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="qc_anti_sway_alignment",
      body_name="gantry",
      distance=4.0,
      elevation=-5.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    decimation=5,
    episode_length_s=episode_length_s,
  )