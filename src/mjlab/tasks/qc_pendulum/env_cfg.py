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
from mjlab.asset_zoo.robots.qc_pendulum.qc_pendulum_constants import get_qc_pendulum_robot_cfg
from mjlab.envs import mdp
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  time_out,
)
from mjlab.utils.lab_api.math import sample_uniform

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab.envs import ManagerBasedRlEnv

# Observations
# def pole_angle_cos_sin(
#   env: ManagerBasedRlEnv,
#   asset_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#   """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2]."""
#   asset: Entity = env.scene[asset_cfg.name]
#   angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
#   return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


# Rewards.
class TrolleyTargetCommand(CommandTerm):
  cfg: "TrolleyTargetCommandCfg"

  def __init__(self, cfg: "TrolleyTargetCommandCfg", env):
    super().__init__(cfg, env)
    self.target_pos = torch.full(
      (self.num_envs,), cfg.initial_target, device=self.device, dtype=torch.float32
    )
    self.metrics["target_pos"] = self.target_pos

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    self.metrics["target_pos"] = self.target_pos

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.mode == "fixed":
      self.target_pos[env_ids] = self.cfg.initial_target
    else:
      self.target_pos[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(*self.cfg.target_range)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class TrolleyTargetCommandCfg(CommandTermCfg):
  entity_name: str
  target_range: tuple[float, float] = (0.0, 1.4)
  initial_target: float = 0.7
  mode: Literal["fixed", "random"] = "random"

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
  """Approximate previous-step trolley velocity using current velocity and acceleration."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  dt = env.step_dt
  return vel - acc * dt


def previous_action_obs(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """Get previous-step action from action manager with stable shape."""
  action_term = env.action_manager.get_term(action_name)
  if hasattr(action_term, "prev_action") and action_term.prev_action is not None:
    action = action_term.prev_action
  elif hasattr(action_term, "_prev_action") and action_term._prev_action is not None:
    action = action_term._prev_action
  elif hasattr(action_term, "_processed_actions") and action_term._processed_actions is not None:
    action = torch.zeros_like(action_term._processed_actions)
  elif hasattr(action_term, "raw_actions") and action_term.raw_actions is not None:
    action = torch.zeros_like(action_term.raw_actions)
  else:
    action = torch.zeros((env.num_envs, action_term.action_dim), device=env.device)

  if action.ndim == 1:
    action = action.unsqueeze(-1)
  return action


def reset_trolley_with_ball(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  angle_range_deg: tuple[float, float],
  trolley_cfg: SceneEntityCfg,
) -> None:
  """Reset trolley joint and initialize ball with a random pendulum angle.

  The tendon length is kept constant (0.9 m) while the line from
  ``anchor_trolley`` to ``anchor_ball`` is initialized at a random angle in the
  Y-Z plane.

  Angle convention:
  - 0 deg: line overlaps with vertical downward direction.
  - positive angle: ball shifts toward +Y.
  - negative angle: ball shifts toward -Y.

  Initialization policy:
  - Trolley starts at the absolute joint position sampled from ``position_range``
    (clamped only by joint limits). This is independent of the ball angle.
  - Pendulum starts from ``angle_range_deg`` sample relative to the trolley position.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[trolley_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  rope_len = 0.9

  angle_deg = sample_uniform(
    angle_range_deg[0],
    angle_range_deg[1],
    (len(env_ids), 1),
    env.device,
  )
  angle_rad = angle_deg * (math.pi / 180.0)

  # Step 1: Compute trolley joint reset from position_range (absolute joint position).
  # Use the default shape as reference; sample directly so the trolley position is
  # fully determined by position_range and independent of the ball angle.
  _shape = default_joint_pos[env_ids][:, trolley_cfg.joint_ids].shape
  joint_pos = sample_uniform(*position_range, _shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, trolley_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, trolley_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = trolley_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )

  # Step 2: Move ball so tendon length stays fixed with random initial angle.
  # Geometry (from XML): anchor_trolley = (0.61, -0.7 + p, 1.9),
  # default anchor_ball = (0.61, -0.7, 1.0), so rope length is 0.9 m.
  n = len(env_ids)

  anchor_y = -0.7 + joint_pos
  anchor_z = torch.full((n, 1), 1.9, device=env.device)
  ball_y = anchor_y + rope_len * torch.sin(angle_rad)
  ball_z = anchor_z - rope_len * torch.cos(angle_rad)

  ball_qpos = torch.zeros(n, 7, device=env.device)
  ball_qpos[:, 0] = 0.61
  ball_qpos[:, 1] = ball_y.squeeze(-1)
  ball_qpos[:, 2] = ball_z.squeeze(-1)
  ball_qpos[:, 3] = 1.0  # qw=1: identity quaternion

  # free_joint_q_adr holds the 7 qpos indices for the ball freejoint.
  # The entity's is_fixed_base=True (trolley_joint is first in the MuJoCo tree),
  # so write_root_link_pose_to_sim would raise; write directly instead.
  ball_q_adr = asset.data.indexing.free_joint_q_adr  # shape [7]
  ball_v_adr = asset.data.indexing.free_joint_v_adr  # shape [6]
  asset.data.data.qpos[env_ids.unsqueeze(1), ball_q_adr.unsqueeze(0)] = ball_qpos
  asset.data.data.qvel[
    env_ids.unsqueeze(1), ball_v_adr.unsqueeze(0)
  ] = torch.zeros(n, 6, device=env.device)


def trolley_track_pos_reward(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  pos_std: float = 0.5,
) -> torch.Tensor:
  """Compute smooth reward for QC pendulum trolley positioning task.

  This reward function encourages the trolley to reach and maintain the target position
  specified by the command system. It uses an exponential reward based on position error.

  Args:
    env: The RL environment instance containing scene, command manager, and other components.
    trolley_cfg: Configuration specifying which entity and joints to use for trolley control.
                Should reference the "qc_pendulum" entity and "trolley_joint".

  Returns:
    torch.Tensor: Reward values for each environment in the batch.
                 Shape: (num_envs,) - values between 0 and 1, where 1.0 is perfect tracking.
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
  pos_stop = _track_lin_pos_exp(trolley_pos, cmd_pos=target_cmd, std=pos_std)

  return pos_stop


def trolley_out_of_bounds(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  min_pos: float = 0.0,
  max_pos: float = 1.4,
) -> torch.Tensor:
  """Termination condition: trolley position exceeds bounds.

  Args:
    env: The RL environment instance.
    trolley_cfg: Configuration specifying which entity and joints to use.
    min_pos: Minimum allowed position.
    max_pos: Maximum allowed position.

  Returns:
    torch.Tensor: Boolean tensor indicating out of bounds (True = terminate).
  """
  asset: Entity = env.scene[trolley_cfg.name]
  trolley_pos = asset.data.joint_pos[:, trolley_cfg.joint_ids].squeeze(-1)
  return (trolley_pos < min_pos) | (trolley_pos > max_pos)


def trolley_out_of_bounds_penalty(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  min_pos: float = 0.0,
  max_pos: float = 1.4,
) -> torch.Tensor:
  """Penalty reward when trolley position exceeds bounds.

  Args:
    env: The RL environment instance.
    trolley_cfg: Configuration specifying which entity and joints to use.
    min_pos: Minimum allowed position.
    max_pos: Maximum allowed position.

  Returns:
    torch.Tensor: Penalty values (negative for out of bounds, zero otherwise).
  """
  asset: Entity = env.scene[trolley_cfg.name]
  trolley_pos = asset.data.joint_pos[:, trolley_cfg.joint_ids].squeeze(-1)
  out_of_bounds = (trolley_pos < min_pos) | (trolley_pos > max_pos)
  return -torch.where(out_of_bounds, torch.ones_like(trolley_pos), torch.zeros_like(trolley_pos))


def trolley_vel_smoothness(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  vel_std: float = 0.3,
) -> torch.Tensor:
  """Reward for smooth trolley motion (low velocity/acceleration).

  This reward encourages the trolley to move smoothly by penalizing high velocities.
  Typically used alongside position tracking rewards to balance speed vs accuracy.

  Args:
    env: The RL environment instance.
    trolley_cfg: Configuration specifying which entity and joints to use.
    vel_std: Standard deviation for velocity Gaussian decay. Lower values -> stricter penalty on velocity.

  Returns:
    torch.Tensor: Reward values between 0 and 1. Higher velocity -> lower reward.
  """
  asset: Entity = env.scene[trolley_cfg.name]
  trolley_vel = asset.data.joint_vel[:, trolley_cfg.joint_ids].squeeze(-1)
  
  # Reward based on low velocity using Gaussian decay
  # Encourages smooth, gentle motion
  smoothness = torch.exp(-0.5 * (torch.abs(trolley_vel) / vel_std) ** 2)
  
  return smoothness


def trolley_acceleration_smoothness_reward(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
  acc_std: float = 0.5,
) -> torch.Tensor:
  """Reward for smooth trolley acceleration (low jerk/jerkiness).

  This reward encourages the trolley to accelerate/decelerate smoothly by penalizing
  high accelerations. This promotes gentle, jerk-free motion which is important for
  precise positioning tasks.

  Args:
    env: The RL environment instance.
    trolley_cfg: Configuration specifying which entity and joints to use.
    acc_std: Standard deviation for acceleration Gaussian decay. Lower values -> stricter penalty on acceleration.

  Returns:
    torch.Tensor: Reward values between 0 and 1. Higher acceleration -> lower reward.
  """
  asset: Entity = env.scene[trolley_cfg.name]
  trolley_acc = asset.data.joint_acc[:, trolley_cfg.joint_ids].squeeze(-1)
  
  # Reward based on low acceleration using Gaussian decay
  # Encourages smooth acceleration/deceleration
  acc_smoothness = torch.exp(-0.5 * (torch.abs(trolley_acc) / acc_std) ** 2)
  
  return acc_smoothness




def qc_pendulum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # Create separate SceneEntityCfg instances for each manager to avoid mutation conflicts
  actor_terms = {
    "trolley_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",))},
    ),
    "trolley_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",))},
    ),
    "prev_trolley_pos": ObservationTermCfg(
      func=previous_trolley_pos_obs,
      params={"asset_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",))},
    ),
    "prev_trolley_vel": ObservationTermCfg(
      func=previous_trolley_vel_obs,
      params={"asset_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",))},
    ),
    "prev_action": ObservationTermCfg(
      func=previous_action_obs,
      params={"action_name": "position"},
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
      entity_name="qc_pendulum",
      actuator_names=("trolley_joint",),
      scale=1.0,  
    ),
  }

  commands = {
    "trolley_target": TrolleyTargetCommandCfg(
      resampling_time_range=(2.0, 5.0),
      entity_name="qc_pendulum",
      target_range=(-0.1, 1.5),
      initial_target=0.7,
      mode="random",
      debug_vis=False,
    ),
  }

  events = {
    "reset_trolley": EventTermCfg(
      func=reset_trolley_with_ball,
      mode="reset",
      params={
        # For negative angles, keep a small margin from the lower limit (0.0)
        # to avoid the trolley being pinned against the joint stop.
        "position_range": (0.002, 1.4),
        "velocity_range": (-0.4, 0.4),
        "angle_range_deg": (-10.0, 10.0),
        "trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)),
      },
    ),
  }

  rewards = {
    "trolley_track_pos_reward": RewardTermCfg(
      func=trolley_track_pos_reward,
      weight=1.0,
      params={"trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)), "pos_std": 0.5},
    ),
    "trolley_vel_smoothness": RewardTermCfg(
      func=trolley_vel_smoothness,
      weight=0.4,
      params={"trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)), "vel_std": 3.5},
    ),    
    "trolley_acc_smoothness": RewardTermCfg(
      func=trolley_acceleration_smoothness_reward,
      weight=0.2,
      params={"trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)), "acc_std": 1.5},
    ),    
    "trolley_out_of_bounds_penalty": RewardTermCfg(
      func=trolley_out_of_bounds_penalty,
      weight=1.0,
      params={"trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)), "min_pos": 0.0, "max_pos": 1.4},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "out_of_bounds": TerminationTermCfg(
      func=trolley_out_of_bounds,
      params={"trolley_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)), "min_pos": 0.0, "max_pos": 1.4},
    ),
  }
  
  if play:
    episode_length_s = 1e10
    observations["actor"].enable_corruption = False
  else:
    episode_length_s = 50.0

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"qc_pendulum": get_qc_pendulum_robot_cfg()},
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
      entity_name="qc_pendulum",
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