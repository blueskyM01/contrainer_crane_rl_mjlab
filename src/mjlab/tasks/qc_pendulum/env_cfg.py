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
  reset_joints_by_offset,
  time_out,
)

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


def qc_pendulum_smooth_reward(
  env: ManagerBasedRlEnv,
  trolley_cfg: SceneEntityCfg,
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
  pos_stop = _track_lin_pos_exp(trolley_pos, cmd_pos=target_cmd, std=0.125)


  
  return pos_stop



def qc_pendulum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  trolley_cfg = SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",))

  actor_terms = {
    "trolley_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": trolley_cfg},
    ),
    # "pole_angle": ObservationTermCfg(
    #   func=pole_angle_cos_sin,
    #   params={"asset_cfg": hinge_cfg},
    # ),
    "trolley_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": trolley_cfg},
    ),
    "command": ObservationTermCfg(
      func=generated_commands_obs,
      params={"command_name": "trolley_target"},
    ),
    # "pole_vel": ObservationTermCfg(
    #   func=joint_vel_rel,
    #   params={"asset_cfg": hinge_cfg},
    # ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "position": JointPositionActionCfg(
      entity_name="qc_pendulum",
      actuator_names=("trolley_joint",),
      scale=1.0,  # 稍微降低scale以获得更稳定的控制
    ),
  }

  commands = {
    "trolley_target": TrolleyTargetCommandCfg(
      resampling_time_range=(10.0, 10.0),
      entity_name="qc_pendulum",
      target_range=(0.0, 1.4),
      initial_target=0.7,
      mode="random",
      debug_vis=False,
    ),
  }

  events = {
    "reset_trolley": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.2),
        "velocity_range": (0.0, 0.01),
        "asset_cfg": SceneEntityCfg("qc_pendulum", joint_names=("trolley_joint",)),
      },
    ),
  }

  rewards = {
    "smooth_reward": RewardTermCfg(
      func=qc_pendulum_smooth_reward,
      weight=1.0,
      params={"trolley_cfg": trolley_cfg},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
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