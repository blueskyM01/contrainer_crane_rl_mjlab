from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfg import cartpole_env_cfg
from .rl_cfg import cartpole_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Cartpole",
  env_cfg=cartpole_env_cfg(play=False, swing_up=True),
  play_env_cfg=cartpole_env_cfg(play=True, swing_up=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)