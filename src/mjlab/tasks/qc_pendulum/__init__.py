from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfg import qc_pendulum_env_cfg
from .rl_cfg import qc_pendulum_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-QcPendulum",
  env_cfg=qc_pendulum_env_cfg(play=False),
  play_env_cfg=qc_pendulum_env_cfg(play=True),
  rl_cfg=qc_pendulum_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)