from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import qc_anti_sway_alignment_env_cfg
from .rl_cfg import qc_anti_sway_alignment_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-QcAntiSwayAlignment",
  env_cfg=qc_anti_sway_alignment_env_cfg(play=False),
  play_env_cfg=qc_anti_sway_alignment_env_cfg(play=True),
  rl_cfg=qc_anti_sway_alignment_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)