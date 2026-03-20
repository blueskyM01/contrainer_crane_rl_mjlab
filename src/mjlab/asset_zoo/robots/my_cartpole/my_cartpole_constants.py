from pathlib import Path
import mujoco, math

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg

MY_CARTPOLE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "my_cartpole" / "xmls" / "my_cartpole.xml"
)
assert MY_CARTPOLE_XML.exists(), f"XML not found: {MY_CARTPOLE_XML}"


_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)

_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi},
  joint_vel={".*": 0.0},
)

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MY_CARTPOLE_XML))

def get_cartpole_robot_cfg(swing_up: bool = False) -> EntityCfg:
  """Get a fresh CartPole robot configuration instance."""

  articulation = EntityArticulationInfoCfg(
    actuators=(XmlMotorActuatorCfg(target_names_expr=("slider",)),),
  )
  return EntityCfg(
    spec_fn=get_spec,
    articulation=articulation,
    init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT
    )
  
