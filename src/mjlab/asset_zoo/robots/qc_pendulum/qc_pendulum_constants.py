from pathlib import Path
import mujoco, math

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg, XmlPositionActuatorCfg

QC_PENDULUM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "qc_pendulum" / "xmls" / "qc_pendulum.xml"
)
assert QC_PENDULUM_XML.exists(), f"XML not found: {QC_PENDULUM_XML}"


_TROLLEY_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"trolley_joint": 0.0},
  joint_vel={".*": 0.0},
)

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(QC_PENDULUM_XML))

def get_qc_pendulum_robot_cfg() -> EntityCfg:
  """Get a fresh QC Pendulum robot configuration instance."""

  articulation = EntityArticulationInfoCfg(
    actuators=(XmlPositionActuatorCfg(target_names_expr=("trolley_joint",)),),
  )
  return EntityCfg(
    spec_fn=get_spec,
    articulation=articulation,
    init_state=_TROLLEY_INIT
    )
  