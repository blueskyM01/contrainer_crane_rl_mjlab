from pathlib import Path
import mujoco, math

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg, XmlPositionActuatorCfg

QC_ANTI_SWAY_ALIGNMENT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "qc_anti_sway_alignment" / "xmls" / "qc_anti_sway_alignment.xml"
)
assert QC_ANTI_SWAY_ALIGNMENT_XML.exists(), f"XML not found: {QC_ANTI_SWAY_ALIGNMENT_XML}"

FROZEN_HOIST_JOINT_POS = 0.0


_TROLLEY_HOIST_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"trolley_joint": 0.0, "hoist_joint": FROZEN_HOIST_JOINT_POS},
  joint_vel={".*": 0.0},
)

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(QC_ANTI_SWAY_ALIGNMENT_XML))

def get_qc_anti_sway_alignment_robot_cfg() -> EntityCfg:
  """Get a fresh QC Anti-Sway Alignment robot configuration instance."""

  articulation = EntityArticulationInfoCfg(
    actuators=(
      XmlPositionActuatorCfg(target_names_expr=("trolley_joint",)),
      XmlPositionActuatorCfg(target_names_expr=("hoist_joint",)),
    ),
  )
  return EntityCfg(
    spec_fn=get_spec,
    articulation=articulation,
    init_state=_TROLLEY_HOIST_INIT
    )
  