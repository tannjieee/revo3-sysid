"""BrainCo REVO3 Ultra21 50 Hz closed-loop system identification and MuJoCo/Isaac Lab asset exporter."""
from __future__ import annotations

from .closedloop import (
    ActuatorChannelParams,
    DEFAULT_GRAVITY,
    FINGER_SLICES,
    Revo3ClosedLoopConfig,
    Revo3ClosedLoopModel,
    T_DIFF_45,
    ThermalCurrentLimit,
)
from .controller import Revo3Controller
from .data import Revo3Trial, TrialFragment, evaluate_fragments, extract_fragments, load_trial_unified
from .device import DOF, JOINT_NAMES, Revo3Device, discover
from .isaaclab_actuator import HAS_ISAACLAB, Revo3DifferentialThermalNumpyCore
from .isaaclab_export import compute_isaaclab_joint_params, export_identified_usda, export_isaaclab_assets
from .model import Revo3Clearance, Revo3HandModel, Revo3Sim, mjcf_joint_names
from .paths import (
    default_assets_dir,
    default_plans_dir,
    default_pretrained_dir,
    default_pretrained_json,
    default_usd_path,
    default_xml_path,
)
from .pipeline import evaluate_dataset, export_all_assets, run_full_pipeline

__all__ = [
    "DOF",
    "JOINT_NAMES",
    "DEFAULT_GRAVITY",
    "FINGER_SLICES",
    "T_DIFF_45",
    "HAS_ISAACLAB",
    "ActuatorChannelParams",
    "ThermalCurrentLimit",
    "Revo3ClosedLoopConfig",
    "Revo3ClosedLoopModel",
    "Revo3Device",
    "Revo3Controller",
    "Revo3HandModel",
    "Revo3Clearance",
    "Revo3Sim",
    "Revo3Trial",
    "TrialFragment",
    "Revo3DifferentialThermalNumpyCore",
    "discover",
    "mjcf_joint_names",
    "load_trial_unified",
    "extract_fragments",
    "evaluate_fragments",
    "compute_isaaclab_joint_params",
    "export_identified_usda",
    "export_isaaclab_assets",
    "export_all_assets",
    "evaluate_dataset",
    "run_full_pipeline",
    "default_assets_dir",
    "default_xml_path",
    "default_usd_path",
    "default_plans_dir",
    "default_pretrained_dir",
    "default_pretrained_json",
]

__version__ = "0.1.0"
