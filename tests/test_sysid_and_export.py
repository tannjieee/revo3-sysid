"""Tests for Layer 2 closed-loop simulator, native C kernel, and portable MuJoCo/Isaac Lab export."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import pytest

from revo3_sysid import (
    DOF,
    Revo3ClosedLoopConfig,
    Revo3ClosedLoopModel,
    Revo3DifferentialThermalNumpyCore,
    ThermalCurrentLimit,
    default_pretrained_json,
    default_usd_path,
    default_xml_path,
    export_all_assets,
    mjcf_joint_names,
)
from revo3_sysid._c_rollout import get_c_rollout_lib
from revo3_sysid.cli import main as cli_main


def test_bundled_assets_load_cleanly():
    xml = default_xml_path("right")
    usd = default_usd_path("right")
    assert xml.is_file()
    assert usd.is_file()
    m = mj.MjModel.from_xml_path(str(xml))
    assert m.njnt == 21
    assert m.nmesh >= 140


def test_c_kernel_matches_python_reference_bit_exact():
    cfg = Revo3ClosedLoopConfig.load(default_pretrained_json())
    sim = Revo3ClosedLoopModel(default_xml_path("right"), cfg, finger="thumb")

    ct = np.arange(50, dtype=float) * 0.02
    qt = np.zeros((50, 5), dtype=float)
    vt = np.zeros((50, 5), dtype=float)
    qt[:, 3] = 0.35 * np.sin(2.0 * np.pi * 1.2 * ct)
    qt[:, 4] = -0.25 * np.sin(2.0 * np.pi * 0.9 * ct)
    vt[:, 3] = 0.35 * 2.0 * np.pi * 1.2 * np.cos(2.0 * np.pi * 1.2 * ct)
    vt[:, 4] = -0.25 * 2.0 * np.pi * 0.9 * np.cos(2.0 * np.pi * 0.9 * ct)

    q0 = np.zeros(5)
    v0 = np.zeros(5)
    q_py, u_py = sim.rollout(q0, v0, ct, ct, qt, vt, kp=3.5, kd=0.1, use_i2t=True, return_currents=True, backend="python")
    if get_c_rollout_lib() is not None:
        q_c, u_c = sim.rollout(q0, v0, ct, ct, qt, vt, kp=3.5, kd=0.1, use_i2t=True, return_currents=True, backend="c")
        np.testing.assert_allclose(q_c, q_py, atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(u_c, u_py, atol=1e-11, rtol=0.0)


def test_export_all_assets_relative_and_standalone(tmp_path: Path):
    cfg = Revo3ClosedLoopConfig.load(default_pretrained_json())

    # 1. Relative export
    rel_dir = tmp_path / "rel_out"
    exp_rel = export_all_assets(cfg, rel_dir, kp_cmd=2.0, kd_cmd=0.2, sim_dt=0.002, standalone=False)
    for key in ("json", "mujoco_xml", "isaaclab_usda", "isaaclab_params_json", "isaaclab_cfg_py"):
        assert exp_rel[key].is_file(), f"Missing exported file {key}"
    # Verify MuJoCo loads exported XML cleanly and has NO absolute meshdir
    m_rel = mj.MjModel.from_xml_path(str(exp_rel["mujoco_xml"]))
    assert m_rel.njnt == 21 and m_rel.nu == 21 and m_rel.ntendon == 2
    comp = ET.parse(exp_rel["mujoco_xml"]).getroot().find("compiler")
    assert comp is not None and not Path(comp.get("meshdir", "")).is_absolute()

    # Verify USDA has all 21 joint overrides & relative sublayer
    usda_text = exp_rel["isaaclab_usda"].read_text(encoding="utf-8")
    assert "revo3_right.usd@" in usda_text
    assert "physxJoint:armature" in usda_text
    for jname in mjcf_joint_names("right"):
        assert f'over "{jname}"' in usda_text

    # 2. Standalone export (copies meshes, textures, USD into output folder)
    sa_dir = tmp_path / "standalone_out"
    exp_sa = export_all_assets(cfg, sa_dir, standalone=True)
    assert (sa_dir / "meshes").is_dir()
    assert (sa_dir / "textures" / "revo3_screen.png").is_file()
    assert (sa_dir / "revo3_right.usd").is_file()
    m_sa = mj.MjModel.from_xml_path(str(exp_sa["mujoco_xml"]))
    assert m_sa.njnt == 21 and m_sa.nu == 21


def test_isaaclab_differential_thermal_core_physics():
    cfg = Revo3ClosedLoopConfig.load(default_pretrained_json())
    # Force strong differential asymmetry & fast i^2*t saturation to test both mechanisms
    cfg.channels[19].current_limit = ThermalCurrentLimit(i_cont=0.40, i_peak=0.80, tau_s=0.02)
    cfg.channels[20].current_limit = ThermalCurrentLimit(i_cont=0.40, i_peak=0.80, tau_s=0.02)

    jnames = list(mjcf_joint_names("right"))
    core = Revo3DifferentialThermalNumpyCore(
        cfg, jnames, kp_cmd=4.0, kd_cmd=0.1, sim_dt=0.002, num_envs=2, use_i2t=True, include_passive_spring=False
    )

    q_des = np.zeros((2, DOF))
    v_des = np.zeros((2, DOF))
    q_meas = np.zeros((2, DOF))
    v_meas = np.zeros((2, DOF))
    q_des[:, 19] = 1.0  # large step on J19 only

    taus = []
    for _ in range(30):
        taus.append(core.step(q_des, v_des, q_meas, v_meas).copy())

    # Initial peak torque should exceed continuous limit, then thermally decay back toward continuous limit
    tau0 = abs(taus[0][0, 19])
    tau_end = abs(taus[-1][0, 19])
    assert tau0 > 0.70, f"Expected initial transient above i_cont, got {tau0}"
    assert tau_end < tau0 - 0.15, f"Expected i^2*t thermal decay: start={tau0}, end={tau_end}"

    # Reset only env 0; env 1 thermal state must remain elevated
    core.reset([0])
    assert core.s_therm[0, 19] == 0.0
    assert core.s_therm[1, 19] > 0.5


def test_cli_demo_and_dry_run_collect(tmp_path: Path):
    assert cli_main(["demo"]) == 0
    assert cli_main(["collect", "--mode", "quick", "--out", str(tmp_path / "dry")]) == 0
    # Dry-run MUST NOT create hardware captures
    assert not (tmp_path / "dry" / "manifest.json").exists()
