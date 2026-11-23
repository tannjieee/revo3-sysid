"""Export identified REVO3 21-DOF parameters as portable Isaac Lab / OpenUSD assets.

Generates three complementary deliverables in the chosen output directory:
  1. ``revo3_right_identified.usda`` — Lightweight OpenUSD sublayer referencing the
     base ``revo3_right.usd`` (zero binary duplication unless ``standalone=True``)
     and authoring identified PhysX joint armature, Coulomb friction, and drive defaults.
  2. ``isaaclab_actuator_params.json`` — Clean machine-readable joint-by-joint Isaac Lab
     actuator parameter table (including 45-deg J19/J20 differential + i^2*t thermal data).
  3. ``isaaclab_revo3_cfg.py`` — Ready-to-import Python module providing:
       * ``make_revo3_delayed_pd_cfg(...)`` (standard Isaac Lab ``DelayedPDActuatorCfg``)
       * ``make_revo3_implicit_cfg(...)`` (fast RL ``ImplicitActuatorCfg``)
       * ``make_revo3_differential_thermal_cfg(...)`` (full-fidelity 45-deg diff + ``i^2*t``)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np

from .closedloop import Revo3ClosedLoopConfig
from .device import DOF, JOINT_NAMES
from .model import mjcf_joint_names
from .paths import default_usd_path


def compute_isaaclab_joint_params(
    config: Revo3ClosedLoopConfig,
    *,
    kp_cmd: float = 2.0,
    kd_cmd: float = 0.2,
    sim_dt: float = 0.002,
) -> dict:
    """Convert identified 21-DOF closed-loop config into per-joint Isaac Lab parameters."""
    mj_names = mjcf_joint_names(config.side)
    armature: dict[str, float] = {}
    friction: dict[str, float] = {}
    viscous_friction: dict[str, float] = {}
    stiffness_pd: dict[str, float] = {}
    damping_pd: dict[str, float] = {}
    effort_limit_cont: dict[str, float] = {}
    effort_limit_peak: dict[str, float] = {}
    kp_scale_map: dict[str, float] = {}
    kd_scale_map: dict[str, float] = {}
    delay_s_map: dict[str, float] = {}
    passive_spring_k: dict[str, float] = {}
    passive_spring_ref: dict[str, float] = {}

    for jid, jname in enumerate(mj_names):
        if config.thumb_differential and jid in (19, 20):
            # Orthonormal 45-deg T => diagonal joint-space equivalent is mean of the two modes;
            # full cross-axis coupling & i^2*t live in Revo3MITDifferentialActuator.
            c0, c1 = config.channels[19], config.channels[20]
            mode = jid - 19
            arm = 0.5 * (c0.armature + c1.armature)
            damp = 0.5 * (c0.damping + c1.damping)
            fric = 0.5 * (c0.frictionloss + c1.frictionloss)
            kps = 0.5 * (c0.kp_scale + c1.kp_scale)
            kds = 0.5 * (c0.kd_scale + c1.kd_scale)
            ic = max(c0.current_limit.i_cont, c1.current_limit.i_cont)
            ip = max(c0.current_limit.i_peak, c1.current_limit.i_peak)
            d_s = 0.5 * (c0.delay_s + c1.delay_s)
            sk = float(config.thumb_joint_stiffness[mode])
            sref = float(config.thumb_joint_springref[mode])
        else:
            ch = config.channels[jid]
            arm = ch.armature
            damp = ch.damping
            fric = ch.frictionloss
            kps = ch.kp_scale
            kds = ch.kd_scale
            ic = ch.current_limit.i_cont
            ip = ch.current_limit.i_peak
            d_s = ch.delay_s
            sk = ch.stiffness
            sref = ch.springref

        armature[jname] = round(float(arm), 6)
        friction[jname] = round(float(fric), 6)
        viscous_friction[jname] = round(float(damp), 6)
        kp_scale_map[jname] = round(float(kps), 6)
        kd_scale_map[jname] = round(float(kds), 6)
        stiffness_pd[jname] = round(float(kps * kp_cmd), 6)
        damping_pd[jname] = round(float(kds * kd_cmd), 6)
        effort_limit_cont[jname] = round(float(ic), 6)
        effort_limit_peak[jname] = round(float(ip), 6)
        delay_s_map[jname] = round(float(d_s), 6)
        passive_spring_k[jname] = round(float(sk), 6)
        passive_spring_ref[jname] = round(float(sref), 6)

    delays = list(delay_s_map.values())
    mean_delay_s = float(np.mean(delays))
    min_delay_steps = max(0, int(round(min(delays) / sim_dt)))
    max_delay_steps = max(0, int(round(max(delays) / sim_dt)))
    mean_delay_steps = max(0, int(round(mean_delay_s / sim_dt)))

    return dict(
        format="revo3-sysid-isaaclab-v1",
        side=config.side,
        reference_mit_gains=dict(kp_cmd=float(kp_cmd), kd_cmd=float(kd_cmd)),
        sim_dt_s=float(sim_dt),
        delay_summary=dict(
            mean_delay_s=round(mean_delay_s, 6),
            min_delay_steps=min_delay_steps,
            mean_delay_steps=mean_delay_steps,
            max_delay_steps=max_delay_steps,
        ),
        thumb_differential=dict(
            enabled=bool(config.thumb_differential),
            joints=[mj_names[19], mj_names[20]],
            T_matrix=config.thumb_T,
            note="For exact 45-deg J19/J20 cross-axis coupling and i^2*t thermal limiting, use make_revo3_differential_thermal_cfg().",
        ),
        joint_order_sdk_to_usd=dict(zip(JOINT_NAMES, mj_names)),
        armature=armature,
        friction=friction,
        viscous_friction=viscous_friction,
        kp_scale=kp_scale_map,
        kd_scale=kd_scale_map,
        stiffness=stiffness_pd,
        damping=damping_pd,
        effort_limit_cont=effort_limit_cont,
        effort_limit_peak=effort_limit_peak,
        delay_s=delay_s_map,
        passive_spring_stiffness=passive_spring_k,
        passive_spring_ref=passive_spring_ref,
    )


def export_identified_usda(
    config: Revo3ClosedLoopConfig,
    out_usda: Path | str,
    *,
    base_usd: Path | str | None = None,
    kp_cmd: float = 2.0,
    kd_cmd: float = 0.2,
    standalone: bool = False,
) -> Path:
    """Write a portable OpenUSD ASCII overlay authoring identified PhysX joint properties.

    References ``base_usd`` via a relative sublayer path so the file is only ~6 KB.
    When ``standalone=True``, copies ``revo3_right.usd`` and ``textures/`` alongside
    ``out_usda`` so the output folder is 100% self-contained.
    """
    out_path = Path(out_usda).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_usd = default_usd_path(config.side) if base_usd is None else Path(base_usd).expanduser().resolve(strict=True)

    if standalone:
        dst_usd = out_path.parent / src_usd.name
        if src_usd != dst_usd:
            shutil.copy2(src_usd, dst_usd)
        src_tex = src_usd.parent / "textures"
        dst_tex = out_path.parent / "textures"
        if src_tex.is_dir() and src_tex.resolve() != dst_tex.resolve():
            shutil.copytree(src_tex, dst_tex, dirs_exist_ok=True)
        sublayer_rel = f"./{src_usd.name}"
    else:
        sublayer_rel = Path(os.path.relpath(src_usd, out_path.parent)).as_posix()

    params = compute_isaaclab_joint_params(config, kp_cmd=kp_cmd, kd_cmd=kd_cmd)
    mj_names = mjcf_joint_names(config.side)

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "revo3_single"',
        '    doc = "Identified REVO3 Ultra21 right-hand OpenUSD overlay (PhysX / Isaac Lab / Newton)."',
        "    metersPerUnit = 1",
        "    subLayers = [",
        f"        @{sublayer_rel}@",
        "    ]",
        '    upAxis = "Z"',
        ")",
        "",
        'over "revo3_single"',
        "{",
        '    over "joints"',
        "    {",
    ]
    for jname in mj_names:
        arm = params["armature"][jname]
        fric = params["friction"][jname]
        stiff = params["stiffness"][jname]
        # Total angular drive damping = active PD damping + passive viscous damping
        damp_tot = round(params["damping"][jname] + params["viscous_friction"][jname], 6)
        max_f = params["effort_limit_cont"][jname]
        lines.extend([
            f'        over "{jname}" (',
            '            prepend apiSchemas = ["PhysxJointAPI"]',
            "        )",
            "        {",
            f"            float physxJoint:armature = {arm}",
            f"            float physxJoint:jointFriction = {fric}",
            f"            float drive:angular:physics:stiffness = {stiff}",
            f"            float drive:angular:physics:damping = {damp_tot}",
            f"            float drive:angular:physics:maxForce = {max_f}",
            "        }",
        ])
    lines.extend([
        "    }",
        "}",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


_CFG_TEMPLATE = '''"""Auto-generated Isaac Lab ArticulationCfg for BrainCo REVO3 Ultra21 ({side} hand).

Generated by ``revo3-sysid`` with zero hardcoded paths. All USD and JSON paths resolve
relative to this file so you can move or share the entire folder freely.

Quick usage inside an Isaac Lab script::

    from isaaclab_revo3_cfg import (
        make_revo3_delayed_pd_cfg,          # Standard Isaac Lab DelayedPDActuatorCfg
        make_revo3_implicit_cfg,            # Lightweight ImplicitActuatorCfg for fast RL
        make_revo3_differential_thermal_cfg,# Exact 45-deg J19/J20 differential + i^2*t thermal limit
    )

    robot_cfg = make_revo3_delayed_pd_cfg(kp_cmd=2.0, kd_cmd=0.2, sim_dt=0.002)
"""
from __future__ import annotations

import json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
USD_PATH = (_THIS_DIR / "{usda_name}").resolve()
PARAMS_JSON_PATH = (_THIS_DIR / "{params_json_name}").resolve()
CLOSEDLOOP_JSON_PATH = (_THIS_DIR / "{closedloop_json_name}").resolve()

IDENTIFIED_PARAMS = json.loads(PARAMS_JSON_PATH.read_text(encoding="utf-8"))


def _scaled_gains(kp_cmd: float, kd_cmd: float) -> tuple[dict[str, float], dict[str, float]]:
    stiffness = {{j: float(s) * float(kp_cmd) for j, s in IDENTIFIED_PARAMS["kp_scale"].items()}}
    damping = {{j: float(s) * float(kd_cmd) for j, s in IDENTIFIED_PARAMS["kd_scale"].items()}}
    return stiffness, damping


def _spawn_cfg(usd_path: Path | str | None = None):
    import isaaclab.sim as sim_utils

    path = USD_PATH if usd_path is None else Path(usd_path).expanduser().resolve()
    return sim_utils.UsdFileCfg(
        usd_path=str(path),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    )


def _filter_cfg_kwargs(cls, kwargs: dict) -> dict:
    """Keep only fields supported by the installed Isaac Lab version (1.x / 2.x compatible)."""
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        return kwargs
    return {{k: v for k, v in kwargs.items() if k in fields}}


def make_revo3_delayed_pd_cfg(
    *,
    kp_cmd: float = {kp_cmd},
    kd_cmd: float = {kd_cmd},
    sim_dt: float = {sim_dt},
    use_peak_effort: bool = False,
    prim_path: str = "{{ENV_REGEX_NS}}/Robot",
    usd_path: Path | str | None = None,
):
    """Standard Isaac Lab ArticulationCfg using DelayedPDActuatorCfg (no custom class needed)."""
    from isaaclab.actuators import DelayedPDActuatorCfg
    from isaaclab.assets import ArticulationCfg

    stiffness, damping = _scaled_gains(kp_cmd, kd_cmd)
    effort = IDENTIFIED_PARAMS["effort_limit_peak" if use_peak_effort else "effort_limit_cont"]
    delays = list(IDENTIFIED_PARAMS["delay_s"].values())
    min_d = max(0, int(round(min(delays) / sim_dt)))
    max_d = max(min_d, int(round(max(delays) / sim_dt)))

    act_kwargs = _filter_cfg_kwargs(
        DelayedPDActuatorCfg,
        dict(
            joint_names_expr=[".*_joint"],
            stiffness=stiffness,
            damping=damping,
            armature=IDENTIFIED_PARAMS["armature"],
            friction=IDENTIFIED_PARAMS["friction"],
            dynamic_friction=IDENTIFIED_PARAMS["friction"],
            viscous_friction=IDENTIFIED_PARAMS["viscous_friction"],
            effort_limit_sim=effort,
            effort_limit=effort,
            min_delay=min_d,
            max_delay=max_d,
        ),
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_spawn_cfg(usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={{"revo3_hand": DelayedPDActuatorCfg(**act_kwargs)}},
    )


def make_revo3_implicit_cfg(
    *,
    kp_cmd: float = {kp_cmd},
    kd_cmd: float = {kd_cmd},
    use_peak_effort: bool = False,
    prim_path: str = "{{ENV_REGEX_NS}}/Robot",
    usd_path: Path | str | None = None,
):
    """Lightweight PhysX ImplicitActuatorCfg for maximum GPU throughput in RL training."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    stiffness, damping = _scaled_gains(kp_cmd, kd_cmd)
    effort = IDENTIFIED_PARAMS["effort_limit_peak" if use_peak_effort else "effort_limit_cont"]

    act_kwargs = _filter_cfg_kwargs(
        ImplicitActuatorCfg,
        dict(
            joint_names_expr=[".*_joint"],
            stiffness=stiffness,
            damping=damping,
            armature=IDENTIFIED_PARAMS["armature"],
            friction=IDENTIFIED_PARAMS["friction"],
            dynamic_friction=IDENTIFIED_PARAMS["friction"],
            viscous_friction=IDENTIFIED_PARAMS["viscous_friction"],
            effort_limit_sim=effort,
            effort_limit=effort,
        ),
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_spawn_cfg(usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={{"revo3_hand": ImplicitActuatorCfg(**act_kwargs)}},
    )


def make_revo3_differential_thermal_cfg(
    *,
    kp_cmd: float = {kp_cmd},
    kd_cmd: float = {kd_cmd},
    sim_dt: float = {sim_dt},
    use_i2t: bool = True,
    include_passive_spring: bool = True,
    prim_path: str = "{{ENV_REGEX_NS}}/Robot",
    usd_path: Path | str | None = None,
):
    """Full-fidelity Isaac Lab ArticulationCfg with 45-deg J19/J20 differential + i^2*t limiter.

    Requires ``revo3-sysid`` installed in the Isaac Lab Python environment.
    """
    from isaaclab.assets import ArticulationCfg
    from revo3_sysid.isaaclab_actuator import Revo3MITDifferentialActuatorCfg

    effort_peak = IDENTIFIED_PARAMS["effort_limit_peak"]
    act_kwargs = _filter_cfg_kwargs(
        Revo3MITDifferentialActuatorCfg,
        dict(
            joint_names_expr=[".*_joint"],
            config_json=str(CLOSEDLOOP_JSON_PATH),
            kp_cmd=float(kp_cmd),
            kd_cmd=float(kd_cmd),
            sim_dt=float(sim_dt),
            use_i2t=bool(use_i2t),
            include_passive_spring=bool(include_passive_spring),
            armature=IDENTIFIED_PARAMS["armature"],
            friction=IDENTIFIED_PARAMS["friction"],
            dynamic_friction=IDENTIFIED_PARAMS["friction"],
            viscous_friction=IDENTIFIED_PARAMS["viscous_friction"],
            effort_limit_sim=effort_peak,
            effort_limit=effort_peak,
        ),
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_spawn_cfg(usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={{"revo3_hand": Revo3MITDifferentialActuatorCfg(**act_kwargs)}},
    )
'''


def export_isaaclab_assets(
    config: Revo3ClosedLoopConfig,
    out_dir: Path | str,
    *,
    base_usd: Path | str | None = None,
    closedloop_json_name: str = "revo3_right_identified.json",
    kp_cmd: float = 2.0,
    kd_cmd: float = 0.2,
    sim_dt: float = 0.002,
    standalone: bool = False,
) -> dict[str, Path]:
    """Export USDA overlay, Isaac Lab actuator JSON, and ready-to-import ArticulationCfg."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    usda_name = f"revo3_{config.side}_identified.usda"
    params_name = "isaaclab_actuator_params.json"
    cfg_name = "isaaclab_revo3_cfg.py"

    usda_path = export_identified_usda(
        config,
        out / usda_name,
        base_usd=base_usd,
        kp_cmd=kp_cmd,
        kd_cmd=kd_cmd,
        standalone=standalone,
    )
    params = compute_isaaclab_joint_params(config, kp_cmd=kp_cmd, kd_cmd=kd_cmd, sim_dt=sim_dt)
    params_path = out / params_name
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    cfg_path = out / cfg_name
    cfg_code = _CFG_TEMPLATE.format(
        side=config.side,
        usda_name=usda_name,
        params_json_name=params_name,
        closedloop_json_name=closedloop_json_name,
        kp_cmd=kp_cmd,
        kd_cmd=kd_cmd,
        sim_dt=sim_dt,
    )
    cfg_path.write_text(cfg_code, encoding="utf-8")

    return dict(
        usda=usda_path,
        params_json=params_path,
        cfg_py=cfg_path,
    )
