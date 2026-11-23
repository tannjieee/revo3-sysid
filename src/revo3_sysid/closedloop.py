"""Unified closed-loop MuJoCo physics engine for REVO3 Ultra21.

Supports:
  - Single-finger or full 21-DOF models built directly from BrainCo's passive MJCF.
  - Configurable transmission coupling (diagonal for J00-J18, 45-deg fixed-tendon
    differential (J19 ± J20)/sqrt(2) for the thumb differential pair).
  - Per-channel delay, multi-gain MIT current law (Kp, Kd, vel_mult, bias).
  - Two-tier leaky-integrator i^2*t thermal-budget current limiter modeling both
    transient peak current (I_peak) and continuous sustained limit (I_cont).
  - Export of deployable 21-DOF MuJoCo XML + versioned parameter JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import mujoco.rollout as mr
import numpy as np

from ._c_rollout import get_c_rollout_lib, run_c_rollout
from .device import DOF, JOINT_NAMES
from .model import mjcf_joint_names

FINGER_SLICES: dict[str, tuple[int, ...]] = {
    "little": (0, 1, 2, 3),
    "ring": (4, 5, 6, 7),
    "middle": (8, 9, 10, 11),
    "index": (12, 13, 14, 15),
    "thumb": (16, 17, 18, 19, 20),
    "full": tuple(range(21)),
}

T_DIFF_45 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float) / np.sqrt(2.0)
DEFAULT_GRAVITY = np.array([-9.81, 0.0, 0.0], dtype=float)


@dataclass
class ThermalCurrentLimit:
    """Two-tier leaky-integrator i^2*t current limiter.

    When i_peak <= i_cont (or tau_s <= 0), reduces to a static symmetric clip at ±i_cont.
    Otherwise, normalized thermal state s in [0, 1] evolves via:
        ds/dt = (u^2 - i_cont^2) / ((i_peak^2 - i_cont^2) * tau_s)
    clipped to [0, 1], with dynamic ceiling:
        i_lim(s) = i_peak - (i_peak - i_cont) * s
    """
    i_cont: float = 0.60
    i_peak: float = 0.60
    tau_s: float = 0.08

    @property
    def is_dynamic(self) -> bool:
        return self.i_peak > self.i_cont + 1e-9 and self.tau_s > 0.0

    def to_dict(self) -> dict:
        return dict(i_cont=float(self.i_cont), i_peak=float(self.i_peak), tau_s=float(self.tau_s))

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalCurrentLimit":
        return cls(i_cont=float(d["i_cont"]), i_peak=float(d["i_peak"]), tau_s=float(d.get("tau_s", 0.08)))


@dataclass
class ActuatorChannelParams:
    """Closed-loop mechanical + electrical parameters for one actuator channel (joint or tendon mode)."""
    armature: float = 0.001
    damping: float = 0.02
    frictionloss: float = 0.01
    kp_scale: float = 2.45
    kd_scale: float = 1.0
    vel_mult: float = 1.0
    delay_s: float = 0.010
    bias: float = 0.0
    stiffness: float = 0.0
    springref: float = 0.0
    current_limit: ThermalCurrentLimit = field(default_factory=ThermalCurrentLimit)

    def to_dict(self) -> dict:
        return dict(
            armature=float(self.armature),
            damping=float(self.damping),
            frictionloss=float(self.frictionloss),
            kp_scale=float(self.kp_scale),
            kd_scale=float(self.kd_scale),
            vel_mult=float(self.vel_mult),
            delay_s=float(self.delay_s),
            bias=float(self.bias),
            stiffness=float(self.stiffness),
            springref=float(self.springref),
            current_limit=self.current_limit.to_dict(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ActuatorChannelParams":
        cl = ThermalCurrentLimit.from_dict(d["current_limit"]) if "current_limit" in d else ThermalCurrentLimit()
        return cls(
            armature=float(d["armature"]),
            damping=float(d["damping"]),
            frictionloss=float(d["frictionloss"]),
            kp_scale=float(d["kp_scale"]),
            kd_scale=float(d["kd_scale"]),
            vel_mult=float(d.get("vel_mult", 1.0)),
            delay_s=float(d.get("delay_s", 0.010)),
            bias=float(d.get("bias", 0.0)),
            stiffness=float(d.get("stiffness", 0.0)),
            springref=float(d.get("springref", 0.0)),
            current_limit=cl,
        )


@dataclass
class Revo3ClosedLoopConfig:
    """Complete 21-DOF closed-loop parameter specification (format: aiowuji-revo3-closedloop-v2)."""
    side: str = "right"
    dt: float = 0.001
    gravity: list[float] = field(default_factory=lambda: DEFAULT_GRAVITY.tolist())
    thumb_differential: bool = True
    thumb_T: list[list[float]] = field(default_factory=lambda: T_DIFF_45.tolist())
    # 21 channels: indices 0..18 map to J00..J18; when thumb_differential=True,
    # indices 19, 20 are tendon modes (q+, q-) while joint springs/biases remain in joint coords.
    channels: list[ActuatorChannelParams] = field(default_factory=lambda: [ActuatorChannelParams() for _ in range(DOF)])
    thumb_joint_stiffness: list[float] = field(default_factory=lambda: [0.0, 0.0])
    thumb_joint_springref: list[float] = field(default_factory=lambda: [0.0, 0.0])
    thumb_joint_bias: list[float] = field(default_factory=lambda: [0.0, 0.0])
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(
            format="revo3-sysid-closedloop-v2",
            side=self.side,
            dt=float(self.dt),
            gravity=[float(x) for x in self.gravity],
            thumb_differential=bool(self.thumb_differential),
            thumb_T=[[float(v) for v in row] for row in self.thumb_T],
            joint_names=list(JOINT_NAMES),
            channels=[c.to_dict() for c in self.channels],
            thumb_joint_stiffness=[float(x) for x in self.thumb_joint_stiffness],
            thumb_joint_springref=[float(x) for x in self.thumb_joint_springref],
            thumb_joint_bias=[float(x) for x in self.thumb_joint_bias],
            metadata=dict(self.metadata),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "Revo3ClosedLoopConfig":
        raw = json.loads(Path(path).read_text())
        if raw.get("format") not in ("revo3-sysid-closedloop-v2", "aiowuji-revo3-closedloop-v2"):
            raise ValueError(f"Unsupported closed-loop config format: {raw.get('format')!r}")
        return cls(
            side=raw["side"],
            dt=float(raw["dt"]),
            gravity=[float(x) for x in raw["gravity"]],
            thumb_differential=bool(raw["thumb_differential"]),
            thumb_T=raw["thumb_T"],
            channels=[ActuatorChannelParams.from_dict(c) for c in raw["channels"]],
            thumb_joint_stiffness=[float(x) for x in raw["thumb_joint_stiffness"]],
            thumb_joint_springref=[float(x) for x in raw["thumb_joint_springref"]],
            thumb_joint_bias=[float(x) for x in raw["thumb_joint_bias"]],
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def from_v2_artifacts(
        cls,
        aio_params_path: Path | str,
        thumb_v2_path: Path | str | None = None,
        j10_v2_override: dict | None = None,
        gravity: np.ndarray = DEFAULT_GRAVITY,
    ) -> "Revo3ClosedLoopConfig":
        """Assemble a complete 21-DOF config from AIO 120 baseline + Thumb V2 + optional J10 V2."""
        base = json.loads(Path(aio_params_path).read_text())
        channels = [ActuatorChannelParams() for _ in range(DOF)]
        thumb_k = [0.0, 0.0]
        thumb_ref = [0.0, 0.0]
        thumb_bias = [0.0, 0.0]
        clip_default = float(base.get("simulated_current_clip_a", 0.60))

        for jid in range(DOF):
            jd = base["joints"][f"J{jid:02d}"]
            channels[jid] = ActuatorChannelParams(
                armature=float(jd["armature"]),
                damping=float(jd["damping"]),
                frictionloss=float(jd["frictionloss"]),
                stiffness=float(jd["stiffness"]),
                springref=float(jd["springref"]),
                kp_scale=float(jd["kp_scale"]),
                kd_scale=float(jd["kd_scale"]),
                vel_mult=1.0,
                bias=float(jd["bias_A"]),
                delay_s=float(jd["delay_s"]),
                current_limit=ThermalCurrentLimit(i_cont=clip_default, i_peak=clip_default, tau_s=0.08),
            )
            if jid == 19:
                thumb_k[0], thumb_ref[0], thumb_bias[0] = float(jd["stiffness"]), float(jd["springref"]), float(jd["bias_A"])
            elif jid == 20:
                thumb_k[1], thumb_ref[1], thumb_bias[1] = float(jd["stiffness"]), float(jd["springref"]), float(jd["bias_A"])

        use_diff = False
        T_mat = np.eye(2).tolist()
        if thumb_v2_path is not None:
            v2 = json.loads(Path(thumb_v2_path).read_text())
            spec = v2["models"]["v2_diff_sym_motor"]
            use_diff = True
            T_mat = v2.get("T_diff", T_DIFF_45.tolist())
            mp_rows = spec["mode_parameters"]
            clip = float(spec["mode_current_clip"])
            thumb_bias = [float(x) for x in spec["bias_joint"]]
            for m_idx in range(2):
                r = mp_rows[m_idx]
                channels[19 + m_idx] = ActuatorChannelParams(
                    armature=float(r[0]),
                    damping=float(r[1]),
                    frictionloss=float(r[2]),
                    kp_scale=float(r[3]),
                    kd_scale=float(r[4]),
                    vel_mult=float(r[5]),
                    delay_s=float(r[6]),
                    bias=0.0,
                    stiffness=0.0,
                    springref=0.0,
                    current_limit=ThermalCurrentLimit(i_cont=clip, i_peak=clip, tau_s=0.08),
                )

        if j10_v2_override is not None:
            c10 = channels[10]
            for k in ("armature", "damping", "frictionloss", "stiffness", "springref", "delay_s", "kp_scale", "kd_scale", "bias"):
                if k in j10_v2_override:
                    setattr(c10, k, float(j10_v2_override[k]))
            if "i_cont" in j10_v2_override or "i_peak" in j10_v2_override:
                c10.current_limit = ThermalCurrentLimit(
                    i_cont=float(j10_v2_override.get("i_cont", c10.current_limit.i_cont)),
                    i_peak=float(j10_v2_override.get("i_peak", c10.current_limit.i_peak)),
                    tau_s=float(j10_v2_override.get("tau_s", c10.current_limit.tau_s)),
                )

        return cls(
            side="right",
            dt=0.001,
            gravity=np.asarray(gravity, float).tolist(),
            thumb_differential=use_diff,
            thumb_T=T_mat,
            channels=channels,
            thumb_joint_stiffness=thumb_k,
            thumb_joint_springref=thumb_ref,
            thumb_joint_bias=thumb_bias,
            metadata=dict(
                aio_source=str(Path(aio_params_path).resolve()),
                thumb_v2_source=None if thumb_v2_path is None else str(Path(thumb_v2_path).resolve()),
            ),
        )


def _prune_mjcf_to_subset(xml_path: Path, side: str, finger: str) -> ET.ElementTree:
    """Load BrainCo MJCF and optionally prune to a single finger while preserving mesh paths."""
    xml_path = xml_path.expanduser().resolve(strict=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attr in ("meshdir", "texturedir", "assetdir"):
        val = compiler.get(attr)
        if val is not None:
            compiler.set(attr, str((xml_path.parent / val).resolve()))
    if compiler.get("meshdir") is None:
        compiler.set("meshdir", str(xml_path.parent.resolve()))
    if compiler.get("texturedir") is None:
        compiler.set("texturedir", str(xml_path.parent.resolve()))

    if finger != "full":
        keep_prefix = f"{side}_{finger}_"
        world = root.find("worldbody")
        base = world.find("body")
        for child in list(base):
            if child.tag == "body":
                bname = child.get("name", "")
                if not bname.startswith(keep_prefix):
                    base.remove(child)
        # Remove unused sensors / actuators / contacts / tendons if any
        for tag in ("actuator", "tendon", "contact", "equality", "sensor"):
            el = root.find(tag)
            if el is not None:
                root.remove(el)
    return tree


class Revo3ClosedLoopModel:
    """Closed-loop MuJoCo simulator supporting differential tendons and i^2*t current limiting."""

    def __init__(self, xml_path: Path | str | None, config: Revo3ClosedLoopConfig, finger: str = "full") -> None:
        if xml_path is None:
            from .paths import default_xml_path
            xml_path = default_xml_path(config.side)
        if finger not in FINGER_SLICES:
            raise ValueError(f"Unknown finger subset {finger!r}; choose from {tuple(FINGER_SLICES)}")
        self.xml_path = Path(xml_path).expanduser().resolve(strict=True)
        self.config = config
        self.finger = finger
        self.joint_indices = np.array(FINGER_SLICES[finger], dtype=int)
        all_names = mjcf_joint_names(config.side)
        self.joint_names = [all_names[i] for i in self.joint_indices]
        self.has_diff = bool(config.thumb_differential and (19 in self.joint_indices and 20 in self.joint_indices))
        self.T_diff = np.asarray(config.thumb_T, dtype=float)

        self._tree = self._build_xml_tree()
        xml_str = ET.tostring(self._tree.getroot(), encoding="unicode")
        self.model = mj.MjModel.from_xml_string(xml_str)
        self.data = mj.MjData(self.model)
        self.jnt_ids = np.array([self.model.joint(n).id for n in self.joint_names], dtype=int)
        self.qadr = self.model.jnt_qposadr[self.jnt_ids]
        self.dadr = self.model.jnt_dofadr[self.jnt_ids]
        self.n_dof = len(self.joint_indices)
        self.apply_config(config)

    def _build_xml_tree(self) -> ET.ElementTree:
        tree = _prune_mjcf_to_subset(self.xml_path, self.config.side, self.finger)
        root = tree.getroot()

        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", f"{self.config.dt:.6f}")
        opt.set("gravity", " ".join(f"{g:.6f}" for g in self.config.gravity))
        flag = opt.find("flag")
        if flag is None:
            flag = ET.SubElement(opt, "flag")
        flag.set("contact", "disable")

        for tag in ("tendon", "actuator"):
            old = root.find(tag)
            if old is not None:
                root.remove(old)

        tendon_el = ET.SubElement(root, "tendon") if self.has_diff else None
        act_el = ET.SubElement(root, "actuator")

        for local_i, jid in enumerate(self.joint_indices):
            jname = self.joint_names[local_i]
            if self.has_diff and jid in (19, 20):
                mode = jid - 19
                tname = f"thumb_diff_{mode}"
                t = ET.SubElement(tendon_el, "fixed", name=tname, armature="0.001", damping="0.01", frictionloss="0.01")
                for col, c_jid in enumerate((19, 20)):
                    c_name = mjcf_joint_names(self.config.side)[c_jid]
                    ET.SubElement(t, "joint", joint=c_name, coef=f"{self.T_diff[mode, col]:.12f}")
                ET.SubElement(
                    act_el, "general", name=f"act_{tname}", tendon=tname,
                    biastype="affine", gainprm="1 0 0", biasprm="0 0 0",
                    forcelimited="true", forcerange="-0.6 0.6",
                )
            else:
                ET.SubElement(
                    act_el, "general", name=f"act_J{jid:02d}", joint=jname,
                    biastype="affine", gainprm="1 0 0", biasprm="0 0 0",
                    forcelimited="true", forcerange="-0.6 0.6",
                )
        return tree

    def apply_config(self, config: Revo3ClosedLoopConfig | None = None) -> None:
        if config is not None:
            self.config = config
        cfg = self.config
        m = self.model
        m.opt.timestep = float(cfg.dt)
        m.opt.gravity[:] = cfg.gravity
        m.opt.disableflags |= mj.mjtDisableBit.mjDSBL_CONTACT

        for local_i, jid in enumerate(self.joint_indices):
            ch = cfg.channels[jid]
            d_idx = int(self.dadr[local_i])
            j_idx = int(self.jnt_ids[local_i])
            if self.has_diff and jid in (19, 20):
                mode = jid - 19
                # Joint-side armature/damping/friction stay 0; effective motor mechanics live on tendons
                m.dof_armature[d_idx] = 0.0
                m.dof_damping[d_idx] = 0.0
                m.dof_frictionloss[d_idx] = 0.0
                m.jnt_stiffness[j_idx] = float(cfg.thumb_joint_stiffness[mode])
                m.qpos_spring[int(self.qadr[local_i])] = float(cfg.thumb_joint_springref[mode])
                m.tendon_armature[mode] = float(ch.armature)
                m.tendon_damping[mode] = float(ch.damping)
                m.tendon_frictionloss[mode] = float(ch.frictionloss)
            else:
                m.dof_armature[d_idx] = float(ch.armature)
                m.dof_damping[d_idx] = float(ch.damping)
                m.dof_frictionloss[d_idx] = float(ch.frictionloss)
                m.jnt_stiffness[j_idx] = float(ch.stiffness)
                m.qpos_spring[int(self.qadr[local_i])] = float(ch.springref)
            lim = float(ch.current_limit.i_cont)
            m.actuator_forcerange[local_i, 0] = -lim
            m.actuator_forcerange[local_i, 1] = lim
        mj.mj_setConst(m, self.data)

    def export_xml(self, out_path: Path | str, *, standalone: bool = False) -> Path:
        """Save a portable closed-loop MuJoCo XML with identified parameters & tendons.

        Rewrites ``<compiler meshdir=... texturedir=.../>`` to clean relative paths
        (zero hardcoded machine paths). When ``standalone=True``, copies ``meshes/``
        and ``textures/`` next to ``out_path`` so the folder can be moved anywhere.
        """
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        mj.mj_saveLastXML(str(out), self.model)

        asset_root = self.xml_path.parent.resolve()
        if standalone:
            for sub in ("meshes", "textures"):
                src_dir = asset_root / sub
                dst_dir = out.parent / sub
                if src_dir.is_dir() and src_dir.resolve() != dst_dir.resolve():
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            rel_dir = "."
        else:
            rel_dir = Path(os.path.relpath(asset_root, out.parent)).as_posix()

        tree = ET.parse(out)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        # Preserve subpaths inside meshes/textures already referenced in <asset>
        if standalone:
            compiler.set("meshdir", ".")
            compiler.set("texturedir", ".")
        else:
            compiler.set("meshdir", rel_dir)
            compiler.set("texturedir", rel_dir)
        # Strip any absolute paths that MuJoCo may have emitted on <mesh>/<texture> file attrs
        for tag in ("mesh", "texture"):
            for elem in root.findall(f".//{tag}"):
                fval = elem.get("file")
                if fval and os.path.isabs(fval):
                    try:
                        rel_f = Path(fval).resolve().relative_to(asset_root).as_posix()
                        elem.set("file", rel_f)
                    except ValueError:
                        pass
        tree.write(out, encoding="utf-8", xml_declaration=False)
        return out

    def rollout(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        t_fb: np.ndarray,
        ct: np.ndarray,
        qt: np.ndarray,
        vt: np.ndarray,
        kp: float | np.ndarray,
        kd: float | np.ndarray,
        *,
        use_i2t: bool = True,
        return_currents: bool = False,
        backend: str = "auto",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Forward closed-loop rollout on exact command/feedback timestamps.

        Args:
            q0, v0: (n_dof,) initial state at t_fb[0] (rad, rad/s).
            t_fb: (M,) strictly increasing feedback timestamps (s).
            ct: (K,) command completion timestamps (s).
            qt, vt: (K, n_dof) commanded positions (rad) and velocities (rad/s).
            kp, kd: scalar or (n_dof,) SDK-native MIT gains.
            use_i2t: if True and any channel has i_peak > i_cont, runs leaky-integrator
                     thermal-budget current limiting.
            backend: "auto" (prefers native C kernel when compiled), "c", "mr" (MuJoCo C rollout),
                     or "python" (reference loop).
        """
        cfg = self.config
        m, d = self.model, self.data
        dt = float(m.opt.timestep)
        t_rel = np.asarray(t_fb, float) - float(t_fb[0])
        ct_rel = np.asarray(ct, float) - float(t_fb[0])
        steps = int(np.ceil(t_rel[-1] / dt)) + 1
        t_grid = np.arange(steps, dtype=float) * dt

        kp_vec = np.broadcast_to(np.asarray(kp, float), (self.n_dof,))
        kd_vec = np.broadcast_to(np.asarray(kd, float), (self.n_dof,))

        # Precompute delayed target arrays on the simulation grid
        q_des = np.empty((steps, self.n_dof), dtype=float)
        v_des = np.empty((steps, self.n_dof), dtype=float)
        kp_eff = np.empty(self.n_dof, dtype=float)
        kd_eff = np.empty(self.n_dof, dtype=float)
        vel_mult = np.empty(self.n_dof, dtype=float)
        bias_act = np.empty(self.n_dof, dtype=float)

        diff_pos = None
        if self.has_diff:
            p19 = int(np.where(self.joint_indices == 19)[0][0])
            p20 = int(np.where(self.joint_indices == 20)[0][0])
            diff_pos = (p19, p20)
            bias_j19_20 = np.asarray(cfg.thumb_joint_bias, dtype=float)

        any_i2t = False
        i_cont = np.empty(self.n_dof, dtype=float)
        i_peak = np.empty(self.n_dof, dtype=float)
        tau_s = np.empty(self.n_dof, dtype=float)

        for local_i, jid in enumerate(self.joint_indices):
            ch = cfg.channels[jid]
            ix = np.clip(np.searchsorted(ct_rel, t_grid - ch.delay_s, side="right") - 1, 0, len(ct_rel) - 1)
            kp_eff[local_i] = kp_vec[local_i] * ch.kp_scale
            kd_eff[local_i] = kd_vec[local_i] * ch.kd_scale
            vel_mult[local_i] = ch.vel_mult
            i_cont[local_i] = ch.current_limit.i_cont
            i_peak[local_i] = ch.current_limit.i_peak
            tau_s[local_i] = ch.current_limit.tau_s
            if use_i2t and ch.current_limit.is_dynamic:
                any_i2t = True

            if self.has_diff and jid in (19, 20):
                mode = jid - 19
                p19, p20 = diff_pos
                q_des[:, local_i] = qt[ix, p19] * self.T_diff[mode, 0] + qt[ix, p20] * self.T_diff[mode, 1]
                v_des[:, local_i] = vt[ix, p19] * self.T_diff[mode, 0] + vt[ix, p20] * self.T_diff[mode, 1]
                bias_act[local_i] = float(self.T_diff[mode] @ bias_j19_20)
            else:
                q_des[:, local_i] = qt[ix, local_i]
                v_des[:, local_i] = vt[ix, local_i]
                bias_act[local_i] = ch.bias

        idx_f = t_rel / dt
        idx_lo = np.minimum(idx_f.astype(int), steps - 1)
        idx_hi = np.minimum(idx_lo + 1, steps - 1)
        alpha = (idx_f - idx_lo)[:, None]

        return self._step_from_prepared(
            q0, v0, steps, q_des, v_des, kp_eff, kd_eff, vel_mult, bias_act,
            i_cont, i_peak, tau_s, idx_lo, idx_hi, alpha,
            use_i2t=any_i2t, return_currents=return_currents, backend=backend,
        )

    def _step_from_prepared(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        steps: int,
        q_des: np.ndarray,
        v_des: np.ndarray,
        kp_eff: np.ndarray,
        kd_eff: np.ndarray,
        vel_mult: np.ndarray,
        bias_act: np.ndarray,
        i_cont: np.ndarray,
        i_peak: np.ndarray,
        tau_s: np.ndarray,
        idx_lo: np.ndarray,
        idx_hi: np.ndarray,
        alpha: np.ndarray,
        *,
        use_i2t: bool,
        return_currents: bool,
        backend: str = "auto",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        m, d = self.model, self.data
        dt = float(m.opt.timestep)
        kd_v = kd_eff * vel_mult

        use_c = False
        if backend == "c":
            use_c = True
        elif backend == "auto":
            use_c = get_c_rollout_lib() is not None

        if use_c:
            m.actuator_biasprm[:] = 0.0
            max_lim = np.maximum(i_cont, i_peak) + 1.0
            m.actuator_forcerange[:, 0] = -max_lim
            m.actuator_forcerange[:, 1] = max_lim
            mj.mj_resetData(m, d)
            d.qpos[self.qadr] = q0
            d.qvel[self.dadr] = v0
            mj.mj_forward(m, d)
            p19 = int(np.where(self.joint_indices == 19)[0][0]) if self.has_diff else -1
            p20 = int(np.where(self.joint_indices == 20)[0][0]) if self.has_diff else -1
            q_traj, u_traj = run_c_rollout(
                m, d, steps, self.n_dof, self.has_diff, p19, p20,
                use_i2t, return_currents, self.T_diff, self.qadr, self.dadr,
                q_des, v_des, kp_eff, kd_v, kd_eff, bias_act, i_cont, i_peak, tau_s,
            )
            q_out = q_traj[idx_lo] * (1.0 - alpha) + q_traj[idx_hi] * alpha
            if return_currents:
                u_out = u_traj[idx_lo] * (1.0 - alpha) + u_traj[idx_hi] * alpha
                return q_out, u_out
            return q_out

        if (backend == "mr" or (backend == "auto" and not use_i2t and not return_currents)):
            m.actuator_biasprm[:, 0] = 0.0
            m.actuator_biasprm[:, 1] = -kp_eff
            m.actuator_biasprm[:, 2] = -kd_eff
            m.actuator_forcerange[:, 0] = -i_cont
            m.actuator_forcerange[:, 1] = i_cont
            ctrl = q_des * kp_eff[None, :] + v_des * kd_v[None, :] + bias_act[None, :]
            mj.mj_resetData(m, d)
            d.qpos[self.qadr] = q0
            d.qvel[self.dadr] = v0
            mj.mj_forward(m, d)
            init = np.empty(mj.mj_stateSize(m, mj.mjtState.mjSTATE_FULLPHYSICS))
            mj.mj_getState(m, d, init, mj.mjtState.mjSTATE_FULLPHYSICS)
            states, _ = mr.rollout(m, d, init[None, :], ctrl[None, :-1, :])
            q_traj = np.vstack([q0[None, :], states[0][:, 1 + self.qadr]])
            return q_traj[idx_lo] * (1.0 - alpha) + q_traj[idx_hi] * alpha

        # Reference pure-Python step loop
        m.actuator_biasprm[:] = 0.0
        max_lim = np.maximum(i_cont, i_peak) + 1.0
        m.actuator_forcerange[:, 0] = -max_lim
        m.actuator_forcerange[:, 1] = max_lim

        mj.mj_resetData(m, d)
        d.qpos[self.qadr] = q0
        d.qvel[self.dadr] = v0
        mj.mj_forward(m, d)

        q_traj = np.empty((steps, self.n_dof), dtype=float)
        u_traj = np.empty((steps, self.n_dof), dtype=float) if return_currents else None
        q_traj[0] = q0

        s_therm = np.zeros(self.n_dof, dtype=float)
        dyn_mask = (i_peak > i_cont + 1e-9) & (tau_s > 0.0) & use_i2t
        denom = np.where(dyn_mask, (i_peak**2 - i_cont**2) * np.maximum(tau_s, 1e-6), 1.0)
        i_cont_sq = i_cont**2
        diff_lim = i_peak - i_cont

        if self.has_diff:
            p19 = int(np.where(self.joint_indices == 19)[0][0])
            p20 = int(np.where(self.joint_indices == 20)[0][0])
            t00, t01 = self.T_diff[0, 0], self.T_diff[0, 1]
            t10, t11 = self.T_diff[1, 0], self.T_diff[1, 1]

        qadr = self.qadr
        dadr = self.dadr

        for k in range(steps - 1):
            qk = d.qpos[qadr]
            vk = d.qvel[dadr]
            if self.has_diff:
                q_act = qk.copy()
                v_act = vk.copy()
                q19, q20 = qk[p19], qk[p20]
                v19, v20 = vk[p19], vk[p20]
                q_act[p19] = t00 * q19 + t01 * q20
                q_act[p20] = t10 * q19 + t11 * q20
                v_act[p19] = t00 * v19 + t01 * v20
                v_act[p20] = t10 * v19 + t11 * v20
            else:
                q_act = qk
                v_act = vk

            u_des = kp_eff * (q_des[k] - q_act) + kd_v * v_des[k] - kd_eff * v_act + bias_act
            ceil_k = np.where(dyn_mask, i_peak - diff_lim * s_therm, i_cont)
            u_cmd = np.clip(u_des, -ceil_k, ceil_k)
            if return_currents:
                u_traj[k] = u_cmd

            if np.any(dyn_mask):
                ds = (u_cmd**2 - i_cont_sq) / denom * dt
                s_therm = np.clip(s_therm + np.where(dyn_mask, ds, 0.0), 0.0, 1.0)

            d.ctrl[:] = u_cmd
            mj.mj_step(m, d)
            q_traj[k + 1] = d.qpos[qadr]

        if return_currents:
            u_traj[-1] = u_traj[-2] if steps > 1 else 0.0
            q_out = q_traj[idx_lo] * (1.0 - alpha) + q_traj[idx_hi] * alpha
            u_out = u_traj[idx_lo] * (1.0 - alpha) + u_traj[idx_hi] * alpha
            return q_out, u_out
        return q_traj[idx_lo] * (1.0 - alpha) + q_traj[idx_hi] * alpha
