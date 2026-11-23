"""End-to-end 3-stage REVO3 Ultra21 closed-loop system identification & asset export pipeline.

Pipeline stages:
  * Stage 1 — Multi-gain unsaturated electrical law (kp_scale, kd_scale, vel_mult, delay, bias)
  * Stage 2 — Inverse-dynamics + quasi-static mechanical seed (armature, damping, friction, spring)
  * Stage 3 — Option A Fast Hybrid closed-loop optimizer:
      - Native C ``i^2*t`` + 45-deg differential rollout kernel (auto-compiled with gcc -O3)
      - Precomputed delay-indexed target grids
      - Coarse-to-fine timestep: parallel CMA-ES (dt = 2 ms) -> parallel-Jacobian TRF/LM (dt = 1 ms)
  * Stage 4 — Strict held-out validation + automatic MuJoCo & Isaac Lab asset export
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares, lsq_linear

from ._c_rollout import get_c_rollout_lib
from .closedloop import (
    ActuatorChannelParams,
    DEFAULT_GRAVITY,
    FINGER_SLICES,
    Revo3ClosedLoopConfig,
    Revo3ClosedLoopModel,
    T_DIFF_45,
    ThermalCurrentLimit,
)
from .cma import CMA
from .data import evaluate_fragments, extract_fragments, load_trial_unified
from .device import DOF, JOINT_NAMES
from .isaaclab_export import export_isaaclab_assets
from .model import mjcf_joint_names
from .paths import default_usd_path, default_xml_path


DELAY_GRID_S = np.arange(0.0, 0.025, 0.002)
STAGE1_DELAYS_S = np.arange(0.0, 0.031, 0.002)


def load_manifest(dataset_or_manifest: Path | str) -> tuple[dict, Path]:
    """Load manifest.json from a dataset directory or direct file path."""
    p = Path(dataset_or_manifest).expanduser().resolve()
    manifest_path = p / "manifest.json" if p.is_dir() else p
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    # Resolve relative capture paths cleanly
    for cap in manifest.get("captures", []):
        cp = Path(cap["path"])
        if not cp.is_absolute():
            cap["path"] = str((base_dir / cp).resolve())
    return manifest, manifest_path


def fit_stage1_electrical(manifest: dict, *, verbose: bool = True) -> tuple[dict, dict]:
    """Stage 1: Multi-gain unsaturated electrical law on training gains (fit split only)."""
    by_plan: dict[str, list] = {"planC": [], "planD": [], "planT": []}
    for cap in manifest["captures"]:
        if cap.get("gain_role", "train") == "train":
            by_plan[cap["plan"]].append((cap, load_trial_unified(cap["path"])))
    for k, items in by_plan.items():
        if not items:
            raise ValueError(f"No training captures found for plan {k!r} in manifest")

    elec: dict[int, dict] = {}
    if verbose:
        print("=== Stage 1: Multi-Gain Electrical Law (Train Gains, Fit Split Only) ===")
        print(f"{'JID':>3s} {'Name':20s} | {'kp_s':>6s} {'kd_s':>6s} {'vmult':>6s} {'bias':>7s} {'delay':>6s} | {'R2':>6s} {'RMSE_A':>6s}")

    for jid in range(DOF):
        plan_key = "planC" if jid < 16 else ("planD" if jid < 19 else "planT")
        trials = by_plan[plan_key]
        all_i = np.concatenate([tr.effort[tr.split == "fit", jid] for _, tr in trials])
        i_p99 = float(np.percentile(np.abs(all_i), 99))
        i_max = float(np.max(np.abs(all_i)))
        sat_thresh = 0.36 if jid >= 19 else 0.48

        best = None
        for d_s in STAGE1_DELAYS_S:
            Xs, ys = [], []
            for cap, tr in trials:
                m = tr.split == "fit"
                if jid >= 19:
                    unsat = (np.abs(tr.effort[:, 19]) < sat_thresh) & (np.abs(tr.effort[:, 20]) < sat_thresh)
                else:
                    unsat = np.abs(tr.effort[:, jid]) < sat_thresh
                m = m & unsat
                if int(m.sum()) < 40:
                    continue
                t_m = tr.t_fb[m]
                ix = np.searchsorted(tr.ct, t_m - d_s, side="right") - 1
                ok = ix >= 0
                ix = ix[ok]
                idx_m = np.flatnonzero(m)[ok]
                kp = float(cap["kp"])
                kd = float(cap["kd"])
                eq = kp * (tr.qt[ix, jid] - tr.q[idx_m, jid])
                ev_cmd = kd * tr.vt[ix, jid]
                ev_meas = -kd * tr.v[idx_m, jid]
                Xs.append(np.column_stack([eq, ev_cmd, ev_meas, np.ones(len(idx_m))]))
                ys.append(tr.effort[idx_m, jid])
            if not Xs:
                continue
            X = np.vstack(Xs)
            y = np.concatenate(ys)
            res = lsq_linear(X, y, bounds=([0.5, 0.1, 0.1, -0.2], [6.0, 4.0, 4.0, 0.2]))
            pred = X @ res.x
            mse = float(np.mean((pred - y) ** 2))
            if best is None or mse < best[0]:
                r2 = 1.0 - mse / max(float(np.var(y)), 1e-12)
                best = (mse, d_s, res.x, r2, len(y))

        mse, d_s, coef, r2, n_samp = best
        kp_s, kd_cmd, kd_meas, bias = [float(v) for v in coef]
        vmult = float(np.clip(kd_cmd / max(kd_meas, 1e-6), 0.2, 2.0))
        elec[jid] = dict(
            kp_scale=kp_s,
            kd_scale=kd_meas,
            vel_mult=vmult,
            bias=bias,
            delay_s=float(d_s),
            r2=float(r2),
            rmse_a=float(np.sqrt(mse)),
            n_samples=int(n_samp),
            i_p99=i_p99,
            i_max=i_max,
        )
        if verbose:
            print(
                f"J{jid:02d} {JOINT_NAMES[jid]:20s} | {kp_s:6.3f} {kd_meas:6.3f} {vmult:6.3f} "
                f"{bias:+7.4f} {d_s*1e3:4.0f}ms | {r2:6.4f} {np.sqrt(mse):6.4f}"
            )
    return elec, by_plan


def fit_stage2_mechanical(
    by_plan: dict,
    *,
    xml_path: Path | str | None = None,
    side: str = "right",
    verbose: bool = True,
) -> dict[int, dict]:
    """Stage 2: Inverse-dynamics + quasi-static mechanical seed on fit split only."""
    xml = default_xml_path(side) if xml_path is None else Path(xml_path).expanduser().resolve(strict=True)
    m = mj.MjModel.from_xml_path(str(xml))
    d = mj.MjData(m)
    m.opt.gravity[:] = DEFAULT_GRAVITY
    m.opt.disableflags |= mj.mjtDisableBit.mjDSBL_CONSTRAINT
    m.dof_armature[:] = 0.0
    m.dof_damping[:] = 0.0
    m.dof_frictionloss[:] = 0.0
    m.jnt_stiffness[:] = 0.0
    mj.mj_setConst(m, d)
    jnames = mjcf_joint_names(side)
    jids_mj = [m.joint(n).id for n in jnames]
    qa = m.jnt_qposadr[jids_mj]
    va = m.jnt_dofadr[jids_mj]

    mech: dict[int, dict] = {}
    if verbose:
        print("\n=== Stage 2: Inverse Dynamics Mechanical Seed (Fit Split Only) ===")
        print(f"{'JID':>3s} | {'armature':>9s} {'damping':>8s} {'friction':>8s} {'stiff':>8s} {'spr_ref':>8s} | {'ID_RMSE_A':>9s}")

    for jid in range(DOF):
        plan_key = "planC" if jid < 16 else ("planD" if jid < 19 else "planT")
        trials = by_plan[plan_key]
        rows_X, rows_y = [], []
        for _, tr in trials:
            idx = np.flatnonzero(tr.split == "fit")
            t = tr.t_fb[idx]
            q = tr.q[idx]
            v = tr.v[idx]
            eff = tr.effort[idx]
            dt_arr = np.diff(t)
            for k in range(2, len(t) - 2):
                if np.max(dt_arr[k - 2 : k + 2]) > 0.06:
                    continue
                if abs(v[k, jid]) < 0.04:
                    continue
                delta = t[k - 2 : k + 3] - t[k]
                c = delta - delta.mean()
                acc = float(c @ v[k - 2 : k + 3, jid] / max(float(c @ c), 1e-12))
                d.qpos[qa] = q[k]
                d.qvel[va] = v[k]
                d.qacc[va] = 0.0
                d.qacc[va[jid]] = acc
                mj.mj_inverse(m, d)
                tau_res = float(eff[k, jid] - d.qfrc_inverse[va[jid]])
                rows_X.append([acc, v[k, jid], np.sign(v[k, jid]), q[k, jid], 1.0])
                rows_y.append(tau_res)
        X = np.asarray(rows_X, dtype=float)
        y = np.asarray(rows_y, dtype=float)
        res = lsq_linear(X, y, bounds=([1e-4, 1e-3, 1e-3, 0.0, -0.5], [0.015, 0.15, 0.15, 0.5, 0.5]))
        arm, damp, fric, stiff, c0 = [float(x) for x in res.x]
        springref = float(np.clip(-c0 / stiff if stiff > 1e-4 else 0.0, -1.5, 2.5))
        err = float(np.sqrt(np.mean((X @ res.x - y) ** 2)))
        mech[jid] = dict(armature=arm, damping=damp, frictionloss=fric, stiffness=stiff, springref=springref, id_rmse_a=err)
        if verbose:
            print(f"J{jid:02d} | {arm:9.5f} {damp:8.4f} {fric:8.4f} {stiff:8.4f} {springref:+8.3f} | {err:9.4f}")
    return mech


def build_seed_config(elec: dict, mech: dict, *, side: str = "right") -> Revo3ClosedLoopConfig:
    """Assemble Stage 1 + Stage 2 initial 21-DOF closed-loop configuration."""
    channels = []
    for jid in range(DOF):
        if jid in (19, 20):
            arm = 0.5 * (mech[19]["armature"] + mech[20]["armature"])
            damp = 0.5 * (mech[19]["damping"] + mech[20]["damping"])
            fric = 0.5 * (mech[19]["frictionloss"] + mech[20]["frictionloss"])
            kp_s = 0.5 * (elec[19]["kp_scale"] + elec[20]["kp_scale"])
            kd_s = 0.5 * (elec[19]["kd_scale"] + elec[20]["kd_scale"])
            vmult = 0.5 * (elec[19]["vel_mult"] + elec[20]["vel_mult"])
            d_s = 0.002 * round(0.5 * (elec[19]["delay_s"] + elec[20]["delay_s"]) / 0.002)
            # Orthonormal tendon-space continuous/peak limits (0.56 A / 0.82 A)
            cl = ThermalCurrentLimit(i_cont=0.56, i_peak=0.82, tau_s=0.04)
            channels.append(ActuatorChannelParams(
                armature=arm, damping=damp, frictionloss=fric,
                kp_scale=kp_s, kd_scale=kd_s, vel_mult=vmult, delay_s=d_s,
                bias=0.0, stiffness=0.0, springref=0.0, current_limit=cl,
            ))
        else:
            e = elec[jid]
            mc = mech[jid]
            i_pk = 0.60 if jid in (0, 4, 8, 12) else max(0.75, min(0.95, e["i_max"] * 1.05))
            cl = ThermalCurrentLimit(i_cont=0.60, i_peak=i_pk, tau_s=0.03)
            channels.append(ActuatorChannelParams(
                armature=mc["armature"], damping=mc["damping"], frictionloss=mc["frictionloss"],
                stiffness=mc["stiffness"], springref=mc["springref"],
                kp_scale=e["kp_scale"], kd_scale=e["kd_scale"], vel_mult=e["vel_mult"],
                delay_s=e["delay_s"], bias=e["bias"], current_limit=cl,
            ))
    return Revo3ClosedLoopConfig(
        side=side,
        dt=0.001,
        gravity=DEFAULT_GRAVITY.tolist(),
        thumb_differential=True,
        thumb_T=T_DIFF_45.tolist(),
        channels=channels,
        thumb_joint_stiffness=[mech[19]["stiffness"], mech[20]["stiffness"]],
        thumb_joint_springref=[mech[19]["springref"], mech[20]["springref"]],
        thumb_joint_bias=[elec[19]["bias"], elec[20]["bias"]],
        metadata=dict(stage="stage1_2_seed"),
    )


# ---------------------------------------------------------------------------
# Stage 3: Option A Fast Hybrid Optimizer (Native C + 2ms CMA-ES + 1ms Par-LM)
# ---------------------------------------------------------------------------
_M: Revo3ClosedLoopModel | None = None
_BASE_CFG: Revo3ClosedLoopConfig | None = None
_PREPPED: list[dict] = []
_LOC: list[int] = []
_JIDS: list[int] = []
_MODE = ""
_FIXED_DELAYS: list[float] | None = None
_STRIDE = 2
_BACKEND = "auto"


def _build_training_frags(manifest: dict, plan_tag: str, finger: str):
    frags, weights = [], []
    for cap in manifest["captures"]:
        if cap["plan"] != plan_tag or cap.get("gain_role", "train") != "train":
            continue
        tr = load_trial_unified(cap["path"])
        fs = extract_fragments(tr, finger=finger, splits=("fit",), group_by_stage=False)
        if not fs:
            continue
        f = fs[0]
        stages = tr.stage[tr.split == "fit"]
        w = np.ones(len(f.t_fb), dtype=float)
        if len(stages) == len(w):
            for i, st in enumerate(stages):
                if "micro" in st:
                    w[i] = 4.0
                elif "chirp" in st or "uncorr" in st:
                    w[i] = 1.5
                elif "rest" in st:
                    w[i] = 0.5
        frags.append(f)
        weights.append(w)
    if not frags:
        raise ValueError(f"No training fragments built for plan={plan_tag!r}, finger={finger!r}")
    return frags, weights


def _precompute_frags(frags, weights, dt: float, has_diff: bool, T_diff: np.ndarray, p19: int, p20: int):
    prepped = []
    for f, w in zip(frags, weights):
        t_rel = f.t_fb - f.t_fb[0]
        ct_rel = f.ct - f.t_fb[0]
        steps = int(np.ceil(t_rel[-1] / dt)) + 1
        t_grid = np.arange(steps, dtype=float) * dt
        idx_f = t_rel / dt
        idx_lo = np.minimum(idx_f.astype(int), steps - 1)
        idx_hi = np.minimum(idx_lo + 1, steps - 1)
        alpha = (idx_f - idx_lo)[:, None]

        qt_by_d, vt_by_d = {}, {}
        for d_idx, d_s in enumerate(DELAY_GRID_S):
            ix = np.clip(np.searchsorted(ct_rel, t_grid - d_s, side="right") - 1, 0, len(ct_rel) - 1)
            qg = f.qt[ix].copy()
            vg = f.vt[ix].copy()
            if has_diff:
                q19, q20 = qg[:, p19].copy(), qg[:, p20].copy()
                v19, v20 = vg[:, p19].copy(), vg[:, p20].copy()
                qg[:, p19] = T_diff[0, 0] * q19 + T_diff[0, 1] * q20
                qg[:, p20] = T_diff[1, 0] * q19 + T_diff[1, 1] * q20
                vg[:, p19] = T_diff[0, 0] * v19 + T_diff[0, 1] * v20
                vg[:, p20] = T_diff[1, 0] * v19 + T_diff[1, 1] * v20
            qt_by_d[d_idx] = qg
            vt_by_d[d_idx] = vg

        prepped.append(dict(
            q0=f.q[0].copy(), v0=f.v[0].copy(), q_meas=f.q.copy(),
            kp=f.kp.copy(), kd=f.kd.copy(), w=w.copy(), sqrt_w=np.sqrt(w),
            steps=steps, idx_lo=idx_lo, idx_hi=idx_hi, alpha=alpha,
            qt_by_d=qt_by_d, vt_by_d=vt_by_d,
        ))
    return prepped


def _fast_rollout_prepped(sim: Revo3ClosedLoopModel, cfg: Revo3ClosedLoopConfig, pf: dict) -> np.ndarray:
    n_dof = sim.n_dof
    steps = pf["steps"]
    q_des = np.empty((steps, n_dof), dtype=float)
    v_des = np.empty((steps, n_dof), dtype=float)
    kp_eff = np.empty(n_dof, dtype=float)
    kd_eff = np.empty(n_dof, dtype=float)
    vel_mult = np.empty(n_dof, dtype=float)
    bias_act = np.empty(n_dof, dtype=float)
    i_cont = np.empty(n_dof, dtype=float)
    i_peak = np.empty(n_dof, dtype=float)
    tau_s = np.empty(n_dof, dtype=float)
    bias_j19_20 = np.asarray(cfg.thumb_joint_bias, dtype=float)
    any_i2t = False

    for loc, jid in enumerate(sim.joint_indices):
        ch = cfg.channels[jid]
        d_idx = int(np.clip(round(ch.delay_s / 0.002), 0, len(DELAY_GRID_S) - 1))
        q_des[:, loc] = pf["qt_by_d"][d_idx][:, loc]
        v_des[:, loc] = pf["vt_by_d"][d_idx][:, loc]
        kp_eff[loc] = pf["kp"][loc] * ch.kp_scale
        kd_eff[loc] = pf["kd"][loc] * ch.kd_scale
        vel_mult[loc] = ch.vel_mult
        i_cont[loc] = ch.current_limit.i_cont
        i_peak[loc] = ch.current_limit.i_peak
        tau_s[loc] = ch.current_limit.tau_s
        if ch.current_limit.is_dynamic:
            any_i2t = True
        if sim.has_diff and jid in (19, 20):
            bias_act[loc] = float(sim.T_diff[jid - 19] @ bias_j19_20)
        else:
            bias_act[loc] = ch.bias

    return sim._step_from_prepared(
        pf["q0"], pf["v0"], steps, q_des, v_des, kp_eff, kd_eff, vel_mult, bias_act,
        i_cont, i_peak, tau_s, pf["idx_lo"], pf["idx_hi"], pf["alpha"],
        use_i2t=any_i2t, return_currents=False, backend=_BACKEND,
    )


def _decode_std(x: np.ndarray, base_cfg: Revo3ClosedLoopConfig, jids: list[int], fixed_delays: list[float] | None):
    cfg = copy.deepcopy(base_cfg)
    stride_p = 7 if fixed_delays is None else 6
    scales = np.array([0.55, 0.55, 0.65, 0.65, 0.25, 0.015, 0.004])
    for k, jid in enumerate(jids):
        xj = x[k * stride_p : (k + 1) * stride_p]
        ch = cfg.channels[jid]
        ch.armature = float(np.clip(ch.armature * np.exp(np.clip(xj[0] * scales[0], -5, 5)), 1e-4, 0.015))
        ch.damping = float(np.clip(ch.damping * np.exp(np.clip(xj[1] * scales[1], -5, 5)), 5e-4, 0.18))
        ch.frictionloss = float(np.clip(ch.frictionloss * np.exp(np.clip(xj[2] * scales[2], -5, 5)), 5e-4, 0.15))
        base_k = max(ch.stiffness, 0.004)
        ch.stiffness = float(np.clip(base_k * np.exp(np.clip(xj[3] * scales[3], -5, 5)), 0.0, 0.35))
        ch.springref = float(np.clip(ch.springref + xj[4] * scales[4], -1.8, 2.5))
        ch.bias = float(np.clip(ch.bias + xj[5] * scales[5], -0.08, 0.08))
        if fixed_delays is None:
            d_raw = float(np.clip(ch.delay_s + xj[6] * scales[6], 0.0, 0.024))
            ch.delay_s = float(0.002 * round(d_raw / 0.002))
        else:
            ch.delay_s = float(fixed_delays[k])
    return cfg


def _decode_diff(x: np.ndarray, base_cfg: Revo3ClosedLoopConfig, fixed_delays: list[float] | None):
    """Symmetric-motor 45-deg differential decoder dynamically anchored to Stage 1 electrical scales."""
    cfg = copy.deepcopy(base_cfg)
    c19 = base_cfg.channels[19]
    c20 = base_cfg.channels[20]
    kp0 = max(0.5, 0.5 * (c19.kp_scale + c20.kp_scale))
    kd0 = max(0.1, 0.5 * (c19.kd_scale + c20.kd_scale))
    vm0 = float(np.clip(0.5 * (c19.vel_mult + c20.vel_mult), 0.5, 1.5))
    arm0 = float(np.clip(0.5 * (c19.armature + c20.armature), 3e-4, 8e-3))
    damp0 = float(np.clip(0.5 * (c19.damping + c20.damping), 2e-3, 8e-2))
    f0 = float(np.clip(0.5 * (c19.frictionloss + c20.frictionloss), 5e-3, 8e-2))
    d0 = float(np.clip(0.5 * (c19.delay_s + c20.delay_s), 0.004, 0.022))

    scales = np.array([0.55, 0.55, 0.65, 0.65, 0.04, 0.10, 0.05, 0.60, 0.25, 0.60, 0.25, 0.015, 0.015, 0.004])
    arm = float(np.clip(arm0 * np.exp(np.clip(x[0] * scales[0], -5, 5)), 2e-4, 0.010))
    damp = float(np.clip(damp0 * np.exp(np.clip(x[1] * scales[1], -5, 5)), 5e-4, 0.12))
    fp = float(np.clip(f0 * np.exp(np.clip(x[2] * scales[2], -5, 5)), 1e-3, 0.12))
    fm = float(np.clip(f0 * np.exp(np.clip(x[3] * scales[3], -5, 5)), 1e-3, 0.12))
    kp_s = float(np.clip(kp0 * np.exp(np.clip(x[4] * scales[4], -1, 1)), kp0 * 0.90, kp0 * 1.10))
    kd_s = float(np.clip(kd0 * np.exp(np.clip(x[5] * scales[5], -1, 1)), kd0 * 0.75, kd0 * 1.25))
    vmult = float(np.clip(vm0 * np.exp(np.clip(x[6] * scales[6], -1, 1)), vm0 * 0.88, vm0 * 1.12))
    if fixed_delays is None:
        d_s = float(0.002 * round(float(np.clip(d0 + x[13] * scales[13], 0.004, 0.022)) / 0.002))
    else:
        d_s = float(fixed_delays[0])
    cl = ThermalCurrentLimit(i_cont=0.56, i_peak=0.82, tau_s=0.04)

    for m_idx, fric in ((0, fp), (1, fm)):
        cfg.channels[19 + m_idx] = ActuatorChannelParams(
            armature=arm, damping=damp, frictionloss=fric,
            kp_scale=kp_s, kd_scale=kd_s, vel_mult=vmult, delay_s=d_s,
            bias=0.0, stiffness=0.0, springref=0.0, current_limit=cl,
        )
    k19_0 = max(base_cfg.thumb_joint_stiffness[0], 0.01)
    k20_0 = max(base_cfg.thumb_joint_stiffness[1], 0.01)
    r19_0 = base_cfg.thumb_joint_springref[0]
    r20_0 = base_cfg.thumb_joint_springref[1]
    b19_0 = base_cfg.thumb_joint_bias[0]
    b20_0 = base_cfg.thumb_joint_bias[1]
    cfg.thumb_joint_stiffness = [
        float(np.clip(k19_0 * np.exp(np.clip(x[7] * scales[7], -5, 5)), 0.0, 0.25)),
        float(np.clip(k20_0 * np.exp(np.clip(x[9] * scales[9], -5, 5)), 0.0, 0.25)),
    ]
    cfg.thumb_joint_springref = [
        float(np.clip(r19_0 + x[8] * scales[8], -1.0, 2.0)),
        float(np.clip(r20_0 + x[10] * scales[10], -1.0, 2.0)),
    ]
    cfg.thumb_joint_bias = [
        float(np.clip(b19_0 + x[11] * scales[11], -0.06, 0.06)),
        float(np.clip(b20_0 + x[12] * scales[12], -0.06, 0.06)),
    ]
    return cfg


def _eval_mse(x: np.ndarray) -> float:
    cfg = _decode_diff(x, _BASE_CFG, _FIXED_DELAYS) if _MODE == "thumb_diff" else _decode_std(x, _BASE_CFG, _JIDS, _FIXED_DELAYS)
    _M.apply_config(cfg)
    num = den = 0.0
    for pf in _PREPPED:
        pred = _fast_rollout_prepped(_M, cfg, pf)
        err = pred[:, _LOC] - pf["q_meas"][:, _LOC]
        num += float(np.sum(pf["w"][:, None] * (err ** 2)))
        den += float(np.sum(pf["w"]) * len(_LOC))
    return num / max(den, 1e-12)


def _eval_residuals(x: np.ndarray) -> np.ndarray:
    cfg = _decode_diff(x, _BASE_CFG, _FIXED_DELAYS) if _MODE == "thumb_diff" else _decode_std(x, _BASE_CFG, _JIDS, _FIXED_DELAYS)
    _M.apply_config(cfg)
    parts = []
    sl = slice(1, None, _STRIDE)
    for pf in _PREPPED:
        pred = _fast_rollout_prepped(_M, cfg, pf)
        err = (pred[sl, :][:, _LOC] - pf["q_meas"][sl, :][:, _LOC]) * pf["sqrt_w"][sl, None]
        parts.append(err.ravel())
    return np.concatenate(parts)


def optimize_block_fast(
    manifest: dict,
    base_cfg: Revo3ClosedLoopConfig,
    block_name: str,
    plan_tag: str,
    finger: str,
    jids: list[int],
    *,
    xml_path: Path,
    cma_gens: int = 20,
    lm_max_nfev: int = 12,
    workers: int | None = None,
    rng_seed: int = 2026,
):
    global _M, _BASE_CFG, _PREPPED, _LOC, _JIDS, _MODE, _FIXED_DELAYS, _STRIDE, _BACKEND
    t0 = time.perf_counter()
    n_cpu = workers or min(os.cpu_count() or 8, 24)
    _BACKEND = "c" if get_c_rollout_lib() is not None else "auto"
    _MODE = "thumb_diff" if block_name == "thumb_diff" else "standard"
    _JIDS = list(jids)
    finger_jids = list(FINGER_SLICES[finger])
    _LOC = [finger_jids.index(j) for j in jids]
    frags, weights = _build_training_frags(manifest, plan_tag, finger)

    # Phase A: dt = 2 ms, parallel CMA-ES (global escape + discrete delay search)
    cfg_2ms = copy.deepcopy(base_cfg)
    cfg_2ms.dt = 0.002
    _BASE_CFG = cfg_2ms
    _FIXED_DELAYS = None
    _M = Revo3ClosedLoopModel(xml_path, cfg_2ms, finger=finger)
    p19 = finger_jids.index(19) if _M.has_diff else -1
    p20 = finger_jids.index(20) if _M.has_diff else -1
    _PREPPED = _precompute_frags(frags, weights, 0.002, _M.has_diff, _M.T_diff, p19, p20)

    dim_a = 14 if _MODE == "thumb_diff" else 7 * len(jids)
    pop = max(8, min(n_cpu, 24))
    x0 = np.zeros(dim_a, dtype=float)
    best_loss, best_x = _eval_mse(x0), x0.copy()
    seed_rmse = float(np.rad2deg(np.sqrt(best_loss)))
    cma = CMA(dim_a, pop, rng_seed)

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
    if hasattr(os, "fork") and pop > 1:
        with ProcessPoolExecutor(max_workers=pop, mp_context=ctx) as pool:
            for _ in range(cma_gens):
                xs = cma.ask()
                losses = np.array(list(pool.map(_eval_mse, xs)), dtype=float)
                cma.tell(xs, losses)
                j_b = int(np.argmin(losses))
                if losses[j_b] < best_loss:
                    best_loss, best_x = float(losses[j_b]), xs[j_b].copy()
    else:
        for _ in range(cma_gens):
            xs = cma.ask()
            losses = np.array([_eval_mse(xv) for xv in xs], dtype=float)
            cma.tell(xs, losses)
            j_b = int(np.argmin(losses))
            if losses[j_b] < best_loss:
                best_loss, best_x = float(losses[j_b]), xs[j_b].copy()

    t_a = time.perf_counter() - t0
    rmse_a = float(np.rad2deg(np.sqrt(best_loss)))

    cfg_after_a = _decode_diff(best_x, cfg_2ms, None) if _MODE == "thumb_diff" else _decode_std(best_x, cfg_2ms, jids, None)
    if _MODE == "thumb_diff":
        fixed_d = [cfg_after_a.channels[19].delay_s]
        x_b0 = best_x[:13].copy()
    else:
        fixed_d = [cfg_after_a.channels[j].delay_s for j in jids]
        x_b0 = np.concatenate([best_x[k * 7 : k * 7 + 6] for k in range(len(jids))])

    # Phase B: dt = 1 ms, parallel-Jacobian TRF/LM polish
    t_b0 = time.perf_counter()
    cfg_1ms = copy.deepcopy(base_cfg)
    cfg_1ms.dt = 0.001
    _BASE_CFG = cfg_1ms
    _FIXED_DELAYS = fixed_d
    _STRIDE = 2
    _M = Revo3ClosedLoopModel(xml_path, cfg_1ms, finger=finger)
    _PREPPED = _precompute_frags(frags, weights, 0.001, _M.has_diff, _M.T_diff, p19, p20)

    dim_b = len(x_b0)
    x_b0 = np.clip(x_b0, -5.8, 5.8)
    eps = 0.015

    cache_r: dict[bytes, np.ndarray] = {}

    def fun_r(xv: np.ndarray) -> np.ndarray:
        key = xv.tobytes()
        if key not in cache_r:
            cache_r[key] = _eval_residuals(xv)
        return cache_r[key]

    if hasattr(os, "fork") and n_cpu > 1:
        with ProcessPoolExecutor(max_workers=min(n_cpu, dim_b), mp_context=ctx) as pool:
            def jac_r(xv: np.ndarray) -> np.ndarray:
                r0 = fun_r(xv)
                pert = []
                for i in range(dim_b):
                    xp = xv.copy()
                    xp[i] += eps
                    pert.append(xp)
                rs = list(pool.map(_eval_residuals, pert))
                J = np.empty((len(r0), dim_b), dtype=float)
                for i, ri in enumerate(rs):
                    J[:, i] = (ri - r0) / eps
                return J

            res = least_squares(
                fun_r, x_b0, jac=jac_r, method="trf",
                bounds=(-6.0, 6.0), xtol=5e-4, ftol=3e-4, gtol=1e-4,
                max_nfev=lm_max_nfev,
            )
    else:
        res = least_squares(
            fun_r, x_b0, method="trf",
            bounds=(-6.0, 6.0), xtol=5e-4, ftol=3e-4, gtol=1e-4,
            max_nfev=lm_max_nfev,
        )

    final_x = res.x
    final_mse = _eval_mse(final_x)
    rmse_b = float(np.rad2deg(np.sqrt(final_mse)))
    t_b = time.perf_counter() - t_b0
    t_tot = time.perf_counter() - t0
    best_cfg = _decode_diff(final_x, cfg_1ms, fixed_d) if _MODE == "thumb_diff" else _decode_std(final_x, cfg_1ms, jids, fixed_d)

    print(
        f"  [{block_name:10s}] seed={seed_rmse:.3f} deg -> CMA(2ms, {t_a:4.1f}s)={rmse_a:.3f} deg "
        f"-> LM(1ms, {t_b:4.1f}s)={rmse_b:.3f} deg | total={t_tot:4.1f}s",
        flush=True,
    )
    return best_cfg, dict(
        block=block_name, joints=jids, seed_rmse=seed_rmse, phaseA_rmse=rmse_a, phaseB_rmse=rmse_b,
        phaseA_s=round(t_a, 2), phaseB_s=round(t_b, 2), total_s=round(t_tot, 2),
        lm_nfev=int(res.nfev), lm_njev=int(res.njev),
    )


def evaluate_dataset(
    config: Revo3ClosedLoopConfig,
    dataset_or_manifest: Path | str,
    *,
    xml_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> dict:
    """Evaluate an identified config across training-gain validation & held-out gain captures."""
    manifest, _ = load_manifest(dataset_or_manifest)
    xml = default_xml_path(config.side) if xml_path is None else Path(xml_path).expanduser().resolve(strict=True)
    sim = Revo3ClosedLoopModel(xml, config, finger="full")

    per_joint_val: dict[int, float] = {}
    per_joint_ho: dict[int, float] = {}
    by_gain_rows: list[dict] = []

    plan_jids = {"planC": list(range(16)), "planD": [16, 17, 18], "planT": [19, 20]}

    for ptag, jids in plan_jids.items():
        val_frags, ho_frags = [], []
        for cap in manifest["captures"]:
            if cap["plan"] != ptag:
                continue
            tr = load_trial_unified(cap["path"])
            is_ho = cap.get("gain_role") == "heldout_gain"
            splits = ("fit", "validation") if is_ho else ("validation",)
            frags = extract_fragments(tr, finger="full", splits=splits, group_by_stage=False)
            if frags:
                r_cap = evaluate_fragments(sim, frags, use_i2t=True)
                mean_cap = float(np.mean([r_cap["per_joint_rmse_deg"][f"J{j:02d}"] for j in jids]))
                by_gain_rows.append(dict(
                    id=cap["id"], plan=ptag, kp=cap["kp"], kd=cap["kd"],
                    gain_role=cap.get("gain_role", "train"), rmse_deg=round(mean_cap, 4),
                ))
            if is_ho:
                ho_frags.extend(frags)
            else:
                val_frags.extend(frags)

        if val_frags:
            r_val = evaluate_fragments(sim, val_frags, use_i2t=True)
            for j in jids:
                per_joint_val[j] = float(r_val["per_joint_rmse_deg"][f"J{j:02d}"])
        if ho_frags:
            r_ho = evaluate_fragments(sim, ho_frags, use_i2t=True)
            for j in jids:
                per_joint_ho[j] = float(r_ho["per_joint_rmse_deg"][f"J{j:02d}"])

    summary = dict(
        val_all_rmse_deg=round(float(np.mean(list(per_joint_val.values()))), 4) if per_joint_val else None,
        heldout_all_rmse_deg=round(float(np.mean(list(per_joint_ho.values()))), 4) if per_joint_ho else None,
        per_joint_val_rmse_deg={f"J{j:02d}": round(v, 4) for j, v in sorted(per_joint_val.items())},
        per_joint_heldout_rmse_deg={f"J{j:02d}": round(v, 4) for j, v in sorted(per_joint_ho.items())},
        by_gain=by_gain_rows,
    )

    if out_dir is not None:
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (out / "per_joint_scores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["joint_id", "joint_name", "val_rmse_deg", "heldout_rmse_deg"])
            for j in range(DOF):
                w.writerow([
                    f"J{j:02d}", JOINT_NAMES[j],
                    f"{per_joint_val.get(j, float('nan')):.4f}",
                    f"{per_joint_ho.get(j, float('nan')):.4f}",
                ])
        with (out / "by_gain_scores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "plan", "kp", "kd", "gain_role", "rmse_deg"])
            w.writeheader()
            w.writerows(by_gain_rows)

        # Optional summary bar chart if matplotlib is installed
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
            xs = np.arange(DOF)
            width = 0.38
            if per_joint_val:
                ax.bar(xs - width / 2, [per_joint_val.get(j, 0.0) for j in xs], width, label="Train Gains (Val Split)")
            if per_joint_ho:
                ax.bar(xs + width / 2, [per_joint_ho.get(j, 0.0) for j in xs], width, label="Held-Out Gain")
            ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, label="1.0 deg target")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"J{j:02d}" for j in xs], fontsize=8)
            ax.set_ylabel("Closed-Loop RMSE (deg)")
            ax.set_title("REVO3 Ultra21 21-DOF Closed-Loop Identification Validation")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "validation_summary.png")
            plt.close(fig)
        except Exception:
            pass
    return summary


def export_all_assets(
    config: Revo3ClosedLoopConfig,
    out_dir: Path | str,
    *,
    xml_path: Path | str | None = None,
    usd_path: Path | str | None = None,
    kp_cmd: float = 2.0,
    kd_cmd: float = 0.2,
    sim_dt: float = 0.002,
    standalone: bool = False,
) -> dict[str, Path]:
    """Export BOTH MuJoCo (JSON + portable XML) AND Isaac Lab (USDA + Cfg + JSON) assets."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    xml = default_xml_path(config.side) if xml_path is None else Path(xml_path).expanduser().resolve(strict=True)
    usd = default_usd_path(config.side) if usd_path is None else Path(usd_path).expanduser().resolve(strict=True)

    json_name = f"revo3_{config.side}_identified.json"
    xml_name = f"revo3_{config.side}_identified.xml"
    json_path = out / json_name
    config.save(json_path)

    sim_full = Revo3ClosedLoopModel(xml, config, finger="full")
    mj_xml_path = sim_full.export_xml(out / xml_name, standalone=standalone)

    isaac_files = export_isaaclab_assets(
        config,
        out,
        base_usd=usd,
        closedloop_json_name=json_name,
        kp_cmd=kp_cmd,
        kd_cmd=kd_cmd,
        sim_dt=sim_dt,
        standalone=standalone,
    )
    return dict(
        json=json_path,
        mujoco_xml=mj_xml_path,
        isaaclab_usda=isaac_files["usda"],
        isaaclab_params_json=isaac_files["params_json"],
        isaaclab_cfg_py=isaac_files["cfg_py"],
    )


def run_full_pipeline(
    dataset_or_manifest: Path | str,
    out_dir: Path | str,
    *,
    xml_path: Path | str | None = None,
    usd_path: Path | str | None = None,
    cma_gens: int = 20,
    lm_max_nfev: int = 12,
    workers: int | None = None,
    kp_cmd: float = 2.0,
    kd_cmd: float = 0.2,
    sim_dt: float = 0.002,
    standalone: bool = False,
) -> dict:
    """Run Stage 1 -> Stage 2 -> Stage 3 Fast Hybrid -> Evaluation -> MuJoCo & Isaac Lab export."""
    for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

    t_all = time.perf_counter()
    manifest, manifest_path = load_manifest(dataset_or_manifest)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    xml = default_xml_path("right") if xml_path is None else Path(xml_path).expanduser().resolve(strict=True)

    # Pre-warm native C kernel before forking workers
    c_lib = get_c_rollout_lib()
    print(f"Native C rollout kernel: {'ENABLED (gcc -O3)' if c_lib is not None else 'UNAVAILABLE (Python/mr fallback)'}")

    elec, by_plan = fit_stage1_electrical(manifest)
    mech = fit_stage2_mechanical(by_plan, xml_path=xml)
    seed_cfg = build_seed_config(elec, mech)
    seed_cfg.save(out / "stage1_2_seed_config.json")

    print("\n=== Stage 3: Fast Hybrid Closed-Loop Optimization (CMA-ES 2ms -> Parallel LM 1ms) ===", flush=True)
    final_cfg = copy.deepcopy(seed_cfg)
    blocks = [
        ("little",     "planC", "little", [0, 1, 2, 3],     cma_gens, lm_max_nfev, 501),
        ("ring",       "planC", "ring",   [4, 5, 6, 7],     cma_gens, lm_max_nfev, 502),
        ("middle",     "planC", "middle", [8, 9, 10, 11],   cma_gens, lm_max_nfev, 503),
        ("index",      "planC", "index",  [12, 13, 14, 15], cma_gens, lm_max_nfev, 504),
        ("thumb_prox", "planD", "thumb",  [16, 17, 18],     cma_gens, lm_max_nfev, 505),
        ("thumb_diff", "planT", "thumb",  [19, 20],         max(cma_gens, 25), max(lm_max_nfev, 14), 506),
    ]
    logs = []
    for bname, ptag, finger, jids, cg, lm_n, sd in blocks:
        b_cfg, info = optimize_block_fast(
            manifest, final_cfg, bname, ptag, finger, jids,
            xml_path=xml, cma_gens=cg, lm_max_nfev=lm_n, workers=workers, rng_seed=sd,
        )
        logs.append(info)
        for j in jids:
            final_cfg.channels[j] = copy.deepcopy(b_cfg.channels[j])
        if bname == "thumb_diff":
            final_cfg.thumb_joint_stiffness = list(b_cfg.thumb_joint_stiffness)
            final_cfg.thumb_joint_springref = list(b_cfg.thumb_joint_springref)
            final_cfg.thumb_joint_bias = list(b_cfg.thumb_joint_bias)

    elapsed = time.perf_counter() - t_all
    final_cfg.metadata = dict(
        dataset=manifest_path.parent.name,
        stage="fast_hybrid_3stage",
        total_elapsed_s=round(elapsed, 2),
        blocks=logs,
    )

    print("\n=== Stage 4: Evaluation & Portable MuJoCo + Isaac Lab Asset Export ===", flush=True)
    eval_summary = evaluate_dataset(final_cfg, manifest_path, xml_path=xml, out_dir=out)
    exported = export_all_assets(
        final_cfg, out, xml_path=xml, usd_path=usd_path,
        kp_cmd=kp_cmd, kd_cmd=kd_cmd, sim_dt=sim_dt, standalone=standalone,
    )

    print(f"\nDone in {elapsed:.1f} s!")
    if eval_summary["val_all_rmse_deg"] is not None:
        print(f"  Validation RMSE (train gains): {eval_summary['val_all_rmse_deg']:.3f} deg")
    if eval_summary["heldout_all_rmse_deg"] is not None:
        print(f"  Held-out Gain RMSE           : {eval_summary['heldout_all_rmse_deg']:.3f} deg")
    print("  Exported assets:")
    for k, p in exported.items():
        print(f"    - {k:20s}: {p}")
    return dict(config=final_cfg, evaluation=eval_summary, exported=exported, elapsed_s=elapsed)
