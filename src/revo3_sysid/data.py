"""Unified multi-source trial loading, evaluation, and closed-loop CMA-ES sysid for REVO3.

Bridges historical PACE recordings (raw_pace_order.npz) and new aiowuji captures
(feedback.npz) into a single Revo3Trial representation compatible with Revo3ClosedLoopModel.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .closedloop import FINGER_SLICES, Revo3ClosedLoopConfig, Revo3ClosedLoopModel
from .device import DOF, JOINT_NAMES
from .model import mjcf_joint_names


@dataclass
class Revo3Trial:
    """Unified 21-DOF recording in native SDK J0-J20 order with both feedback & command clocks."""
    source: str
    side: str
    t_fb: np.ndarray        # (M,) feedback host timestamps in seconds (starts at 0)
    q: np.ndarray           # (M, 21) rad
    v: np.ndarray           # (M, 21) rad/s
    effort: np.ndarray      # (M, 21) A
    ct: np.ndarray          # (K,) command completion timestamps in seconds (same epoch as t_fb)
    qt: np.ndarray          # (K, 21) rad
    vt: np.ndarray          # (K, 21) rad/s
    kp: np.ndarray          # (K, 21) SDK native
    kd: np.ndarray          # (K, 21) SDK native
    split: np.ndarray       # (M,) str label per feedback frame ("fit", "validation", "baseline", ...)
    stage: np.ndarray       # (M,) str label per feedback frame
    metadata: dict


@dataclass
class TrialFragment:
    """Contiguous playback slice for one finger/joint subset with full prior command history."""
    source: str
    finger: str
    joint_indices: np.ndarray
    split: str
    stage: str
    t_fb: np.ndarray        # (m,) s
    q: np.ndarray           # (m, n_dof) rad
    v: np.ndarray           # (m, n_dof) rad/s
    effort: np.ndarray      # (m, n_dof) A
    ct: np.ndarray          # (k,) s (includes command before t_fb[0])
    qt: np.ndarray          # (k, n_dof) rad
    vt: np.ndarray          # (k, n_dof) rad/s
    kp: np.ndarray          # (n_dof,) SDK gain (constant across fragment)
    kd: np.ndarray          # (n_dof,) SDK gain


def load_trial_unified(path: Path | str, side: str | None = None) -> Revo3Trial:
    """Load either an aiowuji feedback.npz or a PACE raw_pace_order.npz into SDK J0-J20 order."""
    p = Path(path).expanduser().resolve(strict=True)
    with np.load(p, allow_pickle=True) as z:
        keys = set(z.keys())
        names = [str(x) for x in z["joint_names"].tolist()]
        if "state_host_ns" in keys:
            # aiowuji feedback.npz
            file_side = str(z["side"].item()).lower() if "side" in keys else None
            if file_side is None:
                src_json = p.parent / "source.json"
                if src_json.exists():
                    info = json.loads(src_json.read_text())
                    file_side = str(info.get("device", {}).get("side", "")).lower() or None
            resolved_side = (side or file_side or "right").lower()
            if names != list(JOINT_NAMES):
                raise ValueError(f"{p}: expected SDK JOINT_NAMES order")
            st_ns = z["state_host_ns"].astype(np.int64)
            t0 = int(st_ns[0])
            t_fb = (st_ns - t0).astype(float) * 1e-9
            sent = z["command_sent"].astype(bool)
            if not np.any(sent):
                raise ValueError(f"{p}: no sent commands in recording")
            ct = (z["send_return_ns"][sent].astype(np.int64) - t0).astype(float) * 1e-9
            q = np.asarray(z["q"], float)
            v = np.asarray(z["qd"], float)
            effort = np.asarray(z["effort"], float)
            qt = np.asarray(z["q_cmd"][sent], float)
            vt = np.asarray(z["qd_cmd"][sent], float)
            kp = np.asarray(z["kp"][sent], float)
            kd = np.asarray(z["kd"][sent], float)
            split = z["split"].astype(str) if "split" in keys else np.full(len(t_fb), "unknown")
            stage = z["stage"].astype(str) if "stage" in keys else np.full(len(t_fb), "unknown")
            meta = {}
            for jf in ("source.json", "run.json"):
                jp = p.parent / jf
                if jp.exists():
                    meta[jf] = json.loads(jp.read_text())
        else:
            # PACE raw_pace_order.npz
            resolved_side = (side or "right").lower()
            model_names = mjcf_joint_names(resolved_side)
            if len(names) != DOF or set(names) != set(model_names):
                raise ValueError(f"{p}: PACE joint names mismatch for {resolved_side}")
            order = [names.index(n) for n in model_names]
            t_raw = np.asarray(z["feedback_time_s"], float)
            t0 = float(t_raw[0])
            t_fb = t_raw - t0
            ct = np.asarray(z["command_return_time_s"], float) - t0
            q = np.asarray(z["q_measured_rad"], float)[:, order]
            v = np.asarray(z["qd_measured_rad_s"], float)[:, order]
            effort = np.asarray(z["current_a"], float)[:, order]
            qt = np.asarray(z["q_target_rad"], float)[:, order]
            vt = np.asarray(z["qd_target_rad_s"], float)[:, order]
            kp_raw = np.asarray(z["command_kp_sdk"], float)
            kd_raw = np.asarray(z["command_kd_sdk"], float)
            if kp_raw.ndim == 1:
                kp = np.broadcast_to(kp_raw[:, None], (len(ct), DOF)).copy()
            else:
                kp = kp_raw[:, order]
            if kd_raw.ndim == 1:
                kd = np.broadcast_to(kd_raw[:, None], (len(ct), DOF)).copy()
            else:
                kd = kd_raw[:, order]
            c_stage = z["command_stage"].astype(str)
            ix = np.clip(np.searchsorted(ct, t_fb, side="right") - 1, 0, len(ct) - 1)
            stage = c_stage[ix]
            # Standard PACE split convention: multisine/chirp -> fit, constant_4Hz/steps -> validation
            split = np.where(
                np.char.find(stage, "multisine") >= 0,
                "fit",
                np.where(
                    np.char.find(stage, "chirp") >= 0,
                    "fit",
                    np.where(
                        (np.char.find(stage, "4Hz") >= 0) | (np.char.find(stage, "steps") >= 0),
                        "validation",
                        "other",
                    ),
                ),
            )
            meta = json.loads(z["metadata_json"].item()) if "metadata_json" in keys else {}

    return Revo3Trial(
        source=str(p),
        side=resolved_side,
        t_fb=t_fb,
        q=q,
        v=v,
        effort=effort,
        ct=ct,
        qt=qt,
        vt=vt,
        kp=kp,
        kd=kd,
        split=split,
        stage=stage,
        metadata=meta,
    )


def extract_fragments(
    trial: Revo3Trial,
    finger: str = "full",
    *,
    splits: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    max_gap_s: float = 0.10,
    min_frames: int = 20,
    group_by_stage: bool = False,
) -> list[TrialFragment]:
    """Split a Revo3Trial into contiguous fragments suitable for closed-loop rollout."""
    jidx = np.array(FINGER_SLICES[finger], dtype=int)
    mask = np.ones(len(trial.t_fb), dtype=bool)
    if splits is not None:
        mask &= np.isin(trial.split, list(splits))
    if stages is not None:
        mask &= np.isin(trial.stage, list(stages))

    idx = np.flatnonzero(mask)
    if len(idx) < min_frames:
        return []

    # Break on non-contiguous index, large feedback gap, or optional stage change
    breaks = [0]
    for pos in range(1, len(idx)):
        i_prev, i_cur = idx[pos - 1], idx[pos]
        if (i_cur != i_prev + 1
                or (trial.t_fb[i_cur] - trial.t_fb[i_prev]) > max_gap_s
                or (group_by_stage and trial.stage[i_cur] != trial.stage[i_prev])):
            breaks.append(pos)
    breaks.append(len(idx))

    frags: list[TrialFragment] = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        seg = idx[a:b]
        if len(seg) < min_frames:
            continue
        t_seg = trial.t_fb[seg]
        # Ensure we have command history up to 0.10s before t_seg[0]
        c_start = max(0, int(np.searchsorted(trial.ct, t_seg[0] - 0.10, side="right")) - 1)
        c_end = min(len(trial.ct), int(np.searchsorted(trial.ct, t_seg[-1] + 0.05, side="right")) + 1)
        if c_end - c_start < 2:
            continue
        ct_seg = trial.ct[c_start:c_end]
        kp_seg = np.median(trial.kp[c_start:c_end][:, jidx], axis=0)
        kd_seg = np.median(trial.kd[c_start:c_end][:, jidx], axis=0)
        frags.append(TrialFragment(
            source=trial.source,
            finger=finger,
            joint_indices=jidx,
            split=str(trial.split[seg[0]]),
            stage=str(trial.stage[seg[0]]),
            t_fb=t_seg.copy(),
            q=trial.q[seg][:, jidx].copy(),
            v=trial.v[seg][:, jidx].copy(),
            effort=trial.effort[seg][:, jidx].copy(),
            ct=ct_seg.copy(),
            qt=trial.qt[c_start:c_end][:, jidx].copy(),
            vt=trial.vt[c_start:c_end][:, jidx].copy(),
            kp=kp_seg,
            kd=kd_seg,
        ))
    return frags


def evaluate_fragments(
    cl_model: Revo3ClosedLoopModel,
    fragments: list[TrialFragment],
    *,
    use_i2t: bool = True,
    skip_first: bool = True,
) -> dict:
    """Evaluate a Revo3ClosedLoopModel across a list of TrialFragments on native clocks."""
    n_dof = cl_model.n_dof
    sse = np.zeros(n_dof, dtype=float)
    abs_sum = np.zeros(n_dof, dtype=float)
    within2 = np.zeros(n_dof, dtype=int)
    within5 = np.zeros(n_dof, dtype=int)
    count = 0
    per_fragment = []

    for f in fragments:
        pred = cl_model.rollout(
            f.q[0], f.v[0], f.t_fb, f.ct, f.qt, f.vt, f.kp, f.kd, use_i2t=use_i2t
        )
        sl = slice(1, None) if skip_first else slice(None)
        e_deg = np.rad2deg(pred[sl] - f.q[sl])
        n = len(e_deg)
        count += n
        sse += np.sum(e_deg**2, axis=0)
        abs_sum += np.sum(np.abs(e_deg), axis=0)
        within2 += np.sum(np.abs(e_deg) <= 2.0, axis=0)
        within5 += np.sum(np.abs(e_deg) <= 5.0, axis=0)
        per_fragment.append(dict(
            source=Path(f.source).parent.name,
            split=f.split,
            stage=f.stage,
            kp=float(np.mean(f.kp)),
            kd=float(np.mean(f.kd)),
            n=n,
            rmse_deg=np.sqrt(np.mean(e_deg**2, axis=0)).tolist(),
        ))

    rmse_deg = np.sqrt(sse / max(1, count))
    return dict(
        finger=cl_model.finger,
        joint_indices=cl_model.joint_indices.tolist(),
        n_frames=int(count),
        overall_rmse_deg=float(np.sqrt(np.sum(sse) / max(1, count * n_dof))),
        per_joint_rmse_deg={f"J{int(j):02d}": float(r) for j, r in zip(cl_model.joint_indices, rmse_deg)},
        per_joint_mae_deg={f"J{int(j):02d}": float(a / max(1, count)) for j, a in zip(cl_model.joint_indices, abs_sum)},
        per_joint_within2_pct={f"J{int(j):02d}": float(100.0 * w / max(1, count)) for j, w in zip(cl_model.joint_indices, within2)},
        fragments=per_fragment,
    )


def analyze_quasistatic_ramp(trial: Revo3Trial, joint_indices: Iterable[int], stage_name: str = "qs_ramp") -> dict:
    """Estimate Coulomb friction torque proxy and static gravity+spring curve from a slow up/down ramp.

    For slow constant-speed motion (|v| in [0.04, 0.30] rad/s):
      I_up(q)   ≈ g_spring(q) + f_coulomb + b*v
      I_down(q) ≈ g_spring(q) - f_coulomb - b*v
    So (I_up - I_down)/2 isolates dry friction and (I_up + I_down)/2 isolates gravity+spring!
    """
    m = (trial.stage == stage_name)
    out = {}
    if not np.any(m):
        return out
    q_seg = trial.q[m]
    v_seg = trial.v[m]
    i_seg = trial.effort[m]
    for j in joint_indices:
        qj, vj, ij = q_seg[:, j], v_seg[:, j], i_seg[:, j]
        up = vj > 0.04
        dn = vj < -0.04
        if up.sum() < 15 or dn.sum() < 15:
            out[f"J{j:02d}"] = dict(status="insufficient_ramp")
            continue
        q_lo = max(np.percentile(qj[up], 10), np.percentile(qj[dn], 10))
        q_hi = min(np.percentile(qj[up], 90), np.percentile(qj[dn], 90))
        if q_hi - q_lo < np.deg2rad(5.0):
            out[f"J{j:02d}"] = dict(status="insufficient_overlap")
            continue
        grid = np.linspace(q_lo, q_hi, 25)
        # Fit linear trend I(q) separately for up and down branches
        c_up = np.polyfit(qj[up], ij[up], 1)
        c_dn = np.polyfit(qj[dn], ij[dn], 1)
        i_up = np.polyval(c_up, grid)
        i_dn = np.polyval(c_dn, grid)
        half_hyst = 0.5 * (i_up - i_dn)
        mid_curve = 0.5 * (i_up + i_dn)
        out[f"J{j:02d}"] = dict(
            status="ok",
            q_range_deg=[float(np.rad2deg(q_lo)), float(np.rad2deg(q_hi))],
            coulomb_proxy_a=float(np.median(half_hyst)),
            static_slope_a_per_rad=float(0.5 * (c_up[0] + c_dn[0])),
            static_mid_a=float(np.median(mid_curve)),
        )
    return out
