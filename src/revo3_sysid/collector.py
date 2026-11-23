"""Safe, dry-run-by-default hardware trajectory collector for REVO3 Ultra21.

Built directly on Layer 1's tick-synchronized ``Revo3Controller.stream_trajectory()``,
drift-free phase-locked 50 Hz scheduling, native ``max_feedback_current`` watchdog,
and independent post-stop software-stop latch readback.

Supports both single-plan captures (``collect_plan``) and guided full-hand multi-gain
campaigns (``collect_campaign``) that write a ``manifest.json`` directly consumable by
``revo3-sysid fit``.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from .controller import Revo3Controller
from .device import DOF, JOINT_NAMES, Revo3Device
from .paths import default_plans_dir


# Standard multi-gain protocols:
#   - "full":  7 orthogonal (Kp, Kd) pairs (6 train + 1 held-out Kp=2.0/Kd=0.2) -> 21 runs (~12 min)
#   - "quick": 3 orthogonal (Kp, Kd) pairs (2 train + 1 held-out Kp=2.0/Kd=0.2) ->  9 runs (~5 min)
GAIN_PROTOCOLS: dict[str, list[tuple[float, float, str]]] = {
    "full": [
        (0.5, 0.05, "train"),
        (1.0, 0.10, "train"),
        (1.5, 0.20, "train"),
        (2.0, 0.20, "heldout_gain"),
        (3.0, 0.10, "train"),
        (3.5, 0.20, "train"),
        (4.0, 0.05, "train"),
    ],
    "quick": [
        (1.0, 0.10, "train"),
        (2.0, 0.20, "heldout_gain"),
        (3.5, 0.20, "train"),
    ],
}

PLAN_FILES: list[tuple[str, str, str | None]] = [
    ("planC", "plan_C_fingers_J00_J15.npz", None),
    ("planD", "plan_D_thumb_J16_J18.npz", "transition_fingers_to_thumb.npz"),
    ("planT", "plan_T_diff_unified.npz", None),
]


def smooth5(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def smooth5_rate(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def build_quintic_blend(q_start: np.ndarray, q_end: np.ndarray, duration_s: float, rate_hz: float):
    n = max(2, int(round(duration_s * rate_hz)))
    u = np.arange(n, dtype=float) / (n - 1)
    s = smooth5(u)[:, None]
    ds = smooth5_rate(u)[:, None] / duration_s
    q = q_start[None, :] + s * (q_end - q_start)[None, :]
    v = ds * (q_end - q_start)[None, :]
    return q, v


async def collect_plan(
    plan_npz: Path | str,
    *,
    out_dir: Path | str | None = None,
    screen_report: Path | str | None = None,
    kp: float = 2.0,
    kd: float = 0.2,
    port: str | None = None,
    slave_id: int = 127,
    serial: str | None = None,
    max_tracking_error_deg: float = 18.0,
    max_feedback_current_a: float = 0.85,
    health_interval_s: float = 0.5,
    config_check_every: int = 8,
    blend_s: float = 3.0,
    settle_s: float = 1.0,
    transition_npz: Path | str | None = None,
    recover_stop: bool = False,
    run: bool = False,
) -> Path | None:
    """Validate collision screening and execute (or dry-run) a tick-synchronized REVO3 capture."""
    plan_path = Path(plan_npz).expanduser().resolve(strict=True)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    screen_entry = None
    if screen_report is not None:
        sp = Path(screen_report).expanduser().resolve(strict=True)
        sc = json.loads(sp.read_text())
        screen_entry = sc.get(plan_path.name)
        if not screen_entry or not screen_entry.get("passed") or screen_entry.get("blocked"):
            raise RuntimeError(f"Plan {plan_path.name} did NOT pass collision screen {sp}")

    z = np.load(plan_path, allow_pickle=True)
    q_ref = np.asarray(z["q"], dtype=float)
    v_ref = np.asarray(z["v"], dtype=float)
    splits = z["split"].astype(str)
    stages = z["stage"].astype(str)
    rate_hz = float(z["rate_hz"]) if "rate_hz" in z else 50.0
    pair_start = int(z["pair_start"]) if "pair_start" in z else 0

    print(f"Plan: {plan_path.name} (sha256={plan_sha[:12]}...)")
    print(f"  Excitation frames: {len(q_ref) - pair_start} @ {rate_hz:.1f} Hz ({(len(q_ref) - pair_start)/rate_hz:.1f} s)")
    print(f"  Gains: Kp={kp}, Kd={kd} | Guards: max_err={max_tracking_error_deg} deg, max_|I|={max_feedback_current_a} A")
    if screen_entry is not None:
        print(f"  Collision screen: passed={screen_entry['passed']}, checked={screen_entry['checked']}, blocked={len(screen_entry['blocked'])}")

    if not run:
        print("  [DRY-RUN] Verified plan & safety metadata. No hardware opened. Pass --run to execute.")
        return None

    if out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        kp_tag = f"kp{kp:g}_kd{kd:g}".replace(".", "p")
        out_path = Path.cwd() / "captures" / f"capture_{plan_path.stem}_{kp_tag}_{stamp}"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hand = Revo3Controller(
        port=port,
        slave_id=slave_id,
        serial=serial,
        mode="impedance",
        legacy_mit=True,
        recover_stop=recover_stop,
        kp=kp,
        kd=kd,
        rate_hz=rate_hz,
        max_tracking_error=np.deg2rad(max_tracking_error_deg),
        max_feedback_current=max_feedback_current_a,
        health_interval_s=health_interval_s,
        config_check_every=config_check_every,
    )

    completed = False
    controller_error = None
    rec_data = None
    entry_mode = None
    sp_stream = None
    st_stream = None

    await hand.start()
    try:
        q0 = hand.initial_target_qpos.copy()
        lows, highs = hand.joint_limits[:, 0], hand.joint_limits[:, 1]
        if np.any(q_ref < lows[None, :]) or np.any(q_ref > highs[None, :]):
            raise ValueError("Planned trajectory exceeds live device joint limits!")

        q_exc = q_ref[pair_start:]
        v_exc = v_ref[pair_start:]
        q_start_exc = q_exc[0]

        q_blocks, v_blocks, sp_blocks, st_blocks = [], [], [], []
        if transition_npz is not None:
            tz = np.load(Path(transition_npz).expanduser().resolve(strict=True), allow_pickle=True)
            q_tr = np.asarray(tz["q"], dtype=float)
            qb, vb = build_quintic_blend(q0, q_tr[0], min(2.5, blend_s), rate_hz)
            vt_tr = np.gradient(q_tr, 1.0 / rate_hz, axis=0)
            vt_tr[0] = vt_tr[-1] = 0.0
            qb2, vb2 = build_quintic_blend(q_tr[-1], q_start_exc, 1.0, rate_hz)
            q_blocks.extend([qb, q_tr, qb2])
            v_blocks.extend([vb, vt_tr, vb2])
            for arr, stn in ((qb, "entry_blend"), (q_tr, "staged_transition"), (qb2, "align_start")):
                sp_blocks.append(np.full(len(arr), "approach", dtype="<U12"))
                st_blocks.append(np.full(len(arr), stn, dtype="<U18"))
            entry_mode = f"staged_transition ({Path(transition_npz).name})"
        else:
            qb, vb = build_quintic_blend(q0, q_start_exc, blend_s, rate_hz)
            q_blocks.append(qb)
            v_blocks.append(vb)
            sp_blocks.append(np.full(len(qb), "approach", dtype="<U12"))
            st_blocks.append(np.full(len(qb), "entry_blend", dtype="<U18"))
            max_diff = float(np.max(np.abs(np.rad2deg(q0 - q_start_exc))))
            entry_mode = f"direct_quintic_blend (max_diff={max_diff:.2f} deg)"

        settle_n = max(1, int(round(settle_s * rate_hz)))
        q_blocks.append(np.repeat(q_start_exc[None, :], settle_n, axis=0))
        v_blocks.append(np.zeros((settle_n, DOF), dtype=float))
        sp_blocks.append(np.full(settle_n, "approach", dtype="<U12"))
        st_blocks.append(np.full(settle_n, "settle", dtype="<U18"))

        q_blocks.append(q_exc)
        v_blocks.append(v_exc)
        sp_blocks.append(splits[pair_start:].astype("<U12"))
        st_blocks.append(stages[pair_start:].astype("<U18"))

        q_stream = np.vstack(q_blocks)
        v_stream = np.vstack(v_blocks)
        sp_stream = np.concatenate(sp_blocks)
        st_stream = np.concatenate(st_blocks)
        print(f"  Entry: {entry_mode} | total tick-locked stream={len(q_stream)} frames ({len(q_stream)/rate_hz:.1f} s)", flush=True)

        hand.start_recording(seconds=len(q_stream) / rate_hz + 10.0)
        await hand.stream_trajectory(q_stream, v_stream)
        completed = True
        rec_data = hand.stop_recording()
    except Exception as exc:
        controller_error = repr(exc)
        try:
            rec_data = hand.stop_recording()
        except Exception:
            pass
    finally:
        await hand.stop()

    # Independent reconnect read-only verification of latched software stop and health
    after_health = None
    after_q_deg = None
    stop_readback_error = None
    dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
    try:
        await dev.connect()
        try:
            after_health = await dev.health(include_config=True)
            after_state = await dev.read()
            after_q_deg = np.rad2deg(after_state["qpos"]).tolist()
        finally:
            await dev.close()
    except Exception as exc:
        stop_readback_error = repr(exc)

    if rec_data is not None and len(rec_data["tick_ns"]) > 0 and sp_stream is not None:
        ridx = rec_data["ref_index"]
        valid_r = np.clip(ridx, 0, len(sp_stream) - 1)
        out_split = np.where(ridx >= 0, sp_stream[valid_r], "idle")
        out_stage = np.where(ridx >= 0, st_stream[valid_r], "idle")
        np.savez_compressed(
            out_path / "feedback.npz",
            **rec_data,
            side=np.array(str(hand.device_info.get("side", "right")).lower()),
            joint_names=np.asarray(JOINT_NAMES),
            split=out_split,
            stage=out_stage,
        )

    source_info = dict(
        plan_file=plan_path.name,
        plan_sha256=plan_sha,
        entry_mode=entry_mode,
        screen=screen_entry,
        device=hand.device_info,
        rate_hz=rate_hz,
        kp=kp,
        kd=kd,
        max_tracking_error_deg=max_tracking_error_deg,
        max_feedback_current_a=max_feedback_current_a,
        health_interval_s=health_interval_s,
        config_check_every=config_check_every,
    )
    (out_path / "source.json").write_text(json.dumps(source_info, indent=2) + "\n")
    run_info = dict(
        completed=completed,
        controller_error=controller_error,
        stop_readback_error=stop_readback_error,
        frames=0 if rec_data is None else int(len(rec_data["tick_ns"])),
        stats=dict(hand.stats),
        after_stop=after_health,
        after_stop_q_deg=after_q_deg,
    )
    (out_path / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    stop_flag = None if after_health is None else after_health.get("software_stop_enabled")
    print(f"  Saved capture -> {out_path} | completed={completed} | frames={run_info['frames']} | overruns={hand.stats.get('overruns')} | software_stop={stop_flag}")
    if not completed:
        raise RuntimeError(f"Capture aborted: {controller_error}")
    return out_path


async def collect_campaign(
    out_dir: Path | str,
    *,
    mode: str = "full",
    plans_dir: Path | str | None = None,
    port: str | None = None,
    slave_id: int = 127,
    serial: str | None = None,
    recover_stop: bool = False,
    cooldown_s: float = 2.0,
    run: bool = False,
) -> Path:
    """Run a complete 21-DOF multi-gain capture campaign and emit manifest.json.

    Uses ``mode="full"`` (7 gains x 3 plans = 21 runs, ~12 min) or ``mode="quick"``
    (3 gains x 3 plans = 9 runs, ~5 min). Dry-run by default unless ``run=True``.
    """
    if mode not in GAIN_PROTOCOLS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {tuple(GAIN_PROTOCOLS)}")
    pdir = default_plans_dir() if plans_dir is None else Path(plans_dir).expanduser().resolve()
    screen = pdir / "screen_report.json"
    out_root = Path(out_dir).expanduser().resolve()
    gains = GAIN_PROTOCOLS[mode]

    print(f"=== REVO3 21-DOF Capture Campaign (mode={mode}, {len(gains)} gains x 3 plans = {len(gains) * 3} runs) ===")
    print(f"  Plans directory : {pdir}")
    print(f"  Output directory: {out_root}")
    print(f"  Hardware mode   : {'LIVE EXECUTION (--run)' if run else 'DRY-RUN (safe verification only)'}\n")

    captures = []
    idx = 0
    for kp, kd, role in gains:
        gain_tag = f"kp{kp:g}_kd{kd:g}".replace(".", "p")
        for plan_tag, plan_fname, trans_fname in PLAN_FILES:
            idx += 1
            cap_id = f"{plan_tag}_{gain_tag}"
            cap_dir = out_root / cap_id
            plan_path = pdir / plan_fname
            # Use staged transition when switching between finger-parked and thumb-parked poses
            trans_path = (pdir / trans_fname) if (trans_fname is not None and idx > 1) else None
            if plan_tag == "planC" and idx > 1:
                trans_path = pdir / "transition_thumb_to_fingers.npz"
            # Plan T needs slightly wider tracking guard at high gains during step response
            err_guard = 22.0 if (plan_tag == "planT" and kp >= 3.0) else (20.0 if kp <= 0.5 else 18.0)
            i_guard = 0.90 if plan_tag == "planT" else 0.85

            print(f"[{idx:02d}/{len(gains)*3:02d}] {cap_id} ({role})")
            res = await collect_plan(
                plan_path,
                out_dir=cap_dir,
                screen_report=screen if screen.is_file() else None,
                kp=kp,
                kd=kd,
                port=port,
                slave_id=slave_id,
                serial=serial,
                max_tracking_error_deg=err_guard,
                max_feedback_current_a=i_guard,
                transition_npz=trans_path,
                recover_stop=(recover_stop or idx > 1),
                run=run,
            )
            if run and res is not None:
                captures.append(dict(
                    id=cap_id,
                    plan=plan_tag,
                    kp=kp,
                    kd=kd,
                    gain_role=role,
                    path=cap_id,  # relative to manifest.json directory!
                ))
                if cooldown_s > 0:
                    await asyncio.sleep(cooldown_s)

    if run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / "manifest.json"
        manifest = dict(
            format="revo3-sysid-dataset-v1",
            mode=mode,
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            captures=captures,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nCampaign complete! Saved manifest -> {manifest_path}")
        print(f"Next step: revo3-sysid fit --dataset {out_root}")
        return manifest_path
    else:
        print("\n[DRY-RUN COMPLETE] All trajectories verified against collision screen.")
        print(f"To run on a real REVO3 hand, re-run with --run (and --recover-stop if software stop is latched).")
        return out_root
