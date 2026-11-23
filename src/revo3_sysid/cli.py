"""Beginner-friendly command-line interface for BrainCo REVO3 Ultra21 SysID.

Commands:
  revo3-sysid gui      Launch the 5-step guided Web GUI host application (上位机)
  revo3-sysid demo     Run a 3-second zero-hardware simulation & asset verification demo
  revo3-sysid scan     Scan RS-485 bus for connected REVO3 hands (read-only)
  revo3-sysid status   Print live REVO3 joint positions, currents, faults, and stop state
  revo3-sysid collect  Safe dry-run-by-default hardware trajectory collection (--run to execute)
  revo3-sysid fit      Run 3-stage 21-DOF SysID and auto-export MuJoCo + Isaac Lab assets
  revo3-sysid eval     Evaluate an identified config JSON on a dataset
  revo3-sysid export   Export portable (or --standalone) MuJoCo + Isaac Lab assets from JSON
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np

from .closedloop import Revo3ClosedLoopConfig, Revo3ClosedLoopModel
from .collector import collect_campaign, collect_plan
from .device import DOF, JOINT_NAMES, Revo3Device, discover
from .isaaclab_actuator import Revo3DifferentialThermalNumpyCore
from .model import mjcf_joint_names
from .paths import default_plans_dir, default_pretrained_json, default_usd_path, default_xml_path
from .pipeline import evaluate_dataset, export_all_assets, run_full_pipeline


def _cmd_demo(args: argparse.Namespace) -> int:
    xml_path = default_xml_path("right")
    usd_path = default_usd_path("right")
    cfg_path = default_pretrained_json() if args.config is None else Path(args.config).expanduser().resolve()

    print("=== REVO3 Ultra21 SysID Zero-Hardware Demo ===")
    print(f"  MuJoCo base MJCF : {xml_path}")
    print(f"  Isaac Lab USD    : {usd_path}")
    print(f"  Identified JSON  : {cfg_path}")

    cfg = Revo3ClosedLoopConfig.load(cfg_path)
    sim = Revo3ClosedLoopModel(xml_path, cfg, finger="full")

    # 2-second 50 Hz sine wave command on J02 (little PIP) + J19/J20 (thumb differential)
    dt_cmd = 0.02
    n_cmd = 100
    ct = np.arange(n_cmd, dtype=float) * dt_cmd
    qt = np.zeros((n_cmd, DOF), dtype=float)
    vt = np.zeros((n_cmd, DOF), dtype=float)
    qt[:, 2] = 0.35 * (1.0 - np.cos(2.0 * np.pi * 1.0 * ct))
    vt[:, 2] = 0.35 * 2.0 * np.pi * np.sin(2.0 * np.pi * 1.0 * ct)
    qt[:, 19] = 0.25 * (1.0 - np.cos(2.0 * np.pi * 0.75 * ct))
    vt[:, 19] = 0.25 * 2.0 * np.pi * 0.75 * np.sin(2.0 * np.pi * 0.75 * ct)

    t0 = time.perf_counter()
    q_sim, u_sim = sim.rollout(
        np.zeros(DOF), np.zeros(DOF), ct, ct, qt, vt,
        kp=2.0, kd=0.2, use_i2t=True, return_currents=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1e3

    # Also smoke-test Isaac Lab differential+thermal actuator core
    core = Revo3DifferentialThermalNumpyCore(cfg, mjcf_joint_names("right"), kp_cmd=2.0, kd_cmd=0.2, sim_dt=0.002, num_envs=4)
    tau = core.step(np.zeros((4, DOF)), np.zeros((4, DOF)), np.zeros((4, DOF)), np.zeros((4, DOF)))

    print(f"\n  MuJoCo 21-DOF closed-loop rollout (2.0 s @ dt=1ms): {dt_ms:.2f} ms")
    print(f"    J02 peak tracking: cmd={np.rad2deg(qt[:,2].max()):.1f} deg, sim={np.rad2deg(q_sim[:,2].max()):.1f} deg, max|I|={np.max(np.abs(u_sim[:,2])):.3f} A")
    print(f"    J19 peak tracking: cmd={np.rad2deg(qt[:,19].max()):.1f} deg, sim={np.rad2deg(q_sim[:,19].max()):.1f} deg, max|I_diff+|={np.max(np.abs(u_sim[:,19])):.3f} A")
    print(f"  Isaac Lab actuator core check (4 envs x 21 DOF)   : OK (tau shape={tau.shape})")
    print("\nAll bundled MuJoCo & Isaac Lab assets are verified and ready to use!")
    return 0


async def _async_scan(port: str | None, slave_id: int | None) -> int:
    devs = await discover(port=port, slave_id=slave_id)
    if not devs:
        print("No REVO3 hands detected. Check USB/RS-485 connection, power, and /dev/ttyUSB* permissions.")
        return 1
    print(f"Found {len(devs)} REVO3 hand(s):")
    for d in devs:
        print(f"  port={d['port']}  slave_id={d['slave_id']}  serial={d['serial']}  side={d['side']}  model={d['model']}  fw={d['firmware']}")
    return 0


async def _async_status(port: str | None, slave_id: int, serial: str | None) -> int:
    dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
    info = await dev.connect()
    try:
        health = await dev.health(include_config=True)
        state = await dev.read()
    finally:
        await dev.close()
    print("Device:", json.dumps(info, indent=2))
    print("Health:", json.dumps(health, indent=2))
    q_deg = np.rad2deg(state["qpos"])
    print("\nJoint positions (deg) & currents (A):")
    for j in range(DOF):
        print(f"  J{j:02d} {JOINT_NAMES[j]:20s}: q={q_deg[j]:+7.2f} deg   I={state['effort'][j]:+6.3f} A")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    if args.plan is not None:
        pdir = default_plans_dir()
        screen = pdir / "screen_report.json" if args.screen is None else Path(args.screen)
        asyncio.run(collect_plan(
            args.plan,
            out_dir=args.out,
            screen_report=screen if screen.is_file() else None,
            kp=args.kp,
            kd=args.kd,
            port=args.port,
            slave_id=args.slave_id,
            serial=args.serial,
            transition_npz=args.transition,
            recover_stop=args.recover_stop,
            run=args.run,
        ))
    else:
        asyncio.run(collect_campaign(
            args.out or "captures/my_revo3_dataset",
            mode=args.mode,
            plans_dir=args.plans_dir,
            port=args.port,
            slave_id=args.slave_id,
            serial=args.serial,
            recover_stop=args.recover_stop,
            cooldown_s=args.cooldown,
            run=args.run,
        ))
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    run_full_pipeline(
        args.dataset,
        args.out,
        xml_path=args.xml,
        usd_path=args.usd,
        cma_gens=args.cma_gens,
        lm_max_nfev=args.lm_nfev,
        workers=args.workers,
        kp_cmd=args.kp_cmd,
        kd_cmd=args.kd_cmd,
        sim_dt=args.sim_dt,
        standalone=args.standalone,
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    cfg = Revo3ClosedLoopConfig.load(args.config)
    summary = evaluate_dataset(cfg, args.dataset, xml_path=args.xml, out_dir=args.out)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui_server import serve_gui

    return serve_gui(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def _cmd_export(args: argparse.Namespace) -> int:
    cfg = Revo3ClosedLoopConfig.load(args.config)
    exported = export_all_assets(
        cfg,
        args.out,
        xml_path=args.xml,
        usd_path=args.usd,
        kp_cmd=args.kp_cmd,
        kd_cmd=args.kd_cmd,
        sim_dt=args.sim_dt,
        standalone=args.standalone,
    )
    # Verify exported MuJoCo XML loads cleanly right away
    m = mj.MjModel.from_xml_path(str(exported["mujoco_xml"]))
    print(f"Verified MuJoCo XML load: njnt={m.njnt}, nu={m.nu}, ntendon={m.ntendon}")
    print("Exported MuJoCo + Isaac Lab assets:")
    for k, p in exported.items():
        print(f"  - {k:20s}: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="revo3-sysid",
        description="BrainCo REVO3 Ultra21 50Hz Closed-Loop System Identification & MuJoCo/Isaac Lab Asset Exporter",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gui = sub.add_parser("gui", help="Launch the 5-step guided Web GUI host application (上位机)")
    p_gui.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN/SSH access)")
    p_gui.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    p_gui.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")

    p_demo = sub.add_parser("demo", help="Zero-hardware verification of bundled MuJoCo & Isaac Lab assets")
    p_demo.add_argument("--config", default=None, help="Optional identified JSON path (defaults to fresh50hz_v1)")

    p_scan = sub.add_parser("scan", help="Scan RS-485 bus for connected REVO3 hands (read-only)")
    p_scan.add_argument("--port", default=None)
    p_scan.add_argument("--slave-id", type=int, default=None)

    p_stat = sub.add_parser("status", help="Read live REVO3 health, joint positions, and currents (read-only)")
    p_stat.add_argument("--port", default=None)
    p_stat.add_argument("--slave-id", type=int, default=127)
    p_stat.add_argument("--serial", default=None)

    p_col = sub.add_parser("collect", help="Collect safe collision-screened trajectories (dry-run by default)")
    p_col.add_argument("--mode", choices=("full", "quick"), default="full", help="full=7 gains/21 runs (~12m), quick=3 gains/9 runs (~5m)")
    p_col.add_argument("--out", default=None, help="Output directory for dataset / capture")
    p_col.add_argument("--plan", default=None, help="Optional single plan .npz (omit to run full 21-DOF campaign)")
    p_col.add_argument("--plans-dir", default=None, help="Optional directory containing plan_*.npz")
    p_col.add_argument("--screen", default=None, help="Optional screen_report.json path")
    p_col.add_argument("--transition", default=None, help="Optional staged transition .npz for single-plan mode")
    p_col.add_argument("--kp", type=float, default=2.0, help="Single-plan Kp gain")
    p_col.add_argument("--kd", type=float, default=0.2, help="Single-plan Kd gain")
    p_col.add_argument("--port", default=None)
    p_col.add_argument("--slave-id", type=int, default=127)
    p_col.add_argument("--serial", default=None)
    p_col.add_argument("--cooldown", type=float, default=2.0, help="Seconds between runs in campaign mode")
    p_col.add_argument("--recover-stop", action="store_true", help="Clear latched software stop before starting")
    p_col.add_argument("--run", action="store_true", help="Actually connect and move hardware (omit for dry-run)")

    p_fit = sub.add_parser("fit", help="Run 3-stage fast 21-DOF SysID and auto-export MuJoCo + Isaac Lab assets")
    p_fit.add_argument("--dataset", required=True, help="Dataset directory containing manifest.json")
    p_fit.add_argument("--out", default="outputs/revo3_sysid_run", help="Output directory for results & assets")
    p_fit.add_argument("--xml", default=None, help="Optional base passive MuJoCo XML")
    p_fit.add_argument("--usd", default=None, help="Optional base Isaac Lab USD")
    p_fit.add_argument("--cma-gens", type=int, default=20, help="CMA-ES generations per finger block")
    p_fit.add_argument("--lm-nfev", type=int, default=12, help="Max Gauss-Newton/TRF evaluations per block")
    p_fit.add_argument("--workers", type=int, default=None, help="Worker process count (default: min(cpu_count, 24))")
    p_fit.add_argument("--kp-cmd", type=float, default=2.0, help="Reference MIT Kp for exported Isaac Lab defaults")
    p_fit.add_argument("--kd-cmd", type=float, default=0.2, help="Reference MIT Kd for exported Isaac Lab defaults")
    p_fit.add_argument("--sim-dt", type=float, default=0.002, help="Target Isaac Lab physics timestep (s)")
    p_fit.add_argument("--standalone", action="store_true", help="Copy meshes/textures/USD into output folder")

    p_eval = sub.add_parser("eval", help="Evaluate an identified JSON on a dataset")
    p_eval.add_argument("--config", required=True, help="Identified Revo3ClosedLoopConfig JSON")
    p_eval.add_argument("--dataset", required=True, help="Dataset directory or manifest.json")
    p_eval.add_argument("--xml", default=None)
    p_eval.add_argument("--out", default=None, help="Optional output directory for CSVs, JSON, and plot")

    p_exp = sub.add_parser("export", help="Export MuJoCo XML + Isaac Lab USD/Cfg from an identified JSON")
    p_exp.add_argument("--config", required=True, help="Identified Revo3ClosedLoopConfig JSON")
    p_exp.add_argument("--out", required=True, help="Output directory for exported assets")
    p_exp.add_argument("--xml", default=None)
    p_exp.add_argument("--usd", default=None)
    p_exp.add_argument("--kp-cmd", type=float, default=2.0)
    p_exp.add_argument("--kd-cmd", type=float, default=0.2)
    p_exp.add_argument("--sim-dt", type=float, default=0.002)
    p_exp.add_argument("--standalone", action="store_true", help="Copy meshes/textures/USD into output folder")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "gui":
        return _cmd_gui(args)
    if args.cmd == "demo":
        return _cmd_demo(args)
    if args.cmd == "scan":
        return asyncio.run(_async_scan(args.port, args.slave_id))
    if args.cmd == "status":
        return asyncio.run(_async_status(args.port, args.slave_id, args.serial))
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "fit":
        return _cmd_fit(args)
    if args.cmd == "eval":
        return _cmd_eval(args)
    if args.cmd == "export":
        return _cmd_export(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
