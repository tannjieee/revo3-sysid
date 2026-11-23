"""Isolated single-shot subprocess worker for read-only hardware scan/status and E-STOP.

Keeping ``bc_revo3_sdk`` (native Rust/Tokio) strictly out of the long-lived HTTP GUI
server process guarantees zero thread/fork conflicts, zero FD leaks, and total crash
isolation if USB is unplugged or the native SDK panics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np

from .device import DOF, JOINT_NAMES, Revo3Device, discover


async def _do_scan(port: str | None, slave_id: int | None) -> dict:
    devs = await discover(port=port, slave_id=slave_id)
    return dict(ok=True, devices=devs)


async def _do_status(port: str | None, slave_id: int, serial: str | None) -> dict:
    dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
    info = await dev.connect()
    try:
        health = await dev.health(include_config=True)
        st = await dev.read()
        limits_deg = np.rad2deg(dev.limits).tolist()
    finally:
        await dev.close()

    q_deg = np.rad2deg(st["qpos"]).tolist()
    eff = st["effort"].tolist()
    joints = []
    warnings = []
    for j in range(DOF):
        lo, hi = limits_deg[j]
        qj = q_deg[j]
        out_of_range = (qj < lo - 1.5) or (qj > hi + 1.5)
        near_limit = (qj < lo) or (qj > hi)
        if out_of_range:
            warnings.append(
                f"J{j:02d} ({JOINT_NAMES[j]}) 当前角度 {qj:+.2f}° 超出软限位 [{lo:.1f}°, {hi:.1f}°] 超过 1.5°，请轻托该手指后再启动采数。"
            )
        joints.append(dict(
            id=f"J{j:02d}",
            index=j,
            name=JOINT_NAMES[j],
            q_deg=round(qj, 2),
            min_deg=round(lo, 1),
            max_deg=round(hi, 1),
            effort_a=round(eff[j], 3),
            fault_code=int(health["motor_fault_codes"][j]),
            out_of_range=out_of_range,
            near_limit=near_limit,
        ))
    return dict(
        ok=True,
        device=info,
        health=health,
        joints=joints,
        warnings=warnings,
        ts=time.strftime("%H:%M:%S"),
    )


async def _do_estop(port: str | None, slave_id: int, serial: str | None) -> dict:
    dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
    await dev.connect()
    try:
        await dev.stop()
        health = await dev.health(include_config=True)
        return dict(ok=True, software_stop_enabled=health.get("software_stop_enabled"), health=health)
    finally:
        await dev.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revo3_sysid._hw_worker")
    ap.add_argument("action", choices=("scan", "status", "estop"))
    ap.add_argument("--port", default=None)
    ap.add_argument("--slave-id", type=int, default=None)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args(argv)

    try:
        if args.action == "scan":
            res = asyncio.run(_do_scan(args.port, args.slave_id))
        elif args.action == "status":
            res = asyncio.run(_do_status(args.port, args.slave_id or 127, args.serial))
        elif args.action == "estop":
            res = asyncio.run(_do_estop(args.port, args.slave_id or 127, args.serial))
        else:
            res = dict(ok=False, error=f"unknown action {args.action}")
    except Exception as exc:
        res = dict(ok=False, error=str(exc))
    sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
