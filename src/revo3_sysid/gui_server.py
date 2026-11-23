"""Zero-dependency local Web GUI host server for guided REVO3 Ultra21 SysID.

Serves a 5-step interactive engineering wizard over ``http.server.ThreadingHTTPServer``
with Server-Sent Events (SSE) live log streaming, subprocess-isolated ``collect`` and
``fit`` execution (100% fork-safe for 24-core CMA-ES/LM), real-time 21-DOF health
telemetry, emergency software-stop (E-STOP), and one-click ZIP asset download.
"""
from __future__ import annotations

import asyncio
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
import zipfile

import numpy as np

from ._c_rollout import get_c_rollout_lib
from .closedloop import Revo3ClosedLoopConfig, Revo3ClosedLoopModel
from .device import DOF, JOINT_NAMES, Revo3Device, discover
from .isaaclab_actuator import Revo3DifferentialThermalNumpyCore
from .model import mjcf_joint_names
from .paths import (
    default_plans_dir,
    default_pretrained_dir,
    default_pretrained_json,
    default_usd_path,
    default_xml_path,
    package_dir,
    repo_root,
)
from .pipeline import export_all_assets


_PROGRESS_COLLECT_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)")
_PROGRESS_BLOCK_RE = re.compile(r"\[(little|ring|middle|index|thumb_prox|thumb_diff)\s*\]")


def _sdk_available() -> tuple[bool, str]:
    try:
        from bc_revo3_sdk import main_mod
        return True, str(main_mod.get_sdk_version())
    except Exception as exc:
        return False, str(exc)


def _discover_datasets(work_root: Path) -> list[dict]:
    found: list[dict] = []
    search_roots = [work_root / "captures", Path.cwd() / "captures", repo_root() / "captures"]
    seen: set[Path] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for mf in sorted(base.glob("*/manifest.json"), reverse=True):
            res = mf.parent.resolve()
            if res in seen:
                continue
            seen.add(res)
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
                found.append(dict(
                    name=res.name,
                    path=str(res),
                    mode=raw.get("mode", "custom"),
                    captures=len(raw.get("captures", [])),
                    created_at=raw.get("created_at", ""),
                ))
            except Exception:
                continue
    return found


def _discover_models(work_root: Path) -> list[dict]:
    models: list[dict] = []
    pre = default_pretrained_json()
    if pre.is_file():
        models.append(dict(
            label="内置预辨识模型 (fresh50hz_v1 · 留出误差 0.338°)",
            path=str(pre),
            dir=str(pre.parent),
            builtin=True,
        ))
    search_roots = [work_root / "outputs", Path.cwd() / "outputs", repo_root() / "outputs"]
    seen: set[Path] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for jf in sorted(base.glob("*/revo3_*_identified.json"), reverse=True):
            res = jf.resolve()
            if res in seen:
                continue
            seen.add(res)
            models.append(dict(
                label=f"本地辨识结果 ({res.parent.name})",
                path=str(res),
                dir=str(res.parent),
                builtin=False,
            ))
    return models


class GuiAppState:
    """Thread-safe backend coordinator for the REVO3 Web GUI."""

    def __init__(self, work_root: Path | None = None) -> None:
        self.work_root = (Path.cwd() if work_root is None else Path(work_root)).expanduser().resolve()
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[str]] = []
        self._log_history: list[dict] = []
        self._proc: subprocess.Popen[str] | None = None
        self.active_job: dict | None = None
        self.last_scan: list[dict] = []
        self.last_status: dict | None = None
        self.last_dataset: str | None = None
        self.last_fit_out: str | None = None
        self.last_export_out: str | None = str(default_pretrained_dir())
        self.last_export_files: dict[str, str] = {}
        self.selected_device: dict = dict(port=None, slave_id=127, serial=None)

    # --- Event bus (SSE) ----------------------------------------------------
    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def emit(self, event: str, data: dict) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        with self._lock:
            if event == "log":
                self._log_history.append(data)
                if len(self._log_history) > 400:
                    self._log_history = self._log_history[-400:]
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def log(self, text: str, level: str = "info") -> None:
        self.emit("log", dict(ts=time.strftime("%H:%M:%S"), level=level, text=text))

    # --- State snapshot -----------------------------------------------------
    def snapshot(self) -> dict:
        sdk_ok, sdk_info = _sdk_available()
        c_ok = get_c_rollout_lib() is not None
        with self._lock:
            job = None if self.active_job is None else dict(self.active_job)
            logs = list(self._log_history[-120:])
        return dict(
            work_root=str(self.work_root),
            repo_root=str(repo_root()),
            sdk_available=sdk_ok,
            sdk_info=sdk_info,
            c_kernel_ready=c_ok,
            cpu_count=os.cpu_count() or 1,
            xml_path=str(default_xml_path("right")),
            usd_path=str(default_usd_path("right")),
            plans_dir=str(default_plans_dir()),
            pretrained_json=str(default_pretrained_json()),
            pretrained_dir=str(default_pretrained_dir()),
            active_job=job,
            selected_device=dict(self.selected_device),
            last_scan=list(self.last_scan),
            last_status=self.last_status,
            last_dataset=self.last_dataset,
            last_fit_out=self.last_fit_out,
            last_export_out=self.last_export_out,
            datasets=_discover_datasets(self.work_root),
            models=_discover_models(self.work_root),
            logs=logs,
        )

    # --- Managed subprocess runner ------------------------------------------
    def start_subprocess_job(
        self,
        kind: str,
        cmd: list[str],
        *,
        meta: dict | None = None,
        on_done=None,
    ) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(f"当前已有正在运行的任务：{self.active_job['kind']}，请等待完成或先点击停止。")
            job = dict(
                id=f"{kind}_{int(time.time())}",
                kind=kind,
                status="running",
                progress=0.0,
                stage="启动中...",
                started_at=time.time(),
                meta=dict(meta or {}),
                returncode=None,
            )
            self.active_job = job

        env = os.environ.copy()
        src_dir = str(package_dir().parent)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir
        env["PYTHONUNBUFFERED"] = "1"

        self.log(f"$ {' '.join(cmd)}", level="cmd")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.work_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with self._lock:
            self._proc = proc
        self.emit("job", dict(job))

        def _reader() -> None:
            completed_blocks: set[str] = set()
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    self.log(line)
                    # Parse progress heuristics
                    updated = False
                    if kind == "collect":
                        m = _PROGRESS_COLLECT_RE.search(line)
                        if m:
                            cur, tot, cap_id = int(m.group(1)), int(m.group(2)), m.group(3)
                            job["progress"] = round(100.0 * (cur - 1) / max(tot, 1), 1)
                            job["stage"] = f"采集中 {cur}/{tot}: {cap_id}"
                            updated = True
                        elif "Saved capture ->" in line or "[DRY-RUN]" in line:
                            updated = True
                    elif kind == "fit":
                        if "Stage 1:" in line:
                            job["progress"] = 8.0
                            job["stage"] = "Stage 1 · 多增益电学律辨识"
                            updated = True
                        elif "Stage 2:" in line:
                            job["progress"] = 18.0
                            job["stage"] = "Stage 2 · 逆动力学力学初值估计"
                            updated = True
                        elif "Stage 3:" in line:
                            job["progress"] = 25.0
                            job["stage"] = "Stage 3 · Option A 混合闭环优化"
                            updated = True
                        else:
                            mb = _PROGRESS_BLOCK_RE.search(line)
                            if mb:
                                completed_blocks.add(mb.group(1))
                                job["progress"] = round(25.0 + 65.0 * len(completed_blocks) / 6.0, 1)
                                job["stage"] = f"Stage 3 · 已完成 {len(completed_blocks)}/6 指块 ({mb.group(1)})"
                                updated = True
                            elif "Stage 4:" in line:
                                job["progress"] = 92.0
                                job["stage"] = "Stage 4 · 留出验证与 MuJoCo/Isaac Lab 资产导出"
                                updated = True
                    if updated:
                        self.emit("job", dict(job))
                rc = proc.wait()
            except Exception as exc:
                self.log(f"后台任务异常: {exc!r}", level="error")
                rc = -1

            job["returncode"] = int(rc)
            if job.get("status") != "aborted":
                job["status"] = "succeeded" if rc == 0 else "failed"
            if rc == 0:
                job["progress"] = 100.0
                job["stage"] = "已完成"
            else:
                job["stage"] = "已中止" if job["status"] == "aborted" else f"执行失败 (exit={rc})"
            job["finished_at"] = time.time()

            extra = None
            if rc == 0 and on_done is not None:
                try:
                    extra = on_done(job)
                except Exception as exc:
                    self.log(f"后处理回调异常: {exc!r}", level="error")
            if extra is not None:
                job["result"] = extra

            with self._lock:
                self._proc = None
                self.active_job = dict(job)
            self.emit("job", dict(job))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return dict(job)

    def stop_active_job(self) -> bool:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return False
            if self.active_job is not None:
                self.active_job["status"] = "aborted"
                self.active_job["stage"] = "正在停止..."
        self.log("用户请求停止当前任务，正在发送 SIGINT/SIGTERM...", level="warn")
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        return True

    # --- Safe path check ----------------------------------------------------
    def is_allowed_path(self, target: Path) -> bool:
        t = target.expanduser().resolve()
        allowed_roots = [
            repo_root().resolve(),
            self.work_root.resolve(),
            Path.cwd().resolve(),
        ]
        for root in allowed_roots:
            try:
                t.relative_to(root)
                return True
            except ValueError:
                continue
        return False


def _run_demo_action(state: GuiAppState) -> dict:
    xml_path = default_xml_path("right")
    usd_path = default_usd_path("right")
    cfg_path = default_pretrained_json()
    cfg = Revo3ClosedLoopConfig.load(cfg_path)
    sim = Revo3ClosedLoopModel(xml_path, cfg, finger="full")

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
    rollout_ms = round((time.perf_counter() - t0) * 1e3, 2)

    core = Revo3DifferentialThermalNumpyCore(cfg, mjcf_joint_names("right"), kp_cmd=2.0, kd_cmd=0.2, sim_dt=0.002, num_envs=4)
    tau = core.step(np.zeros((4, DOF)), np.zeros((4, DOF)), np.zeros((4, DOF)), np.zeros((4, DOF)))

    state.log(f"[自检通过] MuJoCo 21-DOF 闭环仿真 2.0s 耗时 {rollout_ms:.1f} ms | Isaac Lab 核心输出 {tuple(tau.shape)}", level="ok")
    return dict(
        ok=True,
        rollout_ms=rollout_ms,
        c_kernel_ready=get_c_rollout_lib() is not None,
        xml_exists=xml_path.is_file(),
        usd_exists=usd_path.is_file(),
        isaac_tau_shape=list(tau.shape),
        waveforms=dict(
            t_s=np.round(ct, 3).tolist(),
            j02_cmd_deg=np.round(np.rad2deg(qt[:, 2]), 2).tolist(),
            j02_sim_deg=np.round(np.rad2deg(q_sim[:, 2]), 2).tolist(),
            j19_cmd_deg=np.round(np.rad2deg(qt[:, 19]), 2).tolist(),
            j19_sim_deg=np.round(np.rad2deg(q_sim[:, 19]), 2).tolist(),
            j19_i_a=np.round(u_sim[:, 19], 3).tolist(),
        ),
    )


async def _estop_hardware(port: str | None, slave_id: int, serial: str | None) -> dict:
    dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
    await dev.connect()
    try:
        await dev.software_stop()
        health = await dev.health(include_config=True)
        return dict(ok=True, software_stop_enabled=health.get("software_stop_enabled"), health=health)
    finally:
        await dev.close()


def _make_handler(state: GuiAppState):
    static_dir = package_dir() / "gui_static"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            # Keep terminal output clean; events go to the Web GUI log console
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                html = (static_dir / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if path == "/api/state":
                self._send_json(state.snapshot())
                return

            if path == "/api/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = state.subscribe()
                try:
                    # Initial ping
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15.0)
                            self.wfile.write(msg.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    state.unsubscribe(q)
                return

            if path == "/api/file":
                raw_p = (qs.get("path") or [""])[0]
                if not raw_p:
                    self._send_json(dict(error="missing path"), 400)
                    return
                target = Path(raw_p).expanduser().resolve()
                if not state.is_allowed_path(target) or not target.is_file():
                    self._send_json(dict(error="file not found or access denied"), 404)
                    return
                ctype, _ = mimetypes.guess_type(str(target))
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/api/download_zip":
                raw_dir = (qs.get("dir") or [state.last_export_out or ""])[0]
                target_dir = Path(raw_dir).expanduser().resolve()
                if not state.is_allowed_path(target_dir) or not target_dir.is_dir():
                    self._send_json(dict(error="export directory not found"), 404)
                    return
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in sorted(target_dir.rglob("*")):
                        if f.is_file():
                            zf.write(f, arcname=f.relative_to(target_dir).as_posix())
                data = buf.getvalue()
                fname = f"revo3_assets_{target_dir.name}.zip"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self._send_json(dict(error="not found"), 404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json(dict(error=f"invalid JSON: {exc}"), 400)
                return

            try:
                if path == "/api/demo":
                    res = _run_demo_action(state)
                    self._send_json(res)
                    return

                if path == "/api/scan":
                    port = body.get("port") or None
                    slave_id = body.get("slave_id")
                    slave_id = None if slave_id in (None, "") else int(slave_id)
                    state.log(f"正在扫描 RS-485 总线 (port={port or 'auto'}, slave_id={slave_id or 'all'})...")
                    devs = asyncio.run(discover(port=port, slave_id=slave_id))
                    state.last_scan = devs
                    if devs:
                        state.selected_device = dict(
                            port=devs[0]["port"],
                            slave_id=devs[0]["slave_id"],
                            serial=devs[0]["serial"],
                        )
                        state.log(f"发现 {len(devs)} 台 REVO3 设备：{devs[0]['serial']} ({devs[0]['port']}, fw={devs[0]['firmware']})", level="ok")
                    else:
                        state.log("未检测到 REVO3 设备，请检查供电、USB 转 RS-485 接线及串口权限。", level="warn")
                    self._send_json(dict(ok=True, devices=devs))
                    return

                if path == "/api/status":
                    port = body.get("port") or state.selected_device.get("port") or None
                    slave_id = int(body.get("slave_id") or state.selected_device.get("slave_id") or 127)
                    serial = body.get("serial") or state.selected_device.get("serial") or None
                    state.selected_device = dict(port=port, slave_id=slave_id, serial=serial)

                    async def _read_status():
                        dev = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=True)
                        info = await dev.connect()
                        try:
                            health = await dev.health(include_config=True)
                            st = await dev.read()
                            limits_deg = np.rad2deg(dev.limits).tolist()
                        finally:
                            await dev.close()
                        return info, health, st, limits_deg

                    info, health, st, limits_deg = asyncio.run(_read_status())
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
                            warnings.append(f"J{j:02d} ({JOINT_NAMES[j]}) 当前角度 {qj:+.2f}° 超出软限位 [{lo:.1f}°, {hi:.1f}°] 超过 1.5°，请轻托该手指后再启动采数。")
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
                    payload = dict(
                        ok=True,
                        device=info,
                        health=health,
                        joints=joints,
                        warnings=warnings,
                        ts=time.strftime("%H:%M:%S"),
                    )
                    state.last_status = payload
                    state.log(
                        f"状态读取成功：{info['serial']} | safety={health['safety_state']} | faults={health['faulted_motor_count']} | stop_latched={health['software_stop_enabled']}",
                        level="warn" if warnings else "ok",
                    )
                    self._send_json(payload)
                    return

                if path == "/api/estop":
                    stopped_job = state.stop_active_job()
                    hw_res = None
                    sdk_ok, _ = _sdk_available()
                    if sdk_ok and (body.get("hardware", True)):
                        port = body.get("port") or state.selected_device.get("port") or None
                        slave_id = int(body.get("slave_id") or state.selected_device.get("slave_id") or 127)
                        serial = body.get("serial") or state.selected_device.get("serial") or None
                        try:
                            hw_res = asyncio.run(_estop_hardware(port, slave_id, serial))
                            state.log(f"[E-STOP] 已向硬件下发 software_stop()，回读锁存={hw_res.get('software_stop_enabled')}", level="warn")
                        except Exception as exc:
                            hw_res = dict(ok=False, error=str(exc))
                            state.log(f"[E-STOP] 已中止本地任务；硬件重连下发提示：{exc}", level="warn")
                    else:
                        state.log("[E-STOP] 已中止当前本地任务（当前未连接硬件 SDK）。", level="warn")
                    self._send_json(dict(ok=True, stopped_job=stopped_job, hardware=hw_res))
                    return

                if path == "/api/stop_job":
                    stopped = state.stop_active_job()
                    self._send_json(dict(ok=True, stopped=stopped))
                    return

                if path == "/api/collect":
                    mode = str(body.get("mode", "full"))
                    run_live = bool(body.get("run", False))
                    recover_stop = bool(body.get("recover_stop", True))
                    out_name = str(body.get("out") or f"captures/revo3_{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
                    out_path = Path(out_name)
                    if not out_path.is_absolute():
                        out_path = (state.work_root / out_path).resolve()

                    cmd = [
                        sys.executable, "-u", "-m", "revo3_sysid", "collect",
                        "--mode", mode,
                        "--out", str(out_path),
                    ]
                    port = body.get("port") or state.selected_device.get("port")
                    slave_id = body.get("slave_id") or state.selected_device.get("slave_id")
                    serial = body.get("serial") or state.selected_device.get("serial")
                    if port:
                        cmd.extend(["--port", str(port)])
                    if slave_id:
                        cmd.extend(["--slave-id", str(int(slave_id))])
                    if serial:
                        cmd.extend(["--serial", str(serial)])
                    if recover_stop:
                        cmd.append("--recover-stop")
                    if run_live:
                        cmd.append("--run")

                    def _on_collect_done(job_info):
                        if run_live and (out_path / "manifest.json").is_file():
                            state.last_dataset = str(out_path)
                            return dict(dataset=str(out_path), dry_run=False)
                        return dict(dataset=None, dry_run=not run_live)

                    job = state.start_subprocess_job(
                        "collect", cmd,
                        meta=dict(mode=mode, run=run_live, out=str(out_path)),
                        on_done=_on_collect_done,
                    )
                    self._send_json(dict(ok=True, job=job))
                    return

                if path == "/api/fit":
                    dataset = str(body.get("dataset") or state.last_dataset or "").strip()
                    if not dataset:
                        self._send_json(dict(error="请先指定包含 manifest.json 的数据集目录。"), 400)
                        return
                    out_name = str(body.get("out") or f"outputs/revo3_fit_{time.strftime('%Y%m%d_%H%M%S')}")
                    out_path = Path(out_name)
                    if not out_path.is_absolute():
                        out_path = (state.work_root / out_path).resolve()

                    kp_cmd = float(body.get("kp_cmd", 2.0))
                    kd_cmd = float(body.get("kd_cmd", 0.2))
                    sim_dt = float(body.get("sim_dt", 0.002))
                    cma_gens = int(body.get("cma_gens", 20))
                    lm_nfev = int(body.get("lm_nfev", 12))
                    standalone = bool(body.get("standalone", False))

                    cmd = [
                        sys.executable, "-u", "-m", "revo3_sysid", "fit",
                        "--dataset", dataset,
                        "--out", str(out_path),
                        "--cma-gens", str(cma_gens),
                        "--lm-nfev", str(lm_nfev),
                        "--kp-cmd", str(kp_cmd),
                        "--kd-cmd", str(kd_cmd),
                        "--sim-dt", str(sim_dt),
                    ]
                    if standalone:
                        cmd.append("--standalone")

                    def _on_fit_done(job_info):
                        state.last_fit_out = str(out_path)
                        state.last_export_out = str(out_path)
                        summary = {}
                        sp = out_path / "evaluation_summary.json"
                        if sp.is_file():
                            summary = json.loads(sp.read_text(encoding="utf-8"))
                        plot_path = out_path / "validation_summary.png"
                        return dict(
                            out_dir=str(out_path),
                            summary=summary,
                            plot_path=str(plot_path) if plot_path.is_file() else None,
                            config_json=str(out_path / "revo3_right_identified.json"),
                        )

                    job = state.start_subprocess_job(
                        "fit", cmd,
                        meta=dict(dataset=dataset, out=str(out_path)),
                        on_done=_on_fit_done,
                    )
                    self._send_json(dict(ok=True, job=job))
                    return

                if path == "/api/export":
                    cfg_path = Path(str(body.get("config") or default_pretrained_json())).expanduser().resolve()
                    out_name = str(body.get("out") or f"outputs/revo3_assets_{time.strftime('%Y%m%d_%H%M%S')}")
                    out_path = Path(out_name)
                    if not out_path.is_absolute():
                        out_path = (state.work_root / out_path).resolve()

                    kp_cmd = float(body.get("kp_cmd", 2.0))
                    kd_cmd = float(body.get("kd_cmd", 0.2))
                    sim_dt = float(body.get("sim_dt", 0.002))
                    standalone = bool(body.get("standalone", True))

                    cfg = Revo3ClosedLoopConfig.load(cfg_path)
                    exported = export_all_assets(
                        cfg, out_path,
                        kp_cmd=kp_cmd, kd_cmd=kd_cmd, sim_dt=sim_dt,
                        standalone=standalone,
                    )
                    files_str = {k: str(v) for k, v in exported.items()}
                    state.last_export_out = str(out_path)
                    state.last_export_files = files_str
                    state.log(f"[导出完成] 已导出 MuJoCo + Isaac Lab 资产到 {out_path} (standalone={standalone})", level="ok")
                    self._send_json(dict(
                        ok=True,
                        out_dir=str(out_path),
                        standalone=standalone,
                        files=files_str,
                        zip_url=f"/api/download_zip?dir={urllib.parse.quote(str(out_path))}",
                    ))
                    return

                self._send_json(dict(error="unknown endpoint"), 404)
            except Exception as exc:
                state.log(f"操作失败 ({path}): {exc}", level="error")
                self._send_json(dict(ok=False, error=str(exc)), 500)

    return Handler


def create_gui_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    work_root: Path | None = None,
) -> tuple[ThreadingHTTPServer, GuiAppState]:
    state = GuiAppState(work_root=work_root)
    server = ThreadingHTTPServer((host, int(port)), _make_handler(state))
    server.daemon_threads = True
    return server, state


def serve_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    work_root: Path | None = None,
) -> int:
    server, state = create_gui_server(host=host, port=port, work_root=work_root)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print(f"REVO3-SysID 引导上位机已启动: {url}", flush=True)
    print("按 Ctrl+C 退出上位机服务。", flush=True)
    if open_browser and host in ("127.0.0.1", "localhost"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭上位机...", flush=True)
        state.stop_active_job()
    finally:
        server.server_close()
    return 0
