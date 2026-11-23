"""End-to-end HTTP API tests for the zero-dependency REVO3 guided Web GUI server."""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import zipfile

from revo3_sysid.gui_server import create_gui_server


def _http_get(url: str) -> tuple[int, bytes, dict]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read(), dict(resp.headers)


def _http_post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_gui_server_demo_dryrun_export_zip_and_security(tmp_path: Path):
    server, state = create_gui_server(host="127.0.0.1", port=0, work_root=tmp_path)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Index HTML loads cleanly
        code, raw, _ = _http_get(f"{base}/")
        assert code == 200
        html = raw.decode("utf-8")
        assert "REVO3-SysID" in html and "E-STOP" in html and "Step 5" in html

        # 2. State endpoint
        code, raw, _ = _http_get(f"{base}/api/state")
        assert code == 200
        st = json.loads(raw.decode("utf-8"))
        assert Path(st["xml_path"]).is_file()
        assert Path(st["usd_path"]).is_file()
        assert len(st["models"]) >= 1

        # 3. Demo endpoint returns waveforms & fast rollout timing
        code, demo = _http_post(f"{base}/api/demo", {})
        assert code == 200 and demo["ok"] is True
        assert demo["rollout_ms"] > 0
        assert len(demo["waveforms"]["t_s"]) == 100
        assert len(demo["waveforms"]["j19_sim_deg"]) == 100

        # 4. Dry-run collect subprocess completes cleanly without touching hardware
        code, col = _http_post(f"{base}/api/capture", {"mode": "quick", "run": False, "out": "captures/dry_test"})
        assert code == 200 and col["ok"] is True
        deadline = time.time() + 15.0
        final_job = None
        while time.time() < deadline:
            _, raw, _ = _http_get(f"{base}/api/state")
            job = json.loads(raw.decode("utf-8"))["active_job"]
            if job and job["status"] != "running":
                final_job = job
                break
            time.sleep(0.15)
        assert final_job is not None and final_job["status"] == "succeeded"
        assert final_job["result"]["dry_run"] is True

        # 5. Export MuJoCo + Isaac Lab standalone assets and verify ZIP download
        code, exp = _http_post(
            f"{base}/api/export",
            {"out": "outputs/gui_exp_test", "kp_cmd": 2.0, "kd_cmd": 0.2, "sim_dt": 0.002, "standalone": True},
        )
        assert code == 200 and exp["ok"] is True
        assert Path(exp["files"]["mujoco_xml"]).is_file()
        assert Path(exp["files"]["isaaclab_usda"]).is_file()
        assert Path(exp["files"]["isaaclab_cfg_py"]).is_file()

        code, zip_bytes, headers = _http_get(f"{base}{exp['zip_url']}")
        assert code == 200
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = set(zf.namelist())
            assert "revo3_right_identified.xml" in names
            assert "revo3_right_identified.usda" in names
            assert "isaaclab_revo3_cfg.py" in names
            assert "revo3_right.usd" in names

        # 6. Path traversal outside repo/work_root must be blocked
        try:
            _http_get(f"{base}/api/file?path=/etc/passwd")
            assert False, "Expected 404 for /etc/passwd"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()
