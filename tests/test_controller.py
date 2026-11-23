"""Run the same controller lifecycle, scheduling and recording against an async fake device."""

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import numpy as np

from revo3_sysid.controller import DOF, Revo3Controller
from revo3_sysid.device import Revo3Device


class FakeDevice:
    def __init__(self):
        self.legacy_mit = False
        self.info = {"serial": "TEST-21", "side": "right", "firmware": "0.1.8"}
        self.limits = np.tile([-1.0, 1.0], (DOF, 1))
        self.current_limits = np.full(DOF, 0.2)
        self.speed_limits = np.full(DOF, 5.0)
        self.operating_states = np.zeros(DOF, dtype=int)
        self.q = np.linspace(-0.2, 0.2, DOF)
        self.seq = 0
        self.writes = []
        self.calls = []
        self.armed = self.fault = self.hang = self.fail_close = self.frozen = False

    def check_feedback(self, state):
        return Revo3Device.check_feedback(self, state)

    async def connect(self):
        self.calls.append("connect")

    async def read(self):
        if self.hang:
            await asyncio.Event().wait()
        if not self.frozen:
            self.seq += 1
        return dict(qpos=self.q.copy(), qvel=np.zeros(DOF), effort=np.zeros(DOF),
                    timestamp_us=self.seq * 1000, timestamp_clock="test",
                    operating_states=self.operating_states.copy())

    async def check_health(self):
        if self.fault:
            raise RuntimeError("motor fault")

    async def arm(self, mode, **kwargs):
        self.armed = True
        self.calls.append(("arm", mode, kwargs))
        await self.check_health()
        hold = np.clip(self.q, self.limits[:, 0], self.limits[:, 1])
        return hold

    async def write(self, mode, **command):
        self.writes.append((mode, {k: v.copy() for k, v in command.items()}))
        if mode != "torque":
            self.q = command["q"].copy()

    async def close(self):
        if self.armed:
            self.calls.append("software_stop")
        self.armed = False
        self.calls.append("close")
        if self.fail_close:
            raise RuntimeError("stop unconfirmed")


class Revo3Tests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = FakeDevice()
        factory = patch("revo3_sysid.controller.Revo3Device", return_value=self.device)
        factory.start()
        self.addCleanup(factory.stop)

    async def test_position_trajectory_and_recording(self):
        c = Revo3Controller(rate_hz=100)
        await c.start()
        try:
            np.testing.assert_array_equal(c.initial_qpos, self.device.q)
            self.assertEqual(c.device_info["serial"], "TEST-21")
            c.start_recording()
            target = c.initial_qpos.copy()
            target[20] += 0.015
            await c.move(target)
            ticks = c.stop_recording()
            self.assertGreater(len(ticks["q"]), 2)
            self.assertEqual(ticks["q"].shape[1], 21)
            np.testing.assert_array_equal(ticks["q_cmd"][-1], target)
            self.assertTrue(ticks["command_sent"].all())
            self.assertTrue((ticks["send_return_ns"] >= ticks["send_started_ns"]).all())
            self.assertTrue((ticks["send_started_ns"] >= ticks["state_host_ns"]).all())
            self.assertTrue((ticks["state_host_ns"] >= ticks["tick_ns"]).all())
            self.assertTrue(np.isnan(ticks["qd_cmd"]).all())
            self.assertTrue(np.isnan(ticks["effort_cmd"]).all())
            self.assertEqual(self.device.writes[-1][0], "position")
            with self.assertRaises(ValueError):
                c.set_now("qd_desired", np.zeros(21))
            with self.assertRaises(ValueError):
                c.set_now("q_desired", np.zeros(20))
            with self.assertRaises(ValueError):
                c.set_now("q_desired", np.full(21, 1.1))
            with self.assertRaises(RuntimeError):
                c.switch("torque")
        finally:
            await c.stop()
        self.assertEqual(self.device.calls[-2:], ["software_stop", "close"])
        await c.stop()
        self.assertEqual(self.device.calls.count("close"), 1)

    async def test_impedance_and_current_are_distinct(self):
        for mode, field in (("impedance", "effort_ff"), ("torque", "torque")):
            with self.subTest(mode=mode):
                c = Revo3Controller(rate_hz=100)
                c.switch(mode)
                await c.start()
                try:
                    with self.assertRaisesRegex(ValueError, "current limit"):
                        c.set_now(field, np.full(21, 0.21))
                    c.start_recording()
                    if mode == "impedance":
                        c.kp = 1.5
                        c.kd = 0.1
                        c.set_now("qd_desired", np.full(21, 0.15))
                    await c.set(field, np.full(21, 0.12))
                    await c.wait(0.02)
                    ticks = c.stop_recording()
                    np.testing.assert_allclose(ticks["effort_cmd"][-1], 0.12)
                    if mode == "impedance":
                        np.testing.assert_allclose(ticks["kp"][-1], 1.5)
                        np.testing.assert_allclose(ticks["qd_cmd"][-1], 0.15)
                    else:
                        self.assertTrue(np.isnan(ticks["q_cmd"]).all())
                        self.assertTrue(np.isnan(ticks["kp"]).all())
                        with self.assertRaises(ValueError):
                            await c.move()
                finally:
                    await c.stop()

    async def test_dry_run_reads_and_never_writes_even_when_recover_requested(self):
        c = Revo3Controller(dry_run=True, recover_stop=True)
        await c.start()
        c.start_recording()
        await c.set("q_desired", np.zeros(DOF))
        await c.wait(0.03)
        ticks = c.stop_recording()
        await c.stop()
        self.assertGreater(len(ticks["q"]), 0)
        self.assertFalse(ticks["command_sent"].any())
        self.assertTrue(np.isnan(ticks["q_cmd"]).all())
        self.assertFalse(self.device.writes)
        self.assertEqual(self.device.calls, ["connect", "close"])

    async def test_fault_stops_and_releases_before_user_calls_stop(self):
        c = Revo3Controller()
        await c.start()
        self.device.fault = True
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            await c.wait(1.0)
        self.assertEqual(self.device.calls[-2:], ["software_stop", "close"])
        await c.stop()

    async def test_state_read_timeout_and_failed_stop_reported(self):
        c = Revo3Controller(rate_hz=100, stale_ms=80)
        await c.start()
        self.device.hang = self.device.fail_close = True
        with self.assertRaisesRegex(RuntimeError, "stop unconfirmed"):
            await c.move(np.zeros(DOF))
        with self.assertRaisesRegex(RuntimeError, "stop unconfirmed"):
            await c.stop()

    async def test_invalid_initial_pose_cleans_up_without_motion(self):
        c = Revo3Controller()
        self.device.q[19] = 4.0
        with self.assertRaisesRegex(RuntimeError, "outside|limit"):
            await c.start()
        self.assertEqual(self.device.calls, ["connect", "close"])
        self.assertEqual(self.device.writes, [])
        self.device.q[:] = 0
        await c.start()
        await c.stop()

    async def test_repeated_snapshot_is_detected_even_when_reads_succeed(self):
        c = Revo3Controller(rate_hz=100, stale_ms=60)
        await c.start()
        self.device.frozen = True
        with self.assertRaisesRegex(RuntimeError, "timestamp stopped advancing"):
            await c.wait(1.0)
        await c.stop()

    async def test_replacing_an_inflight_move_resets_its_velocity(self):
        c = Revo3Controller(mode="impedance", rate_hz=100)
        await c.start()
        try:
            target = c.initial_qpos + 0.2
            motion = asyncio.create_task(c.move(target))
            await c.wait(0.05)
            c.set_now("q_desired", c.initial_qpos)
            with self.assertRaisesRegex(RuntimeError, "superseded"):
                await motion
            await c.wait(0.03)
            np.testing.assert_array_equal(self.device.writes[-1][1]["qd"], np.zeros(DOF))
        finally:
            await c.stop()

    async def test_feedback_tolerance_only_applies_to_measured_pose(self):
        c = Revo3Controller(rate_hz=100)
        self.device.q[0] = 1 + np.deg2rad(0.6)
        await c.start()
        try:
            self.assertGreater(c.initial_qpos[0], 1)
            self.assertEqual(c.initial_target_qpos[0], 1)
            with self.assertRaisesRegex(ValueError, "outside device limits"):
                c.set_now("q_desired", c.initial_qpos)
            await c.wait(0.03)
            self.assertEqual(self.device.writes[-1][1]["q"][0], 1)
            self.device.q[0] = 1 + np.deg2rad(2)
            with self.assertRaisesRegex(RuntimeError, "outside|limit"):
                await c.wait(0.2)
        finally:
            await c.stop()

    async def test_new_fault_bits_prevent_the_next_write(self):
        c = Revo3Controller()
        await c.start()
        await c.wait(0.03)
        sent = len(self.device.writes)
        self.device.operating_states[4] = 1
        with self.assertRaisesRegex(RuntimeError, "fault"):
            await c.wait(0.2)
        self.assertEqual(len(self.device.writes), sent)
        await c.stop()

    async def test_stream_trajectory_tick_locked_and_ref_index(self):
        c = Revo3Controller(mode="impedance", rate_hz=100, kp=2.0, kd=0.2)
        await c.start()
        try:
            q0 = c.initial_target_qpos.copy()
            n = 15
            q_ref = np.tile(q0, (n, 1))
            q_ref[:, 19] += np.linspace(0.0, 0.05, n)
            qd_ref = np.zeros_like(q_ref)
            qd_ref[:, 19] = 0.25
            seen_k = []
            c.start_recording(seconds=5.0)
            await c.stream_trajectory(q_ref, qd_ref, on_tick=lambda k, st: seen_k.append(k))
            ticks = c.stop_recording()
            self.assertEqual(seen_k, list(range(n)))
            stream_mask = ticks["ref_index"] >= 0
            np.testing.assert_array_equal(ticks["ref_index"][stream_mask], np.arange(n))
            np.testing.assert_allclose(ticks["q_cmd"][stream_mask], q_ref)
            np.testing.assert_allclose(ticks["qd_cmd"][stream_mask], qd_ref)
        finally:
            await c.stop()

    async def test_max_feedback_current_guard_stops_before_write(self):
        c = Revo3Controller(mode="impedance", rate_hz=100, max_feedback_current=0.30)
        await c.start()
        try:
            await c.wait(0.03)
            sent_before = len(self.device.writes)
            # Inject high feedback current in read()
            orig_read = self.device.read
            async def high_curr_read():
                st = await orig_read()
                st["effort"] = np.full(DOF, 0.35)
                return st
            self.device.read = high_curr_read
            with self.assertRaisesRegex(RuntimeError, "feedback \\|current\\|"):
                await c.wait(0.2)
            self.assertEqual(len(self.device.writes), sent_before)
            self.assertGreaterEqual(c.stats["max_feedback_current"], 0.35)
        finally:
            await c.stop()
