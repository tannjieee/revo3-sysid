"""Async joint control for the BrainCo Revo3 Ultra21 in native SDK J0–J20 order."""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np

from .device import DOF, JOINT_NAMES, Revo3Device


def _vector(value, name, *, scalar=False):
    arr = np.asarray(value, dtype=float)
    if scalar and arr.ndim == 0:
        arr = np.full(DOF, float(arr))
    if arr.shape != (DOF,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain {DOF} finite values")
    return arr.copy()


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class _Recording:
    """Allocate the same per-tick columns as the Wuji recorder, with send provenance."""

    def __init__(self, capacity):
        self.n = 0
        self.columns = {
            name: np.empty(capacity, dtype=np.int64)
            for name in ("tick_ns", "scheduled_ns", "seq", "state_host_ns", "timestamp_us",
                         "send_started_ns", "send_return_ns", "send_ns", "ref_index")
        }
        self.columns["command_sent"] = np.empty(capacity, dtype=bool)
        self.columns["timestamp_clock"] = np.empty(capacity, dtype="U64")
        self.columns.update({
            name: np.full((capacity, DOF), np.nan)
            for name in ("q_cmd", "qd_cmd", "effort_cmd", "kp", "kd", "q", "qd", "effort")
        })

    def append(self, state, tick, scheduled, started, ended, mode, command, legacy_current, ref_index=-1):
        if self.n == len(self.columns["tick_ns"]):
            return
        row = dict(tick_ns=tick, scheduled_ns=scheduled, seq=state["seq"],
                   state_host_ns=state["host_ns"], timestamp_us=state["timestamp_us"],
                   timestamp_clock=str(state["timestamp_clock"]),
                   send_started_ns=started, send_return_ns=ended, send_ns=ended - started,
                   ref_index=int(ref_index),
                   command_sent=bool(started), q=state["qpos"], qd=state["qvel"], effort=state["effort"])
        if started:
            if mode != "torque":
                row["q_cmd"] = command["q"]
            if mode == "impedance":
                row.update(qd_cmd=command["qd"], kp=command["kp"], kd=command["kd"])
            if legacy_current:
                row.update(q_cmd=command["q"], qd_cmd=0.0, kp=0.0, kd=0.0)
            if mode != "position":
                row["effort_cmd"] = command["effort"]
        for name, value in row.items():
            self.columns[name][self.n] = value
        self.n += 1


class Revo3Controller:
    """Single-event-loop servo controller. Positions are radians, velocities rad/s, currents A.

    Select the mode before start. MIT kp/kd are SDK-native gains. effort_limit bounds the
    direct/FF current commands, not the firmware's closed-loop output. A dry run reads live
    feedback. It performs no servo, gain, mode, stop, or recovery write.
    """

    dof = DOF
    joint_names = JOINT_NAMES
    _targets = {"position": ("q_desired",), "impedance": ("q_desired", "qd_desired", "effort_ff"),
                "torque": ("torque",)}

    def __init__(self, port=None, *, slave_id=127, serial=None, mode="position", legacy_mit=False,
                 recover_stop=False, kp=1.0, kd=0.05, effort_limit=0.25, rate_hz=50.0,
                 dry_run=False, max_tracking_error=None, max_feedback_current=None,
                 stale_ms=250.0, health_interval_s=0.25, config_check_every=8):
        self._device = Revo3Device(port=port, slave_id=slave_id, serial=serial, legacy_mit=legacy_mit)
        self._dry_run = bool(dry_run)
        self._recover_stop = bool(recover_stop)
        self._effort_limit = float(effort_limit)
        if not math.isfinite(self._effort_limit) or self._effort_limit < 0:
            raise ValueError("effort_limit must be finite and nonnegative (A)")
        self._rate_hz = _positive(rate_hz, "rate_hz")
        self._timeout = _positive(stale_ms, "stale_ms") / 1000
        if 1 / self.rate_hz >= self._timeout:
            raise ValueError("stale_ms must exceed the control period")
        self._health_interval_ns = round(_positive(health_interval_s, "health_interval_s") * 1e9)
        if type(config_check_every) is not int or config_check_every < 1:
            raise ValueError("config_check_every must be a positive integer")
        self._config_check_every = config_check_every
        self.max_tracking_error = (None if max_tracking_error is None else
                                   _positive(max_tracking_error, "max_tracking_error"))
        self.max_feedback_current = (None if max_feedback_current is None else
                                     _positive(max_feedback_current, "max_feedback_current"))
        self._lifecycle = asyncio.Lock()
        self._stop_requested = asyncio.Event()
        self._task = None
        self._motion = None
        self._stream = None
        self._recording = None
        self._closing = False
        self._cleanup_error = None
        self._seq = 0
        self._last_fresh_ns = 0
        self.error = None
        self.initial_target_qpos = None
        self.initial_qpos = None
        self.state = None
        self.stats = {}
        self._last_set = {}
        self.set_freq(self.rate_hz)
        self.switch(mode)
        self.set_gains(kp, kd)

    @property
    def device_info(self):
        return dict(self._device.info)

    @property
    def joint_limits(self):
        """Device-reported (21, 2) array in radians, available after connect."""
        return self._device.limits.copy()

    @property
    def mode(self):
        return self._mode

    @property
    def rate_hz(self):
        return self._rate_hz

    @property
    def dry_run(self):
        return self._dry_run

    @property
    def effort_limit(self):
        return self._effort_limit

    @property
    def kp(self):
        return self._kp.copy()

    @kp.setter
    def kp(self, value):
        self.set_gains(kp=value)

    @property
    def kd(self):
        return self._kd.copy()

    @kd.setter
    def kd(self, value):
        self.set_gains(kd=value)

    def set_gains(self, kp=None, kd=None):
        """Update the gain arrays included in the next MIT frame."""
        prop = _vector(kp, "kp", scalar=True) if kp is not None else self._kp
        deriv = _vector(kd, "kd", scalar=True) if kd is not None else self._kd
        if np.any(prop < 0) or np.any(deriv < 0):
            raise ValueError("MIT gains must be nonnegative")
        self._kp, self._kd = prop, deriv

    def switch(self, mode):
        if mode not in self._targets:
            raise ValueError(f"unknown Revo3 mode: {mode!r}")
        if self._task is not None or self._lifecycle.locked():
            if self._mode != mode:
                raise RuntimeError("select the mode before start()")
        self._mode = mode

    def set_freq(self, freq):
        self._update_freq = _positive(freq, "freq")

    def _check_running(self):
        if self.error is not None:
            raise RuntimeError(f"Revo3 controller stopped: {self.error}")
        if self._task is None or self._task.done() or self._closing:
            raise RuntimeError("Revo3 controller is not running")

    def _check_position(self, q):
        limits = self._device.limits
        outside = np.flatnonzero((q < limits[:, 0]) | (q > limits[:, 1]))
        if outside.size:
            names = ", ".join(JOINT_NAMES[i] for i in outside)
            raise ValueError(f"position outside device limits: {names}")

    async def _read(self):
        state = await asyncio.wait_for(self._device.read(), self._timeout)
        self._seq += 1
        received = time.monotonic_ns()
        stamp = state["timestamp_us"]  # SDK receive clock, independent of host_ns
        if self.state is None or stamp != self.state["timestamp_us"]:
            self._last_fresh_ns = received
        elif received - self._last_fresh_ns >= self._timeout * 1e9:
            raise RuntimeError("SDK state timestamp stopped advancing")
        state = dict(state, host_ns=received, seq=self._seq)
        for name in ("qpos", "qvel", "effort"):
            state[name] = _vector(state[name], name)
        self.state = state
        return state

    async def start(self):
        async with self._lifecycle:
            if self._task is not None:
                raise RuntimeError("controller already started; call stop() before restarting")
            self.error = self._cleanup_error = self.initial_qpos = self.initial_target_qpos = self.state = None
            self._closing = False
            self._stop_requested.clear()
            self._seq = 0
            self._last_set.clear()
            # Per-loop counters exclude the measured-pose startup priming frame(s).
            self.stats = {"ticks": 0, "sends": 0, "overruns": 0, "max_tracking_error": 0.0,
                          "max_feedback_current": 0.0}
            try:
                await self._device.connect()
                state = await self._read()
                self.initial_qpos = state["qpos"].copy()
                if self.dry_run:
                    limits = self._device.limits
                    hold = np.clip(self.initial_qpos, limits[:, 0], limits[:, 1])
                else:
                    self._device.check_feedback(state)
                    hold = await self._device.arm(self.mode, timeout_ms=math.ceil(self._timeout * 1000),
                                                  recover_stop=self._recover_stop, kp=self._kp, kd=self._kd)
                self.initial_target_qpos = hold.copy()
                self._command = dict(q=hold.copy(), qd=np.zeros(DOF), effort=np.zeros(DOF))
                self._task = asyncio.create_task(self._run(), name="revo3-sysid-ctrl")
            except BaseException:
                await self._device.close()
                raise

    async def stop(self):
        async with self._lifecycle:
            task = self._task
            if task is None:
                return
            self._closing = True
            self._stop_requested.set()
            await asyncio.shield(task)
            self._task = None
            if self._cleanup_error is not None:
                raise RuntimeError(f"Revo3 stop/close failed: {self._cleanup_error}") from self._cleanup_error

    def _frame(self):
        # All updates replace arrays, so the frame stays consistent across the async write.
        return dict(self._command, kp=self._kp, kd=self._kd)

    async def _write(self, command):
        await asyncio.wait_for(self._device.write(self.mode, **command), self._timeout)

    def _end_motion(self, completed=False):
        if self._motion is not None:
            done = self._motion[-1]
            self._motion = None
            if not done.done():
                done.set_result(completed)
        if self._stream is not None:
            done = self._stream["done"]
            self._stream = None
            if not done.done():
                done.set_result(completed)

    def set_now(self, attr, value):
        self._check_running()
        if attr not in self._targets[self.mode]:
            raise ValueError(f"{attr!r} is not a target in {self.mode} mode")
        target = _vector(value, attr)
        if attr == "q_desired":
            self._check_position(target)
        if attr == "qd_desired" and np.any(np.abs(target) > self._device.speed_limits):
            raise ValueError("qd_desired exceeds device velocity limits (rad/s)")
        if attr in ("effort_ff", "torque"):
            if np.any(np.abs(target) > np.minimum(self.effort_limit, self._device.current_limits)):
                raise ValueError(f"{attr} exceeds current limit (A)")
        if (self._motion is not None or self._stream is not None) and attr in ("q_desired", "qd_desired"):
            self._end_motion()
            self._command["qd"] = np.zeros(DOF)
        field = {"q_desired": "q", "qd_desired": "qd", "effort_ff": "effort", "torque": "effort"}[attr]
        self._command[field] = target

    async def set(self, attr, value):
        """Stage a target, then pace consecutive updates of this field."""
        self.set_now(attr, value)
        now = time.monotonic()
        deadline = max(now, self._last_set.get(attr, now)) + 1 / self._update_freq
        self._last_set[attr] = deadline
        await self.wait(deadline - now)

    async def wait(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        deadline = time.monotonic() + seconds
        while True:
            self._check_running()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.02))

    async def move(self, qpos=None, *, max_velocity=1.0, max_acceleration=4.0, max_jerk=40.0):
        """Stream a Ruckig trajectory. Return after its final frame is accepted by the SDK."""
        from ruckig import InputParameter, Result, Ruckig, Trajectory

        self._check_running()
        if self.mode == "torque":
            raise ValueError("move() needs position or impedance mode")
        goal = np.zeros(DOF) if qpos is None else _vector(qpos, "qpos")
        self._check_position(goal)
        inp = InputParameter(DOF)
        inp.current_position = self._command["q"].tolist()
        inp.current_velocity = self._command["qd"].tolist()
        inp.target_position = goal.tolist()
        for name, value in (("max_velocity", max_velocity), ("max_acceleration", max_acceleration),
                            ("max_jerk", max_jerk)):
            limit = _vector(value, name, scalar=True)
            if name == "max_velocity":
                limit = np.minimum(limit, self._device.speed_limits)
            if np.any(limit <= 0):
                raise ValueError(f"{name} must be positive")
            setattr(inp, name, limit.tolist())
        trajectory = Trajectory(DOF)
        result = Ruckig(DOF).calculate(inp, trajectory)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig failed: {result}")
        self._end_motion()
        done = asyncio.get_running_loop().create_future()
        motion = (time.monotonic_ns(), trajectory, goal, done)
        self._motion = motion
        try:
            completed = await done
            self._check_running()
            if not completed:
                raise RuntimeError("move was superseded")
        finally:
            if self._motion is motion:
                self._end_motion()
                self._command["qd"] = np.zeros(DOF)

    async def stream_trajectory(self, q_ref, qd_ref=None, *, on_tick=None):
        """Stream a pre-planned (N, 21) trajectory locked 1-to-1 to servo loop ticks.

        Each tick atomically stages (q_ref[k], qd_ref[k]) inside _run() right before _write(),
        eliminating dual-loop beat/frame skips and recording exact ref_index = k.
        Optional on_tick(k, state) runs synchronously inside _run() before staging frame k;
        raising inside on_tick aborts and triggers safe software stop.
        """
        self._check_running()
        if self.mode == "torque":
            raise ValueError("stream_trajectory() needs position or impedance mode")
        q_arr = np.asarray(q_ref, dtype=float)
        if q_arr.ndim != 2 or q_arr.shape[1] != DOF or len(q_arr) == 0 or not np.isfinite(q_arr).all():
            raise ValueError(f"q_ref must be a nonempty finite (N, {DOF}) array")
        limits = self._device.limits
        if np.any(q_arr < limits[None, :, 0]) or np.any(q_arr > limits[None, :, 1]):
            raise ValueError("q_ref contains positions outside device limits")
        if qd_ref is None:
            qd_arr = np.zeros_like(q_arr)
        else:
            qd_arr = np.asarray(qd_ref, dtype=float)
            if qd_arr.shape != q_arr.shape or not np.isfinite(qd_arr).all():
                raise ValueError("qd_ref must match q_ref shape and be finite")
            if self.mode == "impedance" and np.any(np.abs(qd_arr) > self._device.speed_limits[None, :]):
                raise ValueError("qd_ref exceeds device velocity limits (rad/s)")
        self._end_motion()
        done = asyncio.get_running_loop().create_future()
        stream = dict(q=q_arr.copy(), qd=qd_arr.copy(), cursor=0, on_tick=on_tick, done=done)
        self._stream = stream
        try:
            completed = await done
            self._check_running()
            if not completed:
                raise RuntimeError("stream_trajectory was superseded")
        finally:
            if self._stream is stream:
                self._end_motion()
                self._command["qd"] = np.zeros(DOF)

    def start_recording(self, seconds=60.0):
        self._check_running()
        self._recording = _Recording(math.ceil(_positive(seconds, "seconds") * self.rate_hz) + 1)

    def stop_recording(self):
        recording, self._recording = self._recording, None
        if recording is None:
            raise RuntimeError("start_recording() was not called")
        return {name: values[:recording.n] for name, values in recording.columns.items()}

    def _sample(self, now, state):
        if self._stream is not None:
            st = self._stream
            k = st["cursor"]
            if st["on_tick"] is not None:
                st["on_tick"](k, state)
            q = st["q"][k]
            qd = st["qd"][k] if self.mode == "impedance" else np.zeros(DOF)
            self._command.update(q=q, qd=qd)
            st["cursor"] = k + 1
            final = st if st["cursor"] >= len(st["q"]) else None
            return final, k
        if self._motion is None:
            return None, -1
        started, trajectory, goal, _ = self._motion
        elapsed = (now - started) / 1e9
        if elapsed >= trajectory.duration:
            self._command.update(q=goal, qd=np.zeros(DOF))
            return self._motion, -1
        q, qd, _ = trajectory.at_time(elapsed)
        q = np.asarray(q)
        self._check_position(q)
        self._command.update(q=q, qd=np.asarray(qd))
        return None, -1

    async def _run(self):
        period = max(1, round(1e9 / self.rate_hz))
        scheduled = previous = time.monotonic_ns()
        health_due = 0
        health_count = 0
        try:
            while not self._closing:
                tick = time.monotonic_ns()
                recording = self._recording
                if not self.dry_run and tick - previous >= self._timeout * 1e9:
                    raise RuntimeError("servo scheduling gap exceeded stale_ms")
                previous = tick
                if not self.dry_run and tick >= health_due:
                    include_cfg = (health_count % self._config_check_every) == 0
                    if hasattr(self._device, "check_runtime_health"):
                        await asyncio.wait_for(
                            self._device.check_runtime_health(include_config=include_cfg), self._timeout
                        )
                    else:
                        await asyncio.wait_for(self._device.check_health(), self._timeout)
                    health_count += 1
                    health_due = tick + self._health_interval_ns
                state = await self._read()
                if self._closing:
                    break
                final, ref_idx = self._sample(time.monotonic_ns(), state)
                command = self._frame()
                started = ended = 0
                if not self.dry_run:
                    self._device.check_feedback(state)
                    curr_max = float(np.max(np.abs(state["effort"])))
                    self.stats["max_feedback_current"] = max(self.stats["max_feedback_current"], curr_max)
                    if self.max_feedback_current is not None and curr_max > self.max_feedback_current:
                        raise RuntimeError(f"feedback |current| {curr_max:.3f} A exceeds limit")
                    if self.mode != "torque":
                        error = float(np.max(abs(command["q"] - state["qpos"])))
                        self.stats["max_tracking_error"] = max(self.stats["max_tracking_error"], error)
                        if self.max_tracking_error is not None and error > self.max_tracking_error:
                            raise RuntimeError(f"tracking error {error:.3f} rad exceeds limit")
                    started = time.monotonic_ns()
                    if started - tick >= self._timeout * 1e9:
                        raise RuntimeError("feedback cycle exceeded stale_ms before send")
                    await self._write(command)
                    ended = time.monotonic_ns()
                    self.stats["sends"] += 1
                if recording is not None and self._recording is recording:
                    legacy_current = self.mode == "torque" and self._device.legacy_mit
                    recording.append(state, tick, scheduled, started, ended, self.mode, command, legacy_current,
                                     ref_index=ref_idx)
                if final is not None and (final is self._motion or final is self._stream):
                    self._end_motion(completed=True)
                self.stats["ticks"] += 1
                scheduled += period
                now = time.monotonic_ns()
                if now > scheduled:
                    self.stats["overruns"] += 1
                    if now - scheduled > period:
                        scheduled = now
                sleep_s = (scheduled - now) / 1e9
                if sleep_s > 0:
                    try:
                        await asyncio.wait_for(self._stop_requested.wait(), sleep_s)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0)
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
        finally:
            self._closing = True
            self._end_motion()
            try:
                # The device owns any partially armed servo and confirms its stop before close.
                await self._device.close()
            except Exception as exc:
                self._cleanup_error = exc
                self.error = f"{self.error + '; ' if self.error else ''}stop/close failed: {exc}"
