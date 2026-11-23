"""BrainCo Revo3 Ultra21 v1, RS-485 at 5 Mbps. All public vectors use SDK J0–J20."""

from __future__ import annotations

import asyncio

import numpy as np


JOINT_NAMES = tuple(
    f"{finger}_{joint}"
    for finger in ("little", "ring", "middle", "index")
    for joint in ("abd", "mcp", "pip", "dip")
) + ("thumb_cmc_flex", "thumb_mcp", "thumb_ip", "thumb_cmc_abd", "thumb_cmc_rotation")
DOF = len(JOINT_NAMES)
FEEDBACK_TOLERANCE_RAD = np.deg2rad(1.5)
SDK_VERSION = "2.3.2"


def _sdk():
    try:
        from bc_revo3_sdk import main_mod
    except ImportError as exc:
        raise RuntimeError('Install the REVO3 support with pip install -e ".[hardware]"') from exc
    if main_mod.get_sdk_version() != SDK_VERSION:
        raise RuntimeError(f"REVO3 support requires bc-revo3-sdk=={SDK_VERSION}")
    return main_mod


def _options(sdk, port, slave_id):
    if slave_id is not None and (type(slave_id) is not int or not 1 <= slave_id <= 247):
        raise ValueError("slave_id must be a Modbus unicast address (1–247)")
    return dict(port=port, slave_id=slave_id, protocol=sdk.ProtocolType.Modbus,
                modbus_baudrate=sdk.Rs485Baudrate.Baud5Mbps, broadcast=False)


def _label(value):
    return None if value is None else str(value).rsplit(".", 1)[-1]


def _vector(values, label):
    arr = np.asarray(values, dtype=float)
    if arr.shape != (DOF,) or not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain {DOF} finite SDK joint values")
    return arr


async def discover(port=None, slave_id=None):
    """List responding unicast Modbus endpoints without opening a hand or issuing motion."""
    sdk = _sdk()
    options = _options(sdk, port, slave_id)
    manager = sdk.Manager()
    try:
        devices = await asyncio.wait_for(manager.discover(scan_all=True, **options), 60)
        return [dict(port=d.port_name, slave_id=d.slave_id, serial=d.serial_number,
                     side=_label(d.hand_side), model=_label(d.model), firmware=d.firmware_version)
                for d in devices]
    finally:
        await asyncio.wait_for(manager.close(), 3)


class Revo3Device:
    """An async, single-owner device connection. connect/read/health are read-only."""

    def __init__(self, port=None, slave_id=127, serial=None, legacy_mit=False):
        self.port, self.slave_id, self.serial = port, slave_id, serial
        self.legacy_mit = legacy_mit
        self.manager = self.hand = self._session = None
        self._mode = None
        self.info = {}

    async def connect(self):
        if self.manager is not None:
            raise RuntimeError("Revo3 is already connected")
        sdk = _sdk()
        options = _options(sdk, self.port, self.slave_id)
        self.manager = sdk.Manager()
        try:
            matches = await asyncio.wait_for(self.manager.discover(scan_all=True, **options), 60)
            if self.serial is not None:
                matches = [d for d in matches if d.serial_number == self.serial]
            if not matches:
                raise RuntimeError("No matching REVO3 found on the selected port, slave ID and serial")
            if len(matches) != 1:
                raise RuntimeError("Multiple REVO3 endpoints match; specify the port, slave ID or serial")
            endpoint = matches[0]
            self.hand = await asyncio.wait_for(self.manager.connect(endpoint), 15)
            info, layout = self.hand.device_info, self.hand.joint_layout
            if info is None or not info.serial_number:
                raise RuntimeError("REVO3 device identity is unavailable")
            if ((self.serial is not None and info.serial_number != self.serial)
                    or (endpoint.serial_number and info.serial_number != endpoint.serial_number)):
                raise RuntimeError("REVO3 serial changed between discovery and connection")
            if layout is None or (layout.layout_id, layout.version, layout.joint_count) != ("Revo3Ultra21", 1, DOF):
                raise RuntimeError("Expected the Revo3Ultra21 v1 joint layout")
            self.info = dict(serial=info.serial_number, port=endpoint.port_name, slave_id=endpoint.slave_id,
                             side=_label(info.hand_side), model=_label(info.model), product_code=info.product_code,
                             firmware=self.hand.firmware_info.controller_firmware_version,
                             layout=layout.layout_id, sdk_version=SDK_VERSION)
            cfg = await asyncio.wait_for(self.hand.config.snapshot(), 2)
            self._update_limits(cfg)
            return self.info.copy()
        except BaseException:
            await self.close()
            raise

    def _update_limits(self, cfg):
        lo = _vector(cfg.joint_min_position_deg, "joint_min_position_deg")
        hi = _vector(cfg.joint_max_position_deg, "joint_max_position_deg")
        speed = _vector(cfg.joint_max_speed_rpm, "joint_max_speed_rpm")
        current = _vector(cfg.joint_protect_current_ma, "joint_protect_current_ma")
        if np.any(lo >= hi) or np.any(speed < 0) or np.any(current < 0):
            raise ValueError("Invalid joint limits returned by REVO3")
        limits = np.deg2rad(np.column_stack((lo, hi)))
        speed_limits = speed * (np.pi / 30)
        current_limits = current / 1000
        if self._mode is not None and any(
            not np.array_equal(old, new) for old, new in zip(
                (self.limits, self.speed_limits, self.current_limits), (limits, speed_limits, current_limits))):
            raise RuntimeError("REVO3 joint limits changed during control")
        self.limits, self.speed_limits, self.current_limits = limits, speed_limits, current_limits
        for values in (limits, speed_limits, current_limits):
            values.setflags(write=False)

    async def read(self):
        state = await asyncio.wait_for(self.hand.state.snapshot(), 2)
        stamp = state.timestamp
        return dict(qpos=np.deg2rad(_vector(state.positions_deg, "positions_deg")),
                    qvel=_vector(state.velocities_rpm, "velocities_rpm") * (np.pi / 30),
                    effort=_vector(state.currents_ma, "currents_ma") / 1000,
                    operating_states=_vector(state.operating_states, "operating_states").astype(int),
                    timestamp_us=stamp.sec * 1_000_000 + stamp.nsec // 1000,
                    timestamp_clock=_label(stamp.clock))

    async def health(self, *, include_config=True):
        """Return faults and stop state for inspection, including on a stopped hand."""
        health = await asyncio.wait_for(self.hand.health.snapshot(), 2)
        out = dict(safety_state=_label(health.safety_state), system_state=health.system_state,
                   error_code=health.error_code, faulted_motor_count=health.faulted_motor_count,
                   motor_fault_codes=_vector(health.motor_fault_codes, "motor_fault_codes").astype(int).tolist())
        if include_config:
            cfg = await asyncio.wait_for(self.hand.config.snapshot(), 2)
            self._update_limits(cfg)
            out.update(software_stop_enabled=cfg.software_stop_enabled,
                       teaching_mode_enabled=cfg.teaching_mode_enabled)
        return out

    def _validate_health(self, data, *, allow_stopped=False):
        if (data["system_state"] or data["error_code"] or data["faulted_motor_count"]
                or any(code & 0x3F for code in data["motor_fault_codes"])
                or data["safety_state"] not in ("Operational", "Unknown")
                or data.get("teaching_mode_enabled", False)):
            raise RuntimeError(f"REVO3 health check failed: {data}")
        if data.get("software_stop_enabled", False) and not allow_stopped:
            raise RuntimeError("REVO3 software stop is latched")

    async def _check_mode(self, mode):
        if mode not in ("position", "impedance", "torque"):
            raise ValueError(f"Unsupported REVO3 control mode: {mode}")
        if self.info["firmware"] == "0.1.5":
            if self.info["product_code"] != "UT2":
                raise RuntimeError("Legacy 0.1.5 motion is supported only on the UT2")
            if mode in ("impedance", "torque") and self.legacy_mit:
                return
            if mode == "position" and not self.legacy_mit:
                return
            raise RuntimeError("Use position, or select legacy_mit for impedance/current on UT2 firmware 0.1.5")
        if self.legacy_mit:
            raise RuntimeError("legacy_mit requires UT2 firmware 0.1.5")
        # Whole-hand mode (register 900) is distinct from the Servo packet type.
        # The SDK's control_mode example requires Mit before Servo on supporting firmware.
        actual = _label(await asyncio.wait_for(self.hand.motion.get_control_mode(), 2))
        if actual != "Mit":
            raise RuntimeError(f"REVO3 whole-hand mode is {actual}; select Mit before starting Servo")

    async def check_health(self, *, include_config=True):
        data = await self.health(include_config=include_config)
        self._validate_health(data)
        if self._mode is not None and include_config:
            await self._check_mode(self._mode)

    async def check_runtime_health(self, *, include_config=False):
        """Lightweight cyclic health check for the real-time servo loop."""
        await self.check_health(include_config=include_config)

    def check_feedback(self, state):
        if np.any(state["operating_states"] & 0x3F):
            raise RuntimeError("REVO3 motor fault in state feedback")
        q = state["qpos"]
        lo, hi = self.limits.T
        if np.any(q < lo - FEEDBACK_TOLERANCE_RAD) or np.any(q > hi + FEEDBACK_TOLERANCE_RAD):
            raise RuntimeError("Measured REVO3 pose outside device limits by more than 1.5 degrees")
        if np.any(np.abs(state["qvel"]) > self.speed_limits):
            raise RuntimeError("Measured REVO3 velocity exceeded device limit")

    async def arm(self, mode, *, timeout_ms=250, recover_stop=False, kp=None, kd=None):
        """Start with measured pose (or zero direct current); return the primed pose in rad."""
        if self._mode is not None:
            raise RuntimeError("REVO3 is already armed")
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        data = await self.health()
        self._validate_health(data, allow_stopped=recover_stop)
        await self._check_mode(mode)
        state = await self.read()
        self.check_feedback(state)
        lo, hi = self.limits.T
        target = np.clip(state["qpos"], lo, hi)
        if mode == "torque" and self.legacy_mit:
            self._zero_gain_pose_deg = np.rad2deg(target).tolist()
        zeros = np.zeros(DOF)
        # Acquire ownership before marking the connection as one we must stop.
        self._session = self.hand.motion.open_servo(command_timeout_ms=timeout_ms)
        self._mode = mode
        try:
            await self.write(mode, q=target, qd=zeros, kp=kp, kd=kd, effort=zeros)
            if data["software_stop_enabled"]:
                self._session.close()
                self._session = None
                await asyncio.wait_for(self.hand.motion.recover_software_stop(), 2)
                await self.check_health()
                self._session = self.hand.motion.open_servo(command_timeout_ms=timeout_ms)
                await self.write(mode, q=target, qd=zeros, kp=kp, kd=kd, effort=zeros)
            return target
        except BaseException:
            await self.stop()
            raise

    async def write(self, mode, *, q, qd, kp, kd, effort):
        if mode != self._mode or self._session is None:
            raise RuntimeError("REVO3 servo session is not active in the requested mode")
        if mode != "torque":
            q = _vector(q, "q")
            if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                raise ValueError("REVO3 target position outside device limits")
            pos = np.rad2deg(q).tolist()
        if mode == "position":
            operation = self._session.send_position(pos)
        else:
            amps = _vector(effort, "effort")
            if np.any(np.abs(amps) > self.current_limits):
                raise ValueError("REVO3 current command exceeds device joint protection limit")
            ma = (amps * 1000).tolist()
            if mode == "torque":
                if self.legacy_mit:
                    # Legacy MIT encodes current in the feedforward field. Keep
                    # the position field valid even with both feedback gains zero.
                    zeros = [0.0] * DOF
                    operation = self._session.send_mit(self._zero_gain_pose_deg, zeros, zeros, zeros, ma)
                else:
                    operation = self._session.send_current(ma)
            else:
                qd = _vector(qd, "qd")
                kp, kd = _vector(kp, "kp"), _vector(kd, "kd")
                if np.any(np.abs(qd) > self.speed_limits):
                    raise ValueError("REVO3 target velocity outside device limits")
                if np.any(kp < 0) or np.any(kd < 0):
                    raise ValueError("MIT gains must be nonnegative")
                rpm = (qd * (30 / np.pi)).tolist()
                operation = self._session.send_mit(pos, rpm, kp.tolist(), kd.tolist(), ma)
        await asyncio.wait_for(operation, 2)

    async def stop(self):
        """Release Servo ownership, latch software stop, and verify the device's readback."""
        if self._mode is None:
            return
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self._session = None
            await asyncio.wait_for(self.hand.motion.software_stop(), 2)
            cfg = await asyncio.wait_for(self.hand.config.snapshot(), 2)
            if not cfg.software_stop_enabled:
                raise RuntimeError("REVO3 software stop could not be confirmed")
            self._mode = None

    async def close(self):
        try:
            await self.stop()
        finally:
            try:
                if self.hand is not None:
                    await asyncio.wait_for(self.hand.close(), 3)
            finally:
                self.hand = self._mode = None
                manager, self.manager = self.manager, None
                if manager is not None:
                    await asyncio.wait_for(manager.close(), 3)
