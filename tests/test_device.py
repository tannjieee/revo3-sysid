"""SDK-boundary regression tests. All I/O below is through local doubles."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from aiowuji import revo3_device as r


@pytest.fixture
def device(monkeypatch):
    n = r.DOF
    cfg = NS(joint_min_position_deg=[0.0] * n, joint_max_position_deg=[75.0] * n,
             joint_max_speed_rpm=[110.0] * n, joint_protect_current_ma=[600.0] * n,
             software_stop_enabled=False, teaching_mode_enabled=False)
    hp = NS(safety_state="Unknown", system_state=0, error_code=0, faulted_motor_count=0,
            motor_fault_codes=[2048] * n)
    state = NS(positions_deg=list(range(n)), velocities_rpm=list(range(n)),
               currents_ma=list(range(n)), operating_states=[2048] * n,
               timestamp=NS(sec=3, nsec=4000, clock="ProcessMonotonic"))
    endpoint = NS(serial_number="right", port_name="rs485", slave_id=127,
                  hand_side="Right", model="UltraTouch", firmware_version="0.1.5")
    log = []
    def new_session(**options):
        log.append(("open", options))
        return NS(close=lambda: log.append(("release",)),
                  send_mit=AsyncMock(side_effect=lambda *args: log.append(("mit", *args))),
                  send_current=AsyncMock(side_effect=lambda *args: log.append(("current", *args))),
                  send_position=AsyncMock(side_effect=lambda *args: log.append(("pos", *args))))
    async def stop():
        cfg.software_stop_enabled = True
        log.append(("stop",))
    async def recover():
        cfg.software_stop_enabled = False
        log.append(("recover",))
    motion = NS(get_control_mode=AsyncMock(return_value="Mit"), open_servo=new_session,
                recover_software_stop=recover, software_stop=stop)
    hand = NS(motion=motion, device_info=NS(serial_number="right", model="UltraTouch",
                                            product_code="UT2", hand_side="Right"),
              firmware_info=NS(controller_firmware_version="0.1.5"),
              joint_layout=NS(layout_id="Revo3Ultra21", version=1, joint_count=n),
              state=NS(snapshot=AsyncMock(return_value=state)), config=NS(snapshot=AsyncMock(return_value=cfg)),
              health=NS(snapshot=AsyncMock(return_value=hp)), close=AsyncMock())
    mgr = NS(discover=AsyncMock(return_value=[endpoint]), connect=AsyncMock(return_value=hand),
             close=AsyncMock())
    sdk = NS(Manager=Mock(return_value=mgr), ProtocolType=NS(Modbus="Modbus"),
             Rs485Baudrate=NS(Baud5Mbps=5000000))
    monkeypatch.setattr(r, "_sdk", lambda: sdk)
    return NS(mgr=mgr, hand=hand, endpoint=endpoint, log=log, cfg=cfg, hp=hp, state=state,
              sdk=sdk, create=lambda **kwargs: r.Revo3Device(port="rs485", **kwargs))


def test_discovery_and_status_no_motion(device):
    async def run():
        assert (await r.discover(port="rs485", slave_id=127))[0]["serial"] == "right"
        device.mgr.discover.assert_awaited_once_with(scan_all=True, port="rs485", slave_id=127,
                                                     protocol="Modbus", modbus_baudrate=5000000, broadcast=False)
        d = device.create(serial="right")
        device.cfg.software_stop_enabled = True
        device.hp.error_code = 12
        info = await d.connect()
        assert info["layout"] == "Revo3Ultra21"
        assert (await d.health())["error_code"] == 12
        state = await d.read()
        np.testing.assert_allclose(state["qpos"], np.deg2rad(np.arange(21)))
        np.testing.assert_allclose(state["qvel"], np.arange(21) * np.pi / 30)
        np.testing.assert_allclose(state["effort"], np.arange(21) / 1000)
        assert state["timestamp_us"] == 3000004
        await d.close()
        assert device.log == []
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["serial", "layout", "ambiguous", "broadcast"])
def test_refuse_wrong_connection(device, reason):
    async def run():
        if reason == "layout":
            device.hand.joint_layout.version = 2
        elif reason == "ambiguous":
            device.mgr.discover.return_value = [device.endpoint] * 2
        elif reason == "serial":
            device.hand.device_info.serial_number = "other"
        d = device.create(serial="right", slave_id=0 if reason == "broadcast" else 127)
        with pytest.raises((ValueError, RuntimeError)):
            await d.connect()
        assert d.hand is d.manager is None
        assert device.log == []
    asyncio.run(run())


def test_legacy_mit_converts_commands_and_primes_while_stopped(device):
    async def run():
        d = device.create(legacy_mit=True)
        device.cfg.software_stop_enabled = True
        device.state.positions_deg[0] = -0.57
        await d.connect()
        zeros = np.zeros(21)
        kp, kd = np.full(21, 0.5), np.full(21, 0.1)
        with pytest.raises(RuntimeError, match="latched"):
            await d.arm("impedance", kp=kp, kd=kd)
        assert not device.log
        q = await d.arm("impedance", kp=kp, kd=kd, recover_stop=True)
        assert (await d.read())["qpos"][0] < 0 and q[0] == 0
        assert [e[0] for e in device.log] == ["open", "mit", "release", "recover", "open", "mit"]
        assert np.allclose(device.log[1][1], np.arange(21))
        await d.write("impedance", q=q, qd=np.full(21, np.pi / 30), kp=kp, kd=kd, effort=0.03+zeros)
        np.testing.assert_allclose(device.log[-1][2:], [np.ones(21), kp, kd, np.full(21, 30)])
        with pytest.raises(ValueError, match="position"):
            await d.write("impedance", q=q-0.01, qd=zeros, kp=kp, kd=kd, effort=zeros)
        with pytest.raises(ValueError, match="velocity"):
            await d.write("impedance", q=q, qd=12+zeros, kp=kp, kd=kd, effort=zeros)
        with pytest.raises(ValueError, match="current"):
            await d.write("impedance", q=q, qd=zeros, kp=kp, kd=kd, effort=zeros+0.61)
        await d.close()
        assert [e[0] for e in device.log[-2:]] == ["release", "stop"]
        assert device.cfg.software_stop_enabled
        device.hand.motion.get_control_mode.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["state", "health", "teaching", "pose", "speed", "legacy", "direct"])
def test_no_output_on_failed_preflight(device, fault):
    async def run():
        d = device.create()
        await d.connect()
        if fault == "state":
            device.state.operating_states[0] |= 1
        elif fault == "health":
            device.hp.motor_fault_codes[0] |= 1
        elif fault == "teaching":
            device.cfg.teaching_mode_enabled = True
        elif fault == "pose":
            device.state.positions_deg[0] = -1.6
        elif fault == "speed":
            device.state.velocities_rpm[0] = 111
        mode = {"legacy": "impedance", "direct": "torque"}.get(fault, "position")
        with pytest.raises(RuntimeError):
            await d.arm(mode)
        assert not device.log
        await d.close()
    asyncio.run(run())


def test_legacy_current_uses_zero_gain_mit(device):
    async def run():
        d = device.create(legacy_mit=True)
        await d.connect()
        await d.arm("torque")
        zeros = np.zeros(21)
        first = device.log[-1]
        assert first[0] == "mit"
        np.testing.assert_allclose(first[1], np.arange(21))
        np.testing.assert_array_equal(first[2:], [zeros] * 4)
        current = np.linspace(-0.1, 0.1, 21)
        await d.write("torque", q=None, qd=None, kp=None, kd=None, effort=current)
        command = device.log[-1]
        assert command[0] == "mit"
        np.testing.assert_allclose(command[1], first[1])
        np.testing.assert_array_equal(command[2:5], [zeros] * 3)
        np.testing.assert_allclose(command[5], current * 1000)
        with pytest.raises(ValueError, match="current"):
            await d.write("torque", q=None, qd=None, kp=None, kd=None, effort=zeros + 0.61)
        await d.close()
        device.hand.motion.get_control_mode.assert_not_called()
    asyncio.run(run())


def test_modern_mode_is_independent_of_command_type(device):
    async def run():
        device.hand.firmware_info.controller_firmware_version = "1.0.0"
        d = device.create()
        await d.connect()
        device.hand.motion.get_control_mode.return_value = "Position"
        with pytest.raises(RuntimeError, match="select Mit"):
            await d.arm("torque")
        assert not device.log
        device.hand.motion.get_control_mode.return_value = "Mit"
        await d.arm("torque")
        await d.write("torque", q=None, qd=None, kp=None, kd=None, effort=np.full(21, -0.025))
        assert device.log[-1] == ("current", [-25.0] * 21)
        device.hand.motion.get_control_mode.return_value = "Damping"
        with pytest.raises(RuntimeError):
            await d.check_health()
        await d.close()
        device.hand.close.assert_awaited_once()
        device.mgr.close.assert_awaited_once()
    asyncio.run(run())


def test_failed_stop_readback_surfaces_and_closes_handles(device):
    async def run():
        d = device.create()
        await d.connect()
        await d.arm("position")
        device.hand.motion.software_stop = AsyncMock()  # ack with no latch
        with pytest.raises(RuntimeError, match="could not be confirmed"):
            await d.close()
        assert d.hand is d.manager is None
        device.hand.close.assert_awaited_once()
        device.mgr.close.assert_awaited_once()
    asyncio.run(run())


def test_runtime_health_skips_config_when_requested(device):
    async def run():
        d = device.create(legacy_mit=True)
        await d.connect()
        await d.arm("impedance", kp=np.ones(r.DOF), kd=np.full(r.DOF, 0.1))
        cfg_before = device.hand.config.snapshot.await_count
        hp_before = device.hand.health.snapshot.await_count
        await d.check_runtime_health(include_config=False)
        assert device.hand.health.snapshot.await_count == hp_before + 1
        assert device.hand.config.snapshot.await_count == cfg_before
        device.hp.error_code = 7
        with pytest.raises(RuntimeError, match="health check failed"):
            await d.check_runtime_health(include_config=False)
        device.hp.error_code = 0
        await d.check_runtime_health(include_config=True)
        assert device.hand.config.snapshot.await_count == cfg_before + 1
        await d.close()
    asyncio.run(run())
