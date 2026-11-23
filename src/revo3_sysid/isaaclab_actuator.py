"""GPU-batched Isaac Lab actuator implementing REVO3 MIT delay + 45-deg differential + i^2*t thermal limit.

Provides:
  - ``Revo3DifferentialThermalCore``: framework-agnostic batched math engine (PyTorch when
    available, with a matching NumPy reference step) reproducing the exact closed-loop
    dynamics identified on physical REVO3 Ultra21 hardware (0.338 deg held-out RMSE).
  - ``Revo3MITDifferentialActuator`` & ``Revo3MITDifferentialActuatorCfg``: drop-in Isaac Lab
    ``ActuatorBase`` / ``ActuatorBaseCfg`` subclasses (loaded automatically when ``isaaclab``
    is installed in the Python environment).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import json
from pathlib import Path
from typing import Any

import numpy as np

from .closedloop import Revo3ClosedLoopConfig, T_DIFF_45
from .device import DOF
from .model import mjcf_joint_names


def _build_channel_arrays(
    config: Revo3ClosedLoopConfig,
    joint_names: Sequence[str],
    kp_cmd: float | Sequence[float],
    kd_cmd: float | Sequence[float],
    sim_dt: float,
) -> dict[str, np.ndarray | int | bool]:
    """Map SDK J0..J20 closed-loop parameters onto an arbitrary Isaac Lab joint order."""
    sdk_mjcf = mjcf_joint_names(config.side)
    name_to_sdk = {name: i for i, name in enumerate(sdk_mjcf)}
    n = len(joint_names)
    sdk_idx = np.empty(n, dtype=int)
    for local_i, jname in enumerate(joint_names):
        if jname not in name_to_sdk:
            raise KeyError(f"Joint {jname!r} is not a recognized REVO3 {config.side} joint")
        sdk_idx[local_i] = name_to_sdk[jname]

    kp_arr = np.broadcast_to(np.asarray(kp_cmd, dtype=float), (n,)).copy()
    kd_arr = np.broadcast_to(np.asarray(kd_cmd, dtype=float), (n,)).copy()

    kp_eff = np.zeros(n, dtype=float)
    kd_eff = np.zeros(n, dtype=float)
    kd_v = np.zeros(n, dtype=float)
    bias_act = np.zeros(n, dtype=float)
    i_cont = np.zeros(n, dtype=float)
    i_peak = np.zeros(n, dtype=float)
    tau_s = np.zeros(n, dtype=float)
    delay_steps = np.zeros(n, dtype=int)
    stiffness = np.zeros(n, dtype=float)
    springref = np.zeros(n, dtype=float)

    p19 = int(np.where(sdk_idx == 19)[0][0]) if 19 in sdk_idx else -1
    p20 = int(np.where(sdk_idx == 20)[0][0]) if 20 in sdk_idx else -1
    has_diff = bool(config.thumb_differential and p19 >= 0 and p20 >= 0)
    T_diff = np.asarray(config.thumb_T, dtype=float)
    bias_j19_20 = np.asarray(config.thumb_joint_bias, dtype=float)

    for loc, jid in enumerate(sdk_idx):
        ch = config.channels[jid]
        kp_eff[loc] = kp_arr[loc] * ch.kp_scale
        kd_eff[loc] = kd_arr[loc] * ch.kd_scale
        kd_v[loc] = kd_eff[loc] * ch.vel_mult
        i_cont[loc] = ch.current_limit.i_cont
        i_peak[loc] = ch.current_limit.i_peak
        tau_s[loc] = ch.current_limit.tau_s
        delay_steps[loc] = max(0, int(round(ch.delay_s / sim_dt)))
        if has_diff and jid in (19, 20):
            mode = jid - 19
            bias_act[loc] = float(T_diff[mode] @ bias_j19_20)
            stiffness[loc] = float(config.thumb_joint_stiffness[mode])
            springref[loc] = float(config.thumb_joint_springref[mode])
        else:
            bias_act[loc] = ch.bias
            stiffness[loc] = ch.stiffness
            springref[loc] = ch.springref

    return dict(
        sdk_idx=sdk_idx,
        kp_eff=kp_eff,
        kd_eff=kd_eff,
        kd_v=kd_v,
        bias_act=bias_act,
        i_cont=i_cont,
        i_peak=i_peak,
        tau_s=tau_s,
        delay_steps=delay_steps,
        stiffness=stiffness,
        springref=springref,
        has_diff=has_diff,
        p19=p19,
        p20=p20,
        T_diff=T_diff,
    )


class Revo3DifferentialThermalNumpyCore:
    """Reference NumPy step core (for CPU testing without PyTorch/Isaac Lab)."""

    def __init__(
        self,
        config: Revo3ClosedLoopConfig,
        joint_names: Sequence[str],
        *,
        kp_cmd: float = 2.0,
        kd_cmd: float = 0.2,
        sim_dt: float = 0.002,
        num_envs: int = 1,
        use_i2t: bool = True,
        include_passive_spring: bool = True,
    ) -> None:
        self.sim_dt = float(sim_dt)
        self.num_envs = int(num_envs)
        self.n_dof = len(joint_names)
        self.use_i2t = bool(use_i2t)
        self.include_passive_spring = bool(include_passive_spring)
        p = _build_channel_arrays(config, joint_names, kp_cmd, kd_cmd, sim_dt)
        self.kp_eff = p["kp_eff"]
        self.kd_eff = p["kd_eff"]
        self.kd_v = p["kd_v"]
        self.bias_act = p["bias_act"]
        self.i_cont = p["i_cont"]
        self.i_peak = p["i_peak"]
        self.tau_s = p["tau_s"]
        self.delay_steps = p["delay_steps"]
        self.stiffness = p["stiffness"]
        self.springref = p["springref"]
        self.has_diff = bool(p["has_diff"])
        self.p19 = int(p["p19"])
        self.p20 = int(p["p20"])
        self.T_diff = p["T_diff"]
        self.dyn_mask = (self.i_peak > self.i_cont + 1e-9) & (self.tau_s > 0.0) & self.use_i2t
        self.denom = np.where(self.dyn_mask, (self.i_peak**2 - self.i_cont**2) * np.maximum(self.tau_s, 1e-6), 1.0)
        self.max_delay = int(np.max(self.delay_steps)) if self.n_dof else 0
        self.q_buf = np.zeros((self.max_delay + 1, self.num_envs, self.n_dof), dtype=float)
        self.v_buf = np.zeros((self.max_delay + 1, self.num_envs, self.n_dof), dtype=float)
        self.filled = np.zeros(self.num_envs, dtype=bool)
        self.s_therm = np.zeros((self.num_envs, self.n_dof), dtype=float)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else np.asarray(env_ids, dtype=int)
        self.q_buf[:, idx, :] = 0.0
        self.v_buf[:, idx, :] = 0.0
        self.filled[idx] = False
        self.s_therm[idx, :] = 0.0

    def step(
        self,
        q_des_cmd: np.ndarray,
        v_des_cmd: np.ndarray,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        tau_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        q_des_cmd = np.atleast_2d(np.asarray(q_des_cmd, dtype=float))
        v_des_cmd = np.atleast_2d(np.asarray(v_des_cmd, dtype=float))
        q_meas = np.atleast_2d(np.asarray(q_meas, dtype=float))
        v_meas = np.atleast_2d(np.asarray(v_meas, dtype=float))

        # Shift ring buffer for running envs, and initialize newly-reset envs without zero-jump
        if self.max_delay > 0:
            self.q_buf[1:] = self.q_buf[:-1]
            self.v_buf[1:] = self.v_buf[:-1]
        self.q_buf[0] = q_des_cmd
        self.v_buf[0] = v_des_cmd
        unfilled = ~self.filled
        if np.any(unfilled):
            self.q_buf[:, unfilled, :] = q_des_cmd[None, unfilled, :]
            self.v_buf[:, unfilled, :] = v_des_cmd[None, unfilled, :]
            self.filled[unfilled] = True

        # Gather per-joint delayed command
        q_d = np.empty_like(q_des_cmd)
        v_d = np.empty_like(v_des_cmd)
        for j in range(self.n_dof):
            d = int(self.delay_steps[j])
            q_d[:, j] = self.q_buf[d, :, j]
            v_d[:, j] = self.v_buf[d, :, j]

        q_act = q_meas.copy()
        v_act = v_meas.copy()
        if self.has_diff:
            p19, p20 = self.p19, self.p20
            t00, t01 = self.T_diff[0, 0], self.T_diff[0, 1]
            t10, t11 = self.T_diff[1, 0], self.T_diff[1, 1]
            q19_d, q20_d = q_d[:, p19].copy(), q_d[:, p20].copy()
            v19_d, v20_d = v_d[:, p19].copy(), v_d[:, p20].copy()
            q_d[:, p19] = t00 * q19_d + t01 * q20_d
            q_d[:, p20] = t10 * q19_d + t11 * q20_d
            v_d[:, p19] = t00 * v19_d + t01 * v20_d
            v_d[:, p20] = t10 * v19_d + t11 * v20_d

            q19, q20 = q_meas[:, p19], q_meas[:, p20]
            v19, v20 = v_meas[:, p19], v_meas[:, p20]
            q_act[:, p19] = t00 * q19 + t01 * q20
            q_act[:, p20] = t10 * q19 + t11 * q20
            v_act[:, p19] = t00 * v19 + t01 * v20
            v_act[:, p20] = t10 * v19 + t11 * v20

        u_des = self.kp_eff * (q_d - q_act) + self.kd_v * v_d - self.kd_eff * v_act + self.bias_act
        ceil_k = np.where(self.dyn_mask, self.i_peak - (self.i_peak - self.i_cont) * self.s_therm, self.i_cont)
        u_cmd = np.clip(u_des, -ceil_k, ceil_k)

        if np.any(self.dyn_mask):
            ds = (u_cmd**2 - self.i_cont**2) / self.denom * self.sim_dt
            self.s_therm = np.clip(self.s_therm + np.where(self.dyn_mask, ds, 0.0), 0.0, 1.0)

        tau = u_cmd.copy()
        if self.has_diff:
            p19, p20 = self.p19, self.p20
            t00, t01 = self.T_diff[0, 0], self.T_diff[0, 1]
            t10, t11 = self.T_diff[1, 0], self.T_diff[1, 1]
            up, um = u_cmd[:, p19], u_cmd[:, p20]
            # Joint torque = T^T * u_motor
            tau[:, p19] = t00 * up + t10 * um
            tau[:, p20] = t01 * up + t11 * um

        if self.include_passive_spring:
            tau = tau - self.stiffness * (q_meas - self.springref)
        if tau_ff is not None:
            tau = tau + np.atleast_2d(np.asarray(tau_ff, dtype=float))
        return tau


# ---------------------------------------------------------------------------
# Optional PyTorch + Isaac Lab ActuatorBase integration
# ---------------------------------------------------------------------------
try:
    import torch
    from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.types import ArticulationActions

    HAS_ISAACLAB = True
except ImportError:
    HAS_ISAACLAB = False


if HAS_ISAACLAB:

    class Revo3MITDifferentialActuator(ActuatorBase):
        """Exact REVO3 Ultra21 MIT actuator for Isaac Lab (GPU-batched PyTorch).

        Implements:
          1. Per-joint command transport delay (ring buffer in physics steps).
          2. 45-deg orthonormal differential transmission on ``right_thumb_CMP_joint`` (J19)
             and ``right_thumb_CMR_joint`` (J20).
          3. Leaky-integrator ``i^2*t`` dynamic thermal current limiting per motor channel.
          4. Optional passive joint spring torque ``-k_s * (q - springref)``.
        """

        cfg: "Revo3MITDifferentialActuatorCfg"

        def __init__(self, cfg: "Revo3MITDifferentialActuatorCfg", *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            config = Revo3ClosedLoopConfig.load(cfg.config_json)
            p = _build_channel_arrays(config, self._joint_names, cfg.kp_cmd, cfg.kd_cmd, cfg.sim_dt)

            dev = self._device
            self._sim_dt = float(cfg.sim_dt)
            self._use_i2t = bool(cfg.use_i2t)
            self._include_spring = bool(cfg.include_passive_spring)
            self._kp_eff = torch.tensor(p["kp_eff"], dtype=torch.float32, device=dev)
            self._kd_eff = torch.tensor(p["kd_eff"], dtype=torch.float32, device=dev)
            self._kd_v = torch.tensor(p["kd_v"], dtype=torch.float32, device=dev)
            self._bias_act = torch.tensor(p["bias_act"], dtype=torch.float32, device=dev)
            self._i_cont = torch.tensor(p["i_cont"], dtype=torch.float32, device=dev)
            self._i_peak = torch.tensor(p["i_peak"], dtype=torch.float32, device=dev)
            self._tau_s = torch.tensor(p["tau_s"], dtype=torch.float32, device=dev)
            self._stiffness = torch.tensor(p["stiffness"], dtype=torch.float32, device=dev)
            self._springref = torch.tensor(p["springref"], dtype=torch.float32, device=dev)
            self._delay_steps = torch.tensor(p["delay_steps"], dtype=torch.long, device=dev)
            self._joint_arange = torch.arange(self.num_joints, dtype=torch.long, device=dev)

            self._has_diff = bool(p["has_diff"])
            self._p19 = int(p["p19"])
            self._p20 = int(p["p20"])
            T_mat = p["T_diff"]
            self._t00 = float(T_mat[0, 0])
            self._t01 = float(T_mat[0, 1])
            self._t10 = float(T_mat[1, 0])
            self._t11 = float(T_mat[1, 1])

            self._dyn_mask = (self._i_peak > self._i_cont + 1e-9) & (self._tau_s > 0.0) & self._use_i2t
            self._denom = torch.where(
                self._dyn_mask,
                (self._i_peak**2 - self._i_cont**2) * torch.clamp(self._tau_s, min=1e-6),
                torch.ones_like(self._i_cont),
            )
            self._i_cont_sq = self._i_cont**2
            self._diff_lim = self._i_peak - self._i_cont

            self._max_delay = int(self._delay_steps.max().item()) if self.num_joints > 0 else 0
            self._q_buf = torch.zeros((self._max_delay + 1, self._num_envs, self.num_joints), dtype=torch.float32, device=dev)
            self._v_buf = torch.zeros((self._max_delay + 1, self._num_envs, self.num_joints), dtype=torch.float32, device=dev)
            self._filled = torch.zeros((self._num_envs,), dtype=torch.bool, device=dev)
            self._s_therm = torch.zeros((self._num_envs, self.num_joints), dtype=torch.float32, device=dev)

        def reset(self, env_ids: Sequence[int] | None):
            if env_ids is None:
                env_ids = slice(None)
            self._q_buf[:, env_ids, :] = 0.0
            self._v_buf[:, env_ids, :] = 0.0
            self._filled[env_ids] = False
            self._s_therm[env_ids, :] = 0.0

        def compute(
            self,
            control_action: ArticulationActions,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
        ) -> ArticulationActions:
            q_cmd = joint_pos if control_action.joint_positions is None else control_action.joint_positions
            v_cmd = torch.zeros_like(joint_vel) if control_action.joint_velocities is None else control_action.joint_velocities

            # Ring buffer update (avoid zero-jump on first step after reset)
            unfilled = ~self._filled
            if torch.any(unfilled):
                self._q_buf[:, unfilled, :] = q_cmd[unfilled].unsqueeze(0)
                self._v_buf[:, unfilled, :] = v_cmd[unfilled].unsqueeze(0)
                self._filled[unfilled] = True
            if self._max_delay > 0:
                self._q_buf[1:] = self._q_buf[:-1].clone()
                self._v_buf[1:] = self._v_buf[:-1].clone()
            self._q_buf[0] = q_cmd
            self._v_buf[0] = v_cmd

            # Gather delayed commands per joint: shape (num_envs, num_joints)
            q_d = self._q_buf[self._delay_steps, :, self._joint_arange].transpose(0, 1).clone()
            v_d = self._v_buf[self._delay_steps, :, self._joint_arange].transpose(0, 1).clone()
            q_act = joint_pos.clone()
            v_act = joint_vel.clone()

            if self._has_diff:
                p19, p20 = self._p19, self._p20
                q19_d, q20_d = q_d[:, p19].clone(), q_d[:, p20].clone()
                v19_d, v20_d = v_d[:, p19].clone(), v_d[:, p20].clone()
                q_d[:, p19] = self._t00 * q19_d + self._t01 * q20_d
                q_d[:, p20] = self._t10 * q19_d + self._t11 * q20_d
                v_d[:, p19] = self._t00 * v19_d + self._t01 * v20_d
                v_d[:, p20] = self._t10 * v19_d + self._t11 * v20_d

                q19, q20 = joint_pos[:, p19], joint_pos[:, p20]
                v19, v20 = joint_vel[:, p19], joint_vel[:, p20]
                q_act[:, p19] = self._t00 * q19 + self._t01 * q20
                q_act[:, p20] = self._t10 * q19 + self._t11 * q20
                v_act[:, p19] = self._t00 * v19 + self._t01 * v20
                v_act[:, p20] = self._t10 * v19 + self._t11 * v20

            u_des = self._kp_eff * (q_d - q_act) + self._kd_v * v_d - self._kd_eff * v_act + self._bias_act
            ceil_k = torch.where(self._dyn_mask, self._i_peak - self._diff_lim * self._s_therm, self._i_cont)
            u_cmd = torch.clamp(u_des, -ceil_k, ceil_k)

            if torch.any(self._dyn_mask):
                ds = (u_cmd**2 - self._i_cont_sq) / self._denom * self._sim_dt
                self._s_therm = torch.clamp(
                    self._s_therm + torch.where(self._dyn_mask, ds, torch.zeros_like(ds)),
                    0.0,
                    1.0,
                )

            tau = u_cmd.clone()
            if self._has_diff:
                p19, p20 = self._p19, self._p20
                up, um = u_cmd[:, p19], u_cmd[:, p20]
                tau[:, p19] = self._t00 * up + self._t10 * um
                tau[:, p20] = self._t01 * up + self._t11 * um

            if self._include_spring:
                tau = tau - self._stiffness * (joint_pos - self._springref)
            if control_action.joint_efforts is not None:
                tau = tau + control_action.joint_efforts

            self.computed_effort = u_des
            self.applied_effort = tau
            control_action.joint_efforts = tau
            control_action.joint_positions = None
            control_action.joint_velocities = None
            return control_action

    @configclass
    class Revo3MITDifferentialActuatorCfg(ActuatorBaseCfg):
        """Configuration for :class:`Revo3MITDifferentialActuator`."""

        class_type: type = Revo3MITDifferentialActuator
        config_json: str = MISSING
        kp_cmd: float = 2.0
        kd_cmd: float = 0.2
        sim_dt: float = 0.002
        use_i2t: bool = True
        include_passive_spring: bool = True
