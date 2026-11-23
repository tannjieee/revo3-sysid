"""MuJoCo kinematics and torque-driven dynamics for BrainCo's Revo3 Ultra MJCF.

Supply the external MJCF with its original mesh directory structure. All
joint vectors use SDK order J0–J20 and SI units. The model's joint limits and
collision meshes describe the model, independently of the connected hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_FINGERS = ("little", "ring", "middle", "index", "thumb")


def mjcf_joint_names(handedness: str = "right") -> tuple[str, ...]:
    """SDK J0–J20 mapped to the anatomical joint names in BrainCo's MJCF."""
    if handedness not in ("right", "left"):
        raise ValueError("handedness must be 'right' or 'left'")
    names = []
    for finger in _FINGERS:
        joints = ("MCP", "PIP", "DIP", "CMP", "CMR") if finger == "thumb" else ("MPR", "MCP", "PIP", "DIP")
        names.extend(f"{handedness}_{finger}_{joint}_joint" for joint in joints)
    return tuple(names)


def _vector(values, label):
    values = np.asarray(values, dtype=float)
    if values.shape != (21,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain 21 finite values in SDK joint order")
    return values


class Revo3HandModel:
    """Fixed-base, 21-hinge hand model with joint vectors in SDK order."""

    def __init__(self, xml_path, handedness: str = "right") -> None:
        import mujoco

        names = mjcf_joint_names(handedness)
        self._mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve(strict=True)))
        self.data = mujoco.MjData(m)
        ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
        if m.njnt != 21 or (ids < 0).any() or (m.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE).any():
            raise ValueError(f"expected a fixed-base 21-joint Revo3 {handedness} hand MJCF")
        if not m.jnt_limited[ids].all():
            raise ValueError("the Revo3 model must define ranges for every joint")
        self._qadr = m.jnt_qposadr[ids]
        self._dadr = m.jnt_dofadr[ids]
        self.limits = m.jnt_range[ids].copy()
        # A welded pad/tip belongs to its closest ancestor that has a joint.
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{handedness}_hand_base_link")
        if palm < 0:
            raise ValueError(f"missing {handedness}_hand_base_link in Revo3 model")
        part_by_body = {palm: "palm"}
        for joint, name in zip(ids, names):
            part_by_body[int(m.jnt_bodyid[joint])] = name.split("_")[1]
        self.geoms = {part: [] for part in (*_FINGERS, "palm")}
        for geom in np.flatnonzero(m.geom_contype | m.geom_conaffinity):
            body = m.geom_bodyid[geom]
            while body and body not in part_by_body:
                body = m.body_parentid[body]
            if body in part_by_body:
                self.geoms[part_by_body[body]].append(int(geom))
        if not all(self.geoms.values()):
            raise ValueError("expected collision geometry for the palm and every Revo3 finger")
        mujoco.mj_forward(m, self.data)

    @property
    def q(self) -> np.ndarray:
        """Current joint positions in radians, J0–J20."""
        return self.data.qpos[self._qadr].copy()

    def pose(self, q) -> None:
        self.data.qpos[self._qadr] = _vector(q, "joint position")
        self._mj.mj_kinematics(self.model, self.data)

    def clearance(self, finger: str, reference, margin: float) -> "Revo3Clearance":
        """Check the finger against the palm and the other fingers (metres)."""
        finger = "little" if finger == "pinky" else finger
        if finger not in _FINGERS:
            raise ValueError(f"unknown Revo3 finger {finger!r}")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("margin must be a finite nonnegative distance")
        self.pose(reference)
        return Revo3Clearance(self, finger, float(margin))


class Revo3Clearance:
    """Reference-relative clearance, using BrainCo's individual convex pieces.

    Adjacent knuckle/palm geometry already near in the reference pose can
    share a mechanical joint. Other pairs must keep the margin; a pair
    starting closer must retain at least half of its reference separation.
    """

    def __init__(self, hand: Revo3HandModel, finger: str, margin: float) -> None:
        self.hand = hand
        self.cutoff = margin + 0.01
        others = np.array([g for part, geoms in hand.geoms.items() if part != finger for g in geoms])
        self.a = np.repeat(hand.geoms[finger], len(others))
        self.b = np.tile(others, len(hand.geoms[finger]))
        m = hand.model
        self.radius = m.geom_rbound[self.a] + m.geom_rbound[self.b]
        self._fromto = np.zeros(6)
        self._distance = hand._mj.mj_geomDistance
        gaps = np.full(len(self.a), self.cutoff)
        for i in np.flatnonzero(self._lower_bounds() <= self.cutoff):
            gaps[i] = self._gap(i)
        weld_a = m.body_weldid[m.geom_bodyid[self.a]]
        parent_weld = m.body_weldid[m.body_parentid[weld_a]]
        weld_b = m.body_weldid[m.geom_bodyid[self.b]]
        attached_to_palm = np.isin(self.b, hand.geoms["palm"]) & (parent_weld == weld_b)
        keep = ~(attached_to_palm & (gaps < margin))
        self.a, self.b, self.radius = self.a[keep], self.b[keep], self.radius[keep]
        gaps = gaps[keep]
        self.floors = np.where(gaps < margin, np.maximum(0, 0.5 * gaps), margin)

    def _lower_bounds(self) -> np.ndarray:
        centres = self.hand.data.geom_xpos
        delta = centres[self.a] - centres[self.b]
        return np.sqrt(np.einsum("ij,ij->i", delta, delta)) - self.radius

    def _gap(self, i: int) -> float:
        return self._distance(self.hand.model, self.hand.data, self.a[i], self.b[i], self.cutoff, self._fromto)

    def slack(self, q) -> float:
        """Smallest capped gap minus its allowed minimum (metres)."""
        self.hand.pose(q)
        bounds = self._lower_bounds() - self.floors
        # Pairs beyond the cutoff return the cutoff from MuJoCo. Bound each
        # exact query by its bounding spheres and the best gap seen so far.
        best = self.cutoff - self.floors.max()
        candidates = np.flatnonzero(bounds < best)
        for i in candidates[np.argsort(bounds[candidates])]:
            if bounds[i] >= best:
                break
            best = min(best, self._gap(i) - self.floors[i])
        return float(best)


class Revo3Sim(Revo3HandModel):
    """Step BrainCo's passive MJCF under externally specified joint torques."""

    def __init__(self, xml_path, handedness: str = "right") -> None:
        super().__init__(xml_path, handedness)
        if self.model.nu:
            raise ValueError("Revo3Sim requires the passive Revo3 model without actuators")

    @property
    def qd(self) -> np.ndarray:
        """Current joint velocities in rad/s, J0–J20."""
        return self.data.qvel[self._dadr].copy()

    def reset(self, q=None) -> None:
        positions = None if q is None else _vector(q, "joint position").copy()
        self._mj.mj_resetData(self.model, self.data)
        if positions is not None:
            self.data.qpos[self._qadr] = positions
        self._mj.mj_forward(self.model, self.data)

    def step(self, torque_nm, nstep: int = 1) -> None:
        """Apply the 21 joint torques in N·m for nstep model timesteps."""
        if not isinstance(nstep, (int, np.integer)) or nstep < 1:
            raise ValueError("nstep must be a positive integer")
        self.data.qfrc_applied[self._dadr] = _vector(torque_nm, "joint torque")
        self._mj.mj_step(self.model, self.data, nstep=int(nstep))
