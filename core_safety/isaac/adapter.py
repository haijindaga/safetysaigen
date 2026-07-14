"""Adapters between Isaac Sim and the core pipeline.

Pure-Python glue (no Isaac imports here): converts poses, camera
intrinsics, and semantic-segmentation frames into the interfaces the
pipeline already consumes. The Isaac-specific scene/robot setup lives in
scripts/isaac_mobile_demo.py.
"""
from __future__ import annotations

import numpy as np

from ..grounding.projection import PinholeCamera


def quat_to_yaw(q: np.ndarray) -> float:
    """Yaw from a (w, x, y, z) quaternion (Isaac core convention)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class IsaacPlanarBase:
    """Planar control-affine view of an Isaac mobile base.

    Satisfies the interface pipeline.safe_control expects (state, dt, f, g,
    clip_input); the actual actuation is done by the caller (differential
    controller / locomotion policy), this object only mirrors the pose.
    """

    def __init__(self, dt: float, v_max: float = 0.3, omega_max: float = 1.0,
                 holonomic: bool = False):
        self.state = np.zeros(3)
        self.dt = dt
        self.v_max = v_max
        self.omega_max = omega_max
        self.holonomic = holonomic

    def update_pose(self, position: np.ndarray, orientation_wxyz: np.ndarray):
        self.state = np.array([position[0], position[1],
                               quat_to_yaw(np.asarray(orientation_wxyz))])

    def f(self) -> np.ndarray:
        return np.zeros(3)

    def g(self) -> np.ndarray:
        c, s = np.cos(self.state[2]), np.sin(self.state[2])
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).copy()
        if not self.holonomic:
            u[1] = 0.0
        speed = np.hypot(u[0], u[1])
        if speed > self.v_max:
            u[:2] *= self.v_max / speed
        u[2] = np.clip(u[2], -self.omega_max, self.omega_max)
        return u


def camera_from_intrinsics(K: np.ndarray, width: int, height: int,
                           mount_height: float) -> PinholeCamera:
    """Build the pipeline's pinhole model from an Isaac 3x3 intrinsics matrix."""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    return PinholeCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                         width=width, height=height, mount_height=mount_height)


def parse_semantic_frame(sem_frame: dict) -> tuple[np.ndarray, dict[int, str]]:
    """Convert an Isaac semantic_segmentation frame into (label_img, id->name).

    Isaac returns {"data": HxW uint32, "info": {"idToLabels": {id: labels}}}
    where labels is either {"class": "cone"} or a plain string depending on
    version. Unlabeled ids map to their id-string and are simply never
    requested by the segmenter.
    """
    data = np.asarray(sem_frame["data"]).squeeze()
    id_to_labels = sem_frame["info"]["idToLabels"]
    id_to_name: dict[int, str] = {}
    for key, val in id_to_labels.items():
        idx = int(key)
        if isinstance(val, dict):
            name = val.get("class") or next(iter(val.values()), str(idx))
        else:
            name = str(val)
        id_to_name[idx] = str(name).lower()
    return data.astype(np.int32), id_to_name
