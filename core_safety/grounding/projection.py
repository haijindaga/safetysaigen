"""Depth projection of image-space safe/unsafe sets into the world frame.

Pinhole model (Eq. 16):
    x = d (u - cx) / fx,   y = d (v - cy) / fy,   z = d
with camera axes: z forward, x right, y down.

Measurements closer than min_range or farther than max_range are discarded
(paper: 3 m / 7 m; near obstacles occlude scene context, far depth degrades).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Camera pose relative to robot base: mounted at `height_m` above ground,
    # looking forward along the robot's heading.
    mount_height: float = 0.5

    # Paper's simulated camera: fx=274.9, fy=376.7, cx=250.0, cy=160.0
    @classmethod
    def paper_sim(cls, mount_height: float = 0.5) -> "PinholeCamera":
        return cls(fx=274.9, fy=376.7, cx=250.0, cy=160.0,
                   width=500, height=320, mount_height=mount_height)


def pixels_to_world(cam: PinholeCamera, depth: np.ndarray, mask: np.ndarray,
                    robot_pose: np.ndarray,
                    min_range: float = 3.0, max_range: float = 7.0,
                    max_height: float | None = None) -> np.ndarray:
    """Project masked pixels with registered depth into world XY.

    robot_pose: (x, y, theta). Returns Nx2 array of world (x, y).

    max_height: if set, discard pixels whose 3D point lies higher than this
    above the ground. Mask pixels on walls/background (e.g. from AROUND
    dilation spilling past an object silhouette) carry the wall's depth and
    would otherwise project as ghost obstacles far behind the object.
    """
    v, u = np.nonzero(mask)
    d = depth[v, u]
    valid = np.isfinite(d) & (d >= min_range) & (d <= max_range)
    u, v, d = u[valid], v[valid], d[valid]
    if max_height is not None and len(u):
        # Height above ground for a horizontal camera: mount_height - y_cam
        # (camera y points down; y_cam = d (v - cy) / fy).
        z_world = cam.mount_height - d * (v - cam.cy) / cam.fy
        keep = z_world <= max_height
        u, v, d = u[keep], v[keep], d[keep]
    if len(u) == 0:
        return np.zeros((0, 2))

    # Camera frame (x right, y down, z forward)
    xc = d * (u - cam.cx) / cam.fx
    zc = d

    # World frame: camera at robot position, z-forward along heading theta.
    x, y, th = robot_pose
    fwd = np.array([np.cos(th), np.sin(th)])
    right = np.array([np.sin(th), -np.cos(th)])
    pts = (np.array([x, y])[None, :]
           + zc[:, None] * fwd[None, :]
           + xc[:, None] * right[None, :])
    return pts


def project_masks_to_world(cam: PinholeCamera, depth: np.ndarray,
                           safe_mask: np.ndarray, unsafe_mask: np.ndarray,
                           robot_pose: np.ndarray,
                           min_range: float = 3.0, max_range: float = 7.0,
                           stride: int = 2,
                           max_height: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Project both partitions; stride subsamples pixels for speed."""
    if stride > 1:
        sub = np.zeros_like(safe_mask)
        sub[::stride, ::stride] = True
        safe_mask = safe_mask & sub
        unsafe_mask = unsafe_mask & sub
    safe_pts = pixels_to_world(cam, depth, safe_mask, robot_pose,
                               min_range, max_range, max_height)
    unsafe_pts = pixels_to_world(cam, depth, unsafe_mask, robot_pose,
                                 min_range, max_range, max_height)
    return safe_pts, unsafe_pts
