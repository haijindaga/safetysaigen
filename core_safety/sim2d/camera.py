"""Raycasting pinhole camera: renders RGB, depth, and GT semantic labels.

Camera frame: z forward, x right, y down; mounted at `mount_height` above
ground, looking along the robot heading. Depth is z-depth (distance along
the optical axis), matching the pinhole projection in grounding/projection.
"""
from __future__ import annotations

import numpy as np

from ..grounding.projection import PinholeCamera
from .world import World, Cylinder, Box

# Deterministic pseudo-colors for RGB rendering (class id -> color).
def _class_color(i: int) -> np.ndarray:
    rng = np.random.default_rng(i * 7919)
    return rng.integers(40, 220, size=3).astype(np.uint8)


class RaycastCamera:
    def __init__(self, cam: PinholeCamera | None = None,
                 max_render_range: float = 30.0):
        self.cam = cam or PinholeCamera.paper_sim()
        self.max_range = max_render_range

    def render(self, world: World, robot_pose: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb HxWx3 uint8, depth HxW float [z-depth], labels HxW int)."""
        cam = self.cam
        H, W = cam.height, cam.width
        x0, y0, th = robot_pose
        cz = cam.mount_height

        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)              # HxW
        xc = (uu - cam.cx) / cam.fx             # right component per unit z
        yc = (vv - cam.cy) / cam.fy             # down component per unit z

        # World-frame ray directions per unit forward distance t (= z-depth):
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        right = np.array([np.sin(th), -np.cos(th), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        dir_x = fwd[0] + xc * right[0]
        dir_y = fwd[1] + xc * right[1]
        dir_z = -yc                              # down*yc has z = -yc

        depth = np.full((H, W), np.inf)
        labels = np.zeros((H, W), dtype=np.int32)
        ids = world.class_ids()

        # ---- ground plane: cz + t*dir_z = 0  =>  t = cz / yc (yc > 0) -----
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ground = np.where(yc > 1e-6, cz / yc, np.inf)
        hit = t_ground < self.max_range
        t_g = np.where(hit, t_ground, 0.0)
        gx = x0 + t_g * dir_x
        gy = y0 + t_g * dir_y
        g_ids = world.ground_class_at(gx, gy)
        depth[hit] = t_ground[hit]
        labels[hit] = g_ids[hit]

        # ---- objects -------------------------------------------------------
        for obj in world.objects:
            if isinstance(obj, Cylinder):
                t_hit = self._ray_cylinder(x0, y0, cz, dir_x, dir_y, dir_z, obj)
            elif isinstance(obj, Box):
                t_hit = self._ray_box(x0, y0, cz, dir_x, dir_y, dir_z, obj)
            else:
                continue
            closer = t_hit < depth
            depth[closer] = t_hit[closer]
            labels[closer] = ids[obj.cls]

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[..., 2] = 200  # sky-ish background
        for name, i in ids.items():
            rgb[labels == i] = _class_color(i)
        depth_out = np.where(np.isfinite(depth), depth, np.inf)
        return rgb, depth_out, labels

    @staticmethod
    def _ray_cylinder(x0, y0, cz, dx, dy, dz, cyl: Cylinder) -> np.ndarray:
        """Smallest positive t where ray hits the finite vertical cylinder."""
        ox, oy = x0 - cyl.cx, y0 - cyl.cy
        a = dx ** 2 + dy ** 2
        b = 2 * (ox * dx + oy * dy)
        c = ox ** 2 + oy ** 2 - cyl.radius ** 2
        disc = b ** 2 - 4 * a * c
        t = np.full(dx.shape, np.inf)
        ok = (disc >= 0) & (a > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = np.where(ok, (-b - sq) / (2 * np.maximum(a, 1e-12)), 0.0)
        z1 = cz + t1 * dz
        valid = ok & (t1 > 1e-6) & (z1 >= 0.0) & (z1 <= cyl.height)
        t1 = np.where(valid, t1, np.inf)
        t[valid] = t1[valid]
        return t

    @staticmethod
    def _ray_box(x0, y0, cz, dx, dy, dz, box: Box) -> np.ndarray:
        """Slab-method intersection with an axis-aligned box on the ground."""
        t_near = np.full(dx.shape, -np.inf)
        t_far = np.full(dx.shape, np.inf)
        for o, d, lo, hi in ((x0, dx, box.x_min, box.x_max),
                             (y0, dy, box.y_min, box.y_max),
                             (cz, dz, 0.0, box.height)):
            d_safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
            t1 = (lo - o) / d_safe
            t2 = (hi - o) / d_safe
            t_lo = np.minimum(t1, t2)
            t_hi = np.maximum(t1, t2)
            # Parallel ray outside the slab: no hit.
            parallel_out = (np.abs(d) < 1e-12) & ((o < lo) | (o > hi))
            t_lo = np.where(parallel_out, np.inf, t_lo)
            t_near = np.maximum(t_near, t_lo)
            t_far = np.minimum(t_far, t_hi)
        hit = (t_near <= t_far) & (t_near > 1e-6)
        return np.where(hit, t_near, np.inf)
