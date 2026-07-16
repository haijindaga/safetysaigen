"""2D costmap with three kinds of "unsafe" kept apart.

  - OCCUPANCY (n_unsafe): physical points confirmed by depth. Depth is the
    ground truth here: it decays fast, so if depth stops seeing something
    there, the cell clears within a couple of cycles ("if depth says empty,
    it cannot stay red").
  - ZONE (zone_until): regions the VLM *declared* unsafe (AROUND / BETWEEN
    buffers, prohibited surfaces). These sit on visibly empty floor by
    design, so depth must NOT clear them — they expire only when the VLM
    stops re-asserting them (TTL) .
  - UNKNOWN: cells with no observations at all (neither safe nor unsafe
    evidence). Treated as traversable for the barrier (Assumption 1) but
    distinguishable for display / frontier finding.

n_safe accumulates floor/traversable evidence as in the paper (Eq. 3).
"""
from __future__ import annotations

import time

import numpy as np


class SemanticCostmap:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float = 0.2, tau: float = 0.5,
                 occ_thresh: float = 1.0, zone_ttl: float = 45.0):
        self.res = resolution
        self.tau = tau
        self.occ_thresh = occ_thresh
        self.zone_ttl = zone_ttl
        self.x_min, self.y_min = x_min, y_min
        self.nx = int(np.ceil((x_max - x_min) / resolution))
        self.ny = int(np.ceil((y_max - y_min) / resolution))
        self.n_safe = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.n_unsafe = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.zone_until = np.full((self.nx, self.ny), -np.inf)

    def _to_idx(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((pts[:, 0] - self.x_min) / self.res).astype(int)
        iy = np.floor((pts[:, 1] - self.y_min) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix[ok], iy[ok], ok

    def add_points(self, safe_pts: np.ndarray, unsafe_pts: np.ndarray):
        if len(safe_pts):
            ix, iy, _ = self._to_idx(safe_pts)
            np.add.at(self.n_safe, (ix, iy), 1.0)
        if len(unsafe_pts):
            ix, iy, _ = self._to_idx(unsafe_pts)
            np.add.at(self.n_unsafe, (ix, iy), 1.0)

    def paint_zone(self, pts: np.ndarray, now: float | None = None):
        """Mark VLM-declared zone cells; refreshed cells extend their TTL."""
        if not len(pts):
            return
        now = time.time() if now is None else now
        ix, iy, _ = self._to_idx(pts)
        self.zone_until[ix, iy] = np.maximum(self.zone_until[ix, iy],
                                             now + self.zone_ttl)

    def occupied(self) -> np.ndarray:
        return self.n_unsafe >= self.occ_thresh

    def zone_active(self, now: float | None = None) -> np.ndarray:
        now = time.time() if now is None else now
        return self.zone_until > now

    def observed(self) -> np.ndarray:
        return ((self.n_safe > 0) | (self.n_unsafe > 0)
                | (self.zone_until > -np.inf))

    def p_safe(self) -> np.ndarray:
        total = self.n_safe + self.n_unsafe
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(total > 0, self.n_safe / np.maximum(total, 1e-12), 1.0)
        return p

    def safe_grid(self, now: float | None = None) -> np.ndarray:
        """True where traversal is allowed: neither depth-confirmed
        occupancy nor an active VLM zone (unobserved -> traversable)."""
        return ~(self.occupied() | self.zone_active(now))

    def state_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(np.floor((x - self.x_min) / self.res), 0, self.nx - 1))
        iy = int(np.clip(np.floor((y - self.y_min) / self.res), 0, self.ny - 1))
        return ix, iy

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.res
        return xs, ys

    def recenter(self, x: float, y: float):
        """Ego window: scroll the grid (whole cells) so (x, y) sits at the
        window center. Cell contents keep their WORLD position — only the
        window moves; votes scrolled out of the window are forgotten."""
        cx = self.x_min + self.nx * self.res / 2.0
        cy = self.y_min + self.ny * self.res / 2.0
        dx = int(round((x - cx) / self.res))
        dy = int(round((y - cy) / self.res))
        if dx == 0 and dy == 0:
            return
        src_x = slice(max(dx, 0), self.nx + min(dx, 0))
        dst_x = slice(max(-dx, 0), self.nx + min(-dx, 0))
        src_y = slice(max(dy, 0), self.ny + min(dy, 0))
        dst_y = slice(max(-dy, 0), self.ny + min(-dy, 0))
        for name, fill in (("n_safe", 0.0), ("n_unsafe", 0.0),
                           ("zone_until", -np.inf)):
            arr = getattr(self, name)
            out = np.full_like(arr, fill)
            out[dst_x, dst_y] = arr[src_x, src_y]
            setattr(self, name, out)
        self.x_min += dx * self.res
        self.y_min += dy * self.res
