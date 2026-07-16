"""2D semantic costmap accumulating safe/unsafe observation counts (Eq. 3).

Each cell C_ij keeps counts n_safe and n_unsafe of projected point
observations. P[x is safe] = n_safe / (n_safe + n_unsafe), thresholded at
tau (paper: resolution 0.2 m, tau = 0.5) to form the safe set S.

Unobserved cells are treated as safe: the safe set is defined by identified
unsafe regions (Assumption 1 requires perception to flag unsafe regions
before entry; an all-unsafe prior would freeze the robot at t=0).
"""
from __future__ import annotations

import numpy as np


class SemanticCostmap:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float = 0.2, tau: float = 0.5):
        self.res = resolution
        self.tau = tau
        self.x_min, self.y_min = x_min, y_min
        self.nx = int(np.ceil((x_max - x_min) / resolution))
        self.ny = int(np.ceil((y_max - y_min) / resolution))
        self.n_safe = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.n_unsafe = np.zeros((self.nx, self.ny), dtype=np.float64)

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

    def p_safe(self) -> np.ndarray:
        total = self.n_safe + self.n_unsafe
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(total > 0, self.n_safe / np.maximum(total, 1e-12), 1.0)
        return p

    def safe_grid(self) -> np.ndarray:
        """Boolean grid: True where P[safe] >= tau (unobserved -> safe)."""
        return self.p_safe() >= self.tau

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
        for name in ("n_safe", "n_unsafe"):
            arr = getattr(self, name)
            out = np.zeros_like(arr)
            out[dst_x, dst_y] = arr[src_x, src_y]
            setattr(self, name, out)
        self.x_min += dx * self.res
        self.y_min += dy * self.res
