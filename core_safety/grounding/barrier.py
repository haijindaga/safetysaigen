"""Signed-distance-field barrier function over the thresholded safe set.

    h(x) =  min_{y in dS} ||x - y||   if x in S      (Eq. 4)
         = -min_{y in dS} ||x - y||   otherwise

The gradient is analytical (Eq. 8): the unit vector away from the nearest
boundary point y*(x), found via grid lookups (Euclidean distance transform
with nearest-cell indices).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class SDFBarrier:
    def __init__(self, safe_grid: np.ndarray, x_min: float, y_min: float,
                 resolution: float):
        """safe_grid: (nx, ny) bool, True = safe."""
        self.res = resolution
        self.x_min, self.y_min = x_min, y_min
        self.safe = safe_grid

        if safe_grid.all():
            # No unsafe region observed: h = +inf everywhere (CBF inactive).
            self.h_grid = np.full(safe_grid.shape, np.inf)
            self.grad_x = np.zeros(safe_grid.shape)
            self.grad_y = np.zeros(safe_grid.shape)
            return
        if not safe_grid.any():
            self.h_grid = np.full(safe_grid.shape, -np.inf)
            self.grad_x = np.zeros(safe_grid.shape)
            self.grad_y = np.zeros(safe_grid.shape)
            return

        # Distance (in cells) from each safe cell to nearest unsafe cell,
        # and vice versa; combine into a signed distance field.
        d_out, (ix_o, iy_o) = ndimage.distance_transform_edt(
            safe_grid, return_indices=True)          # inside S: dist to unsafe
        d_in, (ix_i, iy_i) = ndimage.distance_transform_edt(
            ~safe_grid, return_indices=True)         # outside S: dist to safe
        # Half-cell offset places the h = 0 level set on the cell boundary
        # between safe and unsafe cells (EDT distances are center-to-center).
        self.h_grid = np.where(safe_grid, d_out - 0.5, -(d_in - 0.5)) * resolution

        # Nearest boundary cell for each cell -> analytical gradient direction.
        nx, ny = safe_grid.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        near_x = np.where(safe_grid, ix_o, ix_i)
        near_y = np.where(safe_grid, iy_o, iy_i)
        dx = (gx - near_x).astype(float)
        dy = (gy - near_y).astype(float)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        norm = np.where(norm > 0, norm, 1.0)
        sign = np.where(safe_grid, 1.0, -1.0)
        self.grad_x = sign * dx / norm
        self.grad_y = sign * dy / norm

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        nx, ny = self.safe.shape
        ix = int(np.clip(np.floor((x - self.x_min) / self.res), 0, nx - 1))
        iy = int(np.clip(np.floor((y - self.y_min) / self.res), 0, ny - 1))
        return ix, iy

    def h(self, x: float, y: float) -> float:
        ix, iy = self._cell(x, y)
        return float(self.h_grid[ix, iy])

    def grad_h(self, x: float, y: float) -> np.ndarray:
        """d h / d (x, y): unit vector pointing into the safe set interior."""
        ix, iy = self._cell(x, y)
        return np.array([self.grad_x[ix, iy], self.grad_y[ix, iy]])
