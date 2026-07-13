"""Lightweight semantic 3D world for closed-loop testing on Windows.

Objects are vertical extrusions (cylinders / axis-aligned boxes) standing on
the z=0 ground plane; the ground itself is partitioned into semantic patches
(floor, grass, sidewalk, ...). A raycasting pinhole camera renders RGB-D
plus ground-truth semantic labels — enough to exercise the *entire* CORE
grounding stack (segmentation masks, spatial operators, depth projection).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cylinder:
    cls: str
    cx: float
    cy: float
    radius: float
    height: float


@dataclass
class Box:
    cls: str
    cx: float
    cy: float
    wx: float          # full extent along x
    wy: float          # full extent along y
    height: float

    @property
    def x_min(self): return self.cx - self.wx / 2
    @property
    def x_max(self): return self.cx + self.wx / 2
    @property
    def y_min(self): return self.cy - self.wy / 2
    @property
    def y_max(self): return self.cy + self.wy / 2


@dataclass
class GroundPatch:
    """Axis-aligned rectangular ground region with a semantic class."""
    cls: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x >= self.x_min) & (x <= self.x_max)
                & (y >= self.y_min) & (y <= self.y_max))


@dataclass
class World:
    objects: list = field(default_factory=list)       # Cylinder | Box
    ground_patches: list = field(default_factory=list)  # later patches win
    default_ground: str = "floor"
    # Ground-truth unsafe regions for *evaluation only* (never shown to CORE):
    # list of (label, contains_fn) where contains_fn(x, y) -> bool.
    gt_unsafe_regions: list = field(default_factory=list)

    def class_names(self) -> list[str]:
        names = {self.default_ground}
        names.update(p.cls for p in self.ground_patches)
        names.update(o.cls for o in self.objects)
        return sorted(names)

    def ground_class_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized ground-class lookup; returns array of class ids
        according to self.class_ids()."""
        ids = self.class_ids()
        out = np.full(np.shape(x), ids[self.default_ground], dtype=np.int32)
        for p in self.ground_patches:
            m = p.contains(np.asarray(x), np.asarray(y))
            out[m] = ids[p.cls]
        return out

    def class_ids(self) -> dict[str, int]:
        return {n: i + 1 for i, n in enumerate(self.class_names())}

    def id_to_name(self) -> dict[int, str]:
        return {i: n for n, i in self.class_ids().items()}

    def instance_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.cls] = counts.get(o.cls, 0) + 1
        return counts

    def in_gt_unsafe(self, x: float, y: float) -> str | None:
        for label, fn in self.gt_unsafe_regions:
            if fn(x, y):
                return label
        return None
