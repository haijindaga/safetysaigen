"""CORE pipeline orchestrator: reasoning -> grounding -> enforcement.

Wires the three modules together for one perception-control loop:

  observation (RGB-D)                              [fast inner loop]
    -> VLM safety constraints        (slow outer loop, every N steps)
    -> segmentation + spatial ops -> image safe set
    -> depth projection -> costmap counts -> SDF barrier h(x)
    -> CBF-QP filter over u_nom  ->  u_safe
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .predicates import SafetyConstraints
from .reasoning.vlm_client import VLMClient
from .grounding.segmentation import Segmenter
from .grounding.image_safe_set import build_image_safe_set
from .grounding.projection import PinholeCamera, project_masks_to_world
from .grounding.costmap import SemanticCostmap
from .grounding.barrier import SDFBarrier
from .control.cbf_qp import CBFSafetyFilter


@dataclass
class CoreConfig:
    resolution: float = 0.2         # costmap resolution [m]
    tau: float = 0.5                # safe-probability threshold
    alpha_gain: float = 0.25        # class-K gain
    around_kernel_px: int = 50      # AROUND dilation kernel
    min_range: float = 3.0          # depth clip [m] (paper: 3-7 m)
    max_range: float = 7.0
    v_max: float = 0.35
    omega_max: float = 1.0
    perception_period: int = 10     # control steps between perception updates
    pixel_stride: int = 2


@dataclass
class CoreDebug:
    constraints: SafetyConstraints | None = None
    safe_mask: np.ndarray | None = None
    unsafe_mask: np.ndarray | None = None
    h: float = np.inf
    filtered: bool = False
    history: list = field(default_factory=list)


class CorePipeline:
    def __init__(self, vlm: VLMClient, segmenter: Segmenter,
                 camera: PinholeCamera, workspace: tuple[float, float, float, float],
                 config: CoreConfig | None = None):
        self.vlm = vlm
        self.segmenter = segmenter
        self.camera = camera
        self.cfg = config or CoreConfig()
        x_min, x_max, y_min, y_max = workspace
        self.costmap = SemanticCostmap(x_min, x_max, y_min, y_max,
                                       self.cfg.resolution, self.cfg.tau)
        self.barrier: SDFBarrier | None = None
        self.filter = CBFSafetyFilter(self.cfg.alpha_gain,
                                      self.cfg.v_max, self.cfg.omega_max)
        self.debug = CoreDebug()
        self._step = 0

    # ---- perception (slow outer loop) ------------------------------------
    def update_perception(self, rgb: np.ndarray, depth: np.ndarray,
                          robot_pose: np.ndarray,
                          visible_classes: list[str] | None = None,
                          instance_counts: dict[str, int] | None = None):
        """Run reasoning + grounding on one RGB-D frame, refresh the barrier."""
        try:
            kwargs = {}
            if instance_counts is not None:
                kwargs["instance_counts"] = instance_counts
            constraints = self.vlm.infer(rgb, visible_classes=visible_classes, **kwargs)
        except (ValueError, TypeError):
            try:
                constraints = self.vlm.infer(rgb, visible_classes=visible_classes)
            except ValueError:
                # Reasoning failure (invalid VLM output): keep previous barrier.
                return
        self.debug.constraints = constraints

        seg = self.segmenter.segment(rgb, constraints.all_classes())
        if not seg:
            return
        safe_mask, unsafe_mask = build_image_safe_set(
            constraints, seg, self.cfg.around_kernel_px)
        self.debug.safe_mask, self.debug.unsafe_mask = safe_mask, unsafe_mask

        safe_pts, unsafe_pts = project_masks_to_world(
            self.camera, depth, safe_mask, unsafe_mask, robot_pose,
            self.cfg.min_range, self.cfg.max_range, self.cfg.pixel_stride)
        self.costmap.add_points(safe_pts, unsafe_pts)
        self.barrier = SDFBarrier(self.costmap.safe_grid(),
                                  self.costmap.x_min, self.costmap.y_min,
                                  self.costmap.res)

    # ---- control (fast inner loop) ----------------------------------------
    def safe_control(self, u_nom: np.ndarray, robot) -> np.ndarray:
        """Filter u_nom through the CBF-QP given the current barrier.

        A discrete-time backtracking check follows the QP: because h(x) is
        piecewise constant over grid cells, the continuous-time constraint
        can creep past thin unsafe regions over multiple steps (cf. Brunke
        et al. 2024 on discrete-time implementations of continuous-time
        CBF filters, ref. [54] of the paper). We verify forward invariance
        on the predicted next state — h(x_{t+1}) >= 0 (and no sinking when
        already outside) — and scale the input down until it holds (u = 0
        always does). The graceful slowdown near the boundary remains the
        QP's job via alpha; the backtrack only vetoes the final cell
        transition into the unsafe set, which the piecewise-constant grid
        otherwise lets through.
        """
        if self.barrier is None:
            self.debug.h = np.inf
            self.debug.filtered = False
            return u_nom
        x, y, _ = robot.state
        h = self.barrier.h(x, y)
        grad = self.barrier.grad_h(x, y)
        u_safe = self.filter.filter(u_nom, h, grad, robot.f(), robot.g())
        backtracked = False

        if np.isfinite(h):
            dt = getattr(robot, "dt", 0.1)
            h_min_next = 0.0 if h >= 0 else h        # stay in S / never sink
            g = robot.g()
            chosen = np.zeros(3)
            for scale in (1.0, 0.5, 0.25, 0.1, 0.0):
                u_try = robot.clip_input(u_safe * scale)
                nxt = robot.state + dt * (robot.f() + g @ u_try)
                if self.barrier.h(nxt[0], nxt[1]) >= h_min_next - 1e-9:
                    chosen = u_try
                    break
            backtracked = not np.allclose(chosen, u_safe)
            u_safe = chosen
        self.debug.h = h
        self.debug.filtered = self.filter.last_active or backtracked
        return u_safe

    # ---- combined step ------------------------------------------------------
    def step(self, rgb, depth, robot, u_nom, visible_classes=None,
             instance_counts=None) -> np.ndarray:
        if self._step % self.cfg.perception_period == 0:
            self.update_perception(rgb, depth, robot.state, visible_classes,
                                   instance_counts)
        self._step += 1
        return self.safe_control(u_nom, robot)
