"""Closed-loop episode runner and baseline definitions (Sec. V).

Methods:
  core      CORE with the rule-based contextual reasoner (on Windows) or a
            real VLM client (pass one in). Oracle-equivalent when rule-based.
  oracle    Ground-truth contextual constraints (same rulebook, given a
            priori — identical grounding/control stack).
  nocontext Generic pre-deployment rules: only prominent hazard signage is
            treated contextually; terrain semantics are not distinguished.
  geometric Pure metric obstacle avoidance: every object is NEAR-unsafe,
            every surface is traversable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..control.dynamics import PlanarRobot
from ..control.nominal import WaypointFollower
from ..grounding.segmentation import GroundTruthSegmenter
from ..pipeline import CorePipeline, CoreConfig
from ..reasoning.vlm_client import RuleBasedVLM, VLMClient, DEFAULT_RULEBOOK
from .camera import RaycastCamera
from .scenarios import Scenario

ALL_GROUNDS = {"floor", "sidewalk", "grass", "road", "hallway", "flower_bed"}

NO_CONTEXT_RULEBOOK = {
    "navigable": ALL_GROUNDS,          # cannot tell grass from sidewalk
    "non_navigable": set(),
    "hazard_around": {"wet_floor_sign", "hazard_sign"},  # obvious signage only
    "barrier_between": set(),
}

GEOMETRIC_RULEBOOK = {
    "navigable": ALL_GROUNDS,
    "non_navigable": set(),
    "hazard_around": set(),
    "barrier_between": set(),
}


def make_vlm(method: str) -> VLMClient:
    if method in ("core", "oracle"):
        return RuleBasedVLM(DEFAULT_RULEBOOK, contextual=True)
    if method == "nocontext":
        return RuleBasedVLM(NO_CONTEXT_RULEBOOK, contextual=True)
    if method == "geometric":
        return RuleBasedVLM(GEOMETRIC_RULEBOOK, contextual=False)
    raise ValueError(method)


@dataclass
class EpisodeResult:
    scenario: str
    kind: str
    method: str
    success: bool
    violated: bool
    violation_label: str | None
    reached_goal: bool
    steps: int
    min_h: float
    trajectory: np.ndarray = field(repr=False, default=None)


def run_episode(scenario: Scenario, method: str = "core",
                vlm: VLMClient | None = None,
                config: CoreConfig | None = None,
                start_jitter: float = 0.0,
                rng: np.random.Generator | None = None,
                record: bool = False) -> EpisodeResult:
    cfg = config or CoreConfig()
    vlm = vlm or make_vlm(method)

    sx, sy, sth = scenario.start
    if start_jitter > 0:
        rng = rng or np.random.default_rng()
        sx += rng.uniform(-start_jitter, start_jitter)
        sy += rng.uniform(-start_jitter, start_jitter)

    robot = PlanarRobot(sx, sy, sth, v_max=cfg.v_max, omega_max=cfg.omega_max)
    nominal = WaypointFollower(scenario.waypoints, v_max=cfg.v_max,
                               omega_max=cfg.omega_max)
    camera = RaycastCamera()
    seg = GroundTruthSegmenter()
    pipeline = CorePipeline(vlm, seg, camera.cam, scenario.workspace, cfg)

    world = scenario.world
    traj = [robot.state.copy()]
    violated, violation_label = False, None
    min_h = np.inf
    goal = np.asarray(scenario.goal)
    reached = False

    for step in range(scenario.max_steps):
        u_nom = nominal.compute(robot.state)

        if step % cfg.perception_period == 0:
            rgb, depth, labels = camera.render(world, robot.state)
            seg.update(labels, world.id_to_name())
            pipeline.update_perception(
                rgb, depth, robot.state,
                visible_classes=_visible_classes(labels, world),
                instance_counts=world.instance_counts())

        u_safe = pipeline.safe_control(u_nom, robot)
        robot.step(u_safe)
        traj.append(robot.state.copy())
        if np.isfinite(pipeline.debug.h):
            min_h = min(min_h, pipeline.debug.h)

        label = world.in_gt_unsafe(robot.state[0], robot.state[1])
        if label is not None:
            violated, violation_label = True, label
        if np.linalg.norm(robot.state[:2] - goal) < 0.3:
            reached = True
            break
        if nominal.done:
            break

    if scenario.kind == "safe":
        success = reached and not violated
    else:
        success = not violated

    return EpisodeResult(
        scenario=scenario.name, kind=scenario.kind, method=method,
        success=success, violated=violated, violation_label=violation_label,
        reached_goal=reached, steps=len(traj) - 1, min_h=float(min_h),
        trajectory=np.array(traj) if record else None)


def _visible_classes(labels: np.ndarray, world) -> list[str]:
    id2n = world.id_to_name()
    present = np.unique(labels)
    return [id2n[i] for i in present if i in id2n]
