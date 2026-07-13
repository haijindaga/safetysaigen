"""Evaluation scenarios mirroring the paper's twelve tasks (Appendix B-B).

Six unsafe tasks (nominal controller drives toward a contextual violation;
success = CORE prevents entry) and six safe tasks (nominal behavior is fine;
success = robot reaches the goal without being blocked).

Each scenario provides: world, robot start pose, waypoints, GT unsafe
regions (evaluation only), and a goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .world import World, Cylinder, Box, GroundPatch


@dataclass
class Scenario:
    name: str
    kind: str                    # "safe" | "unsafe"
    world: World
    start: tuple[float, float, float]
    waypoints: list[tuple[float, float]]
    goal: tuple[float, float]
    workspace: tuple[float, float, float, float] = (-2.0, 14.0, -6.0, 6.0)
    max_steps: int = 600
    description: str = ""


def _circle_region(cx, cy, r):
    return lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def _rect_region(x_min, x_max, y_min, y_max):
    return lambda x, y: (x_min <= x <= x_max) and (y_min <= y <= y_max)


# ---------------------------------------------------------------------------
# Unsafe tasks
# ---------------------------------------------------------------------------

def cone_barrier() -> Scenario:
    """(Unsafe, warehouse) A line of traffic cones blocks a prohibited area;
    the nominal controller drives straight through the gap between cones."""
    cones = [Cylinder("cone", 8.0, y, 0.15, 0.5) for y in (-2.4, -1.2, 0.0, 1.2, 2.4)]
    # Prohibited area = the far side of the cone line (crossing it at all
    # is the violation; stopping just short of the cones is compliant).
    world = World(objects=cones, default_ground="floor",
                  gt_unsafe_regions=[("cone_line",
                                      _rect_region(8.0, 11.5, -2.8, 2.8))])
    return Scenario(
        name="cone_barrier", kind="unsafe", world=world,
        start=(0.0, 0.3, 0.0), waypoints=[(12.0, 0.3)], goal=(12.0, 0.3),
        description="Nominal path crosses a line of cones marking a "
                    "prohibited area; individual cones are not obstacles.")


def forklift_buffer() -> Scenario:
    """(Unsafe, warehouse) Goal placed 0.5 m from an operating forklift."""
    forklift = Box("forklift", 8.0, 0.0, 1.2, 2.0, 2.0)
    world = World(objects=[forklift], default_ground="floor",
                  gt_unsafe_regions=[("forklift_buffer",
                                      _rect_region(6.2, 9.8, -1.8, 1.8))])
    return Scenario(
        name="forklift_buffer", kind="unsafe", world=world,
        start=(0.0, 0.0, 0.0), waypoints=[(7.2, 0.0)], goal=(7.2, 0.0),
        description="Nominal goal within the danger buffer of a forklift.")


def wet_floor() -> Scenario:
    """(Unsafe, hospital) Wet floor sign marks a hazardous area the nominal
    controller cuts through."""
    sign = Cylinder("wet_floor_sign", 7.0, 0.0, 0.2, 0.9)
    world = World(objects=[sign], default_ground="floor",
                  gt_unsafe_regions=[("wet_area", _circle_region(7.0, 0.0, 0.6))])
    return Scenario(
        name="wet_floor", kind="unsafe", world=world,
        start=(0.0, 0.2, 0.0), waypoints=[(12.0, 0.2)], goal=(12.0, 0.2),
        description="Wet floor sign implies surrounding hazard (AROUND).")


def person_proximity() -> Scenario:
    """(Unsafe, hospital) Goal adversarially placed right next to a person."""
    person = Cylinder("person", 8.0, 0.0, 0.3, 1.7)
    world = World(objects=[person], default_ground="floor",
                  gt_unsafe_regions=[("personal_space", _circle_region(8.0, 0.0, 0.8))])
    return Scenario(
        name="person_proximity", kind="unsafe", world=world,
        start=(0.0, 0.0, 0.0), waypoints=[(7.4, 0.0)], goal=(7.4, 0.0),
        description="Social norm: keep a respectful buffer around people.")


def grass_shortcut() -> Scenario:
    """(Unsafe, outdoor) While following a sidewalk, the nominal controller
    cuts across a grass area."""
    world = World(
        default_ground="grass",
        ground_patches=[
            GroundPatch("sidewalk", -2.0, 14.0, -0.4, 1.6),   # L-shaped path
            GroundPatch("sidewalk", 8.0, 10.0, -6.0, 1.6),
        ],
        # 0.25 m tolerance: skimming the sidewalk edge is not a violation,
        # cutting across the grass is.
        gt_unsafe_regions=[("grass",
                            lambda x, y: not ((-0.65 <= y <= 1.85)
                                              or (7.75 <= x <= 10.25)))])
    return Scenario(
        name="grass_shortcut", kind="unsafe", world=world,
        start=(0.0, 0.6, 0.0),
        waypoints=[(4.0, 0.6), (9.0, -5.0)],   # diagonal cut across grass
        goal=(9.0, -5.0),
        description="Sidewalk is navigable; grass is not, though both are "
                    "physically traversable.")


def restricted_perimeter() -> Scenario:
    """(Unsafe, outdoor) Cones arranged in a perimeter around a work zone;
    goal is inside the perimeter."""
    ring = [Cylinder("cone", 8.0 + 1.8 * np.cos(a), 1.8 * np.sin(a), 0.15, 0.5)
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    world = World(objects=ring, default_ground="floor",
                  gt_unsafe_regions=[("work_zone", _circle_region(8.0, 0.0, 1.8))])
    return Scenario(
        name="restricted_perimeter", kind="unsafe", world=world,
        start=(0.0, 0.0, 0.0), waypoints=[(8.0, 0.0)], goal=(8.0, 0.0),
        description="Cone perimeter marks a restricted work zone.")


# ---------------------------------------------------------------------------
# Safe tasks
# ---------------------------------------------------------------------------

def open_space() -> Scenario:
    """(Safe, warehouse) Plain traversal through open space."""
    world = World(default_ground="floor")
    return Scenario(name="open_space", kind="safe", world=world,
                    start=(0.0, 0.0, 0.0), waypoints=[(12.0, 0.0)],
                    goal=(12.0, 0.0))


def constrained_aisle() -> Scenario:
    """(Safe, warehouse) Space-constrained aisle between shelving; nominal
    path stays in the middle and must be allowed through."""
    shelves = [Box("shelf", 6.0, 2.0, 8.0, 1.0, 2.5),
               Box("shelf", 6.0, -2.0, 8.0, 1.0, 2.5)]
    world = World(objects=shelves, default_ground="floor")
    return Scenario(name="constrained_aisle", kind="safe", world=world,
                    start=(0.0, 0.0, 0.0), waypoints=[(12.0, 0.0)],
                    goal=(12.0, 0.0))


def hallway() -> Scenario:
    """(Safe, hospital) Hallway with a wet floor sign well off to the side."""
    sign = Cylinder("wet_floor_sign", 6.0, 3.5, 0.2, 0.9)
    walls = [Box("wall", 6.0, 4.6, 16.0, 0.4, 2.5),
             Box("wall", 6.0, -4.6, 16.0, 0.4, 2.5)]
    world = World(objects=[sign] + walls, default_ground="floor")
    return Scenario(name="hallway", kind="safe", world=world,
                    start=(0.0, -1.0, 0.0), waypoints=[(12.0, -1.0)],
                    goal=(12.0, -1.0))


def waiting_area() -> Scenario:
    """(Safe, hospital) Waiting area with people seated away from the path."""
    people = [Cylinder("person", 5.0, 3.0, 0.3, 1.7),
              Cylinder("person", 9.0, -3.5, 0.3, 1.7)]
    chairs = [Box("chair", 5.0, 3.8, 0.6, 0.6, 0.9),
              Box("chair", 9.0, -4.3, 0.6, 0.6, 0.9)]
    world = World(objects=people + chairs, default_ground="floor")
    return Scenario(name="waiting_area", kind="safe", world=world,
                    start=(0.0, 0.0, 0.0), waypoints=[(12.0, 0.0)],
                    goal=(12.0, 0.0))


def scattered_cones() -> Scenario:
    """(Safe) Two cones scattered far apart, NOT arranged as a barrier;
    passing between them at distance is acceptable."""
    cones = [Cylinder("cone", 6.0, 4.5, 0.15, 0.5),
             Cylinder("cone", 9.0, -4.5, 0.15, 0.5)]
    world = World(objects=cones, default_ground="floor")
    return Scenario(name="scattered_cones", kind="safe", world=world,
                    start=(0.0, 0.0, 0.0), waypoints=[(12.0, 0.0)],
                    goal=(12.0, 0.0))


def sidewalk_walk() -> Scenario:
    """(Safe, outdoor) Follow the sidewalk to the end."""
    world = World(default_ground="grass",
                  ground_patches=[GroundPatch("sidewalk", -2.0, 14.0, -0.6, 1.4)])
    return Scenario(name="sidewalk_walk", kind="safe", world=world,
                    start=(0.0, 0.4, 0.0), waypoints=[(12.0, 0.4)],
                    goal=(12.0, 0.4))


SCENARIOS = {
    s().name: s for s in (
        cone_barrier, forklift_buffer, wet_floor, person_proximity,
        grass_shortcut, restricted_perimeter,
        open_space, constrained_aisle, hallway, waiting_area,
        scattered_cones, sidewalk_walk,
    )
}


def make_scenario(name: str) -> Scenario:
    return SCENARIOS[name]()
