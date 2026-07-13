"""Nominal controller: waypoint follower with proportional control.

Deliberately safety-unaware — in unsafe tasks it is pointed at
adversarially placed goals and the CBF filter must intervene.
"""
from __future__ import annotations

import numpy as np


class WaypointFollower:
    def __init__(self, waypoints: list[tuple[float, float]],
                 v_max: float = 0.35, omega_max: float = 1.0,
                 k_v: float = 1.0, k_w: float = 2.0,
                 reach_tol: float = 0.3):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.idx = 0
        self.v_max = v_max
        self.omega_max = omega_max
        self.k_v = k_v
        self.k_w = k_w
        self.reach_tol = reach_tol

    @property
    def done(self) -> bool:
        return self.idx >= len(self.waypoints)

    def compute(self, state: np.ndarray) -> np.ndarray:
        """Return body-frame u_nom = (v_x, v_y, omega)."""
        if self.done:
            return np.zeros(3)
        x, y, th = state
        target = self.waypoints[self.idx]
        delta = target - np.array([x, y])
        dist = np.linalg.norm(delta)
        if dist < self.reach_tol:
            self.idx += 1
            return self.compute(state)

        # World-frame desired velocity -> body frame.
        v_world = self.k_v * delta
        speed = np.linalg.norm(v_world)
        if speed > self.v_max:
            v_world *= self.v_max / speed
        c, s = np.cos(th), np.sin(th)
        v_body = np.array([c * v_world[0] + s * v_world[1],
                           -s * v_world[0] + c * v_world[1]])
        # Turn toward motion direction so the camera looks where we go.
        heading_err = np.arctan2(delta[1], delta[0]) - th
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        omega = np.clip(self.k_w * heading_err, -self.omega_max, self.omega_max)
        return np.array([v_body[0], v_body[1], omega])
