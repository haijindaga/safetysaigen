"""Planar control-affine robot model (Eq. 12).

    x_{t+1} = x_t + dt * R(theta_t) u_t

State x = (x, y, theta); input u = (v_x, v_y, omega) — body-frame planar
velocity plus yaw rate. This velocity-command interface is shared by:
  - differential/omnidirectional mobile bases (directly),
  - quadrupeds (locomotion policy consumes body-velocity commands),
  - mobile manipulators (base controller consumes velocity commands),
which is what makes the framework robot-agnostic at the control level.

Continuous form: xdot = f(x) + g(x) u with f = 0, g(x) = R(theta).
"""
from __future__ import annotations

import numpy as np


class PlanarRobot:
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                 dt: float = 0.1,
                 v_max: float = 0.35, omega_max: float = 1.0,
                 holonomic: bool = True):
        self.state = np.array([x, y, theta], dtype=float)
        self.dt = dt
        self.v_max = v_max
        self.omega_max = omega_max
        self.holonomic = holonomic

    @staticmethod
    def rotation(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

    def f(self) -> np.ndarray:
        return np.zeros(3)

    def g(self) -> np.ndarray:
        return self.rotation(self.state[2])

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).copy()
        if not self.holonomic:
            u[1] = 0.0
        speed = np.hypot(u[0], u[1])
        if speed > self.v_max:
            u[:2] *= self.v_max / speed
        u[2] = np.clip(u[2], -self.omega_max, self.omega_max)
        return u

    def step(self, u: np.ndarray) -> np.ndarray:
        u = self.clip_input(u)
        self.state = self.state + self.dt * self.g() @ u
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        return self.state.copy()
