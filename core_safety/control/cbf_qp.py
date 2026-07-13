"""CBF-QP safety filter (Eq. 7).

    u_safe = argmin_u ||u - u_nom||^2
             s.t.  <grad_h(x), f(x) + g(x) u>  >=  -alpha(h(x))

Solved with CVXPY (paper: alpha(x) = 0.25 x). h depends only on planar
position, so grad_h = (dh/dx, dh/dy, 0). u = 0 is always feasible for
f = 0 dynamics, so the QP cannot be infeasible unless input bounds
exclude 0 (they don't).
"""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:  # pragma: no cover
    _HAS_CVXPY = False


class CBFSafetyFilter:
    def __init__(self, alpha_gain: float = 0.25,
                 v_max: float = 0.35, omega_max: float = 1.0,
                 solver: str | None = None):
        self.alpha_gain = alpha_gain
        self.v_max = v_max
        self.omega_max = omega_max
        self.solver = solver
        self.last_active = False  # True when the constraint modified u_nom

    def alpha(self, h: float) -> float:
        return self.alpha_gain * h

    def filter(self, u_nom: np.ndarray, h: float, grad_h_xy: np.ndarray,
               f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute u_safe. grad_h_xy: (dh/dx, dh/dy); f, g: dynamics terms."""
        if not np.isfinite(h):
            # No barrier active (h = +inf): pass through.
            self.last_active = False
            return u_nom
        grad = np.array([grad_h_xy[0], grad_h_xy[1], 0.0])
        a = grad @ g                      # row vector: a . u >= b
        b = -self.alpha(h) - grad @ f

        # Fast path: nominal input already satisfies the constraint.
        if a @ u_nom >= b - 1e-9:
            self.last_active = False
            return u_nom
        self.last_active = True

        if _HAS_CVXPY:
            u = cp.Variable(3)
            prob = cp.Problem(
                cp.Minimize(cp.sum_squares(u - u_nom)),
                [a @ u >= b,
                 cp.norm(u[:2]) <= self.v_max,
                 cp.abs(u[2]) <= self.omega_max])
            try:
                prob.solve(solver=self.solver)
                if u.value is not None:
                    return np.asarray(u.value).ravel()
            except cp.error.SolverError:
                pass

        # Analytical fallback (no input bounds): project u_nom onto the
        # halfspace {u : a.u >= b}.
        denom = float(a @ a)
        if denom < 1e-12:
            return np.zeros(3)
        return u_nom + (b - a @ u_nom) / denom * a
