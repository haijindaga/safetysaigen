"""Probabilistic safe-traversal certificate (Theorem 1, Appendix B-D).

Maximum Traversal Speed (MTS) formulation: consider a worst-case obstacle
at distance D toward which the planner drives at maximum speed, with k
measurement opportunities before contact. The certificate requires

  sum_{i=1}^{k-1}  c / (l + D(k-i)/k)
      * prod_{j=0}^{i-1} (1 - m_D(D(k-j)/k)) * m_D(D(k-i)/k)
  + R(m_D) * c / l   <=   c (delta - gamma) / l

where R(m_D) = prod_j (1 - m_D(D(k-j)/k)) is the probability of reaching
the unsafe region undetected. If it holds for kappa* = min such k, the
robot may traverse at speed up to D / (kappa* * t_perception).

Paper's calibration: m_D(r) = 0.75 - eps*r/D (75% = high-probability lower
bound of the 85% empirical unsafe-detection rate of Gemma 3 27B), D = 4 m,
v_max = 0.35 m/s, latency ~3 s  =>  kappa* = 3; certificate found at
delta = 0.1, gamma = 0, with c = 1, l = 0.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def default_m(r: np.ndarray | float, D: float = 4.0, p: float = 0.75,
              eps: float = 0.05) -> np.ndarray | float:
    """Conservative detection-probability function m_D(r) = p - eps*r/D,
    zero beyond the sensing radius D (Assumption 2)."""
    r = np.asarray(r, dtype=float)
    m = np.where(r < D, p - eps * r / D, 0.0)
    return np.clip(m, 0.0, 1.0 - 1e-9)


def mts_residual(k: int, c: float, ell: float, D: float = 4.0,
                 m=None) -> tuple[float, float]:
    """Return (lhs, rhs_coeff) of the MTS constraint.

    lhs is the expected realized inverse regularized distance for the
    'beeline' scenario; the constraint is lhs <= c*(delta-gamma)/ell.
    """
    if m is None:
        m = lambda r: default_m(r, D=D)
    lhs = 0.0
    surv = 1.0  # prod of (1 - m) so far: still undetected
    for i in range(1, k):
        r_i = D * (k - i) / k
        m_i = float(m(r_i))
        # survive steps j = 0..i-1 then detect at step i
        prob_detect_at_i = surv * m_i
        lhs += prob_detect_at_i * c / (ell + r_i)
        surv *= (1.0 - m_i)
    # remainder: never detected before contact -> inverse distance c/ell
    lhs += surv * c / ell
    return lhs, c / ell


def certificate_holds(k: int, c: float, ell: float, delta: float,
                      gamma: float = 0.0, D: float = 4.0, m=None) -> bool:
    lhs, c_over_l = mts_residual(k, c, ell, D, m)
    return lhs <= (delta - gamma) * c_over_l


@dataclass
class Certificate:
    kappa: int
    c: float
    ell: float
    delta: float
    gamma: float
    D: float
    max_speed: float  # D / (kappa * t_perception)
    residual: float   # slack of the constraint (rhs - lhs), >= 0


def find_certificate(delta: float = 0.1, gamma: float = 0.0, D: float = 4.0,
                     t_perception: float = 3.0, m=None,
                     k_max: int = 50) -> Certificate | None:
    """Search (k, c, l) for the smallest k admitting a certificate.

    The constraint is scale-invariant in c (both sides linear in c), so fix
    c = 1 and grid-search l.
    """
    c = 1.0
    for k in range(1, k_max + 1):
        best = None
        for ell in np.geomspace(1e-3, 10.0, 400):
            lhs, c_over_l = mts_residual(k, c, float(ell), D, m)
            rhs = (delta - gamma) * c_over_l
            slack = rhs - lhs
            if slack >= 0 and (best is None or slack > best[1]):
                best = (float(ell), slack)
        if best is not None:
            ell, slack = best
            return Certificate(kappa=k, c=c, ell=ell, delta=delta,
                               gamma=gamma, D=D,
                               max_speed=D / (k * t_perception),
                               residual=slack)
    return None


def max_safe_speed(delta: float = 0.1, gamma: float = 0.0, D: float = 4.0,
                   t_perception: float = 3.0, m=None) -> float | None:
    cert = find_certificate(delta, gamma, D, t_perception, m)
    return None if cert is None else cert.max_speed
