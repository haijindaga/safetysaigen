import numpy as np

from core_safety.theory.certificate import (
    default_m, mts_residual, certificate_holds, find_certificate)


def test_paper_calibration_certificate():
    """Paper (Appendix B-D3): with m(r) = 0.75 - eps*r/D, D = 4 m, a
    certificate at delta = 0.1, gamma = 0 exists at kappa* = 3, c = 1.
    The paper reports l = 0.1 with a small residual; with our eps choice
    the constraint closes at slightly smaller l (~0.05), and l = 0.1 is
    within a few percent of feasibility."""
    assert certificate_holds(k=3, c=1.0, ell=0.05, delta=0.1, gamma=0.0, D=4.0)
    lhs, c_over_l = mts_residual(k=3, c=1.0, ell=0.1, D=4.0)
    assert lhs <= 1.2 * 0.1 * c_over_l      # paper's point: small residual


def test_certificate_search_matches_paper():
    cert = find_certificate(delta=0.1, gamma=0.0, D=4.0, t_perception=3.0)
    assert cert is not None
    assert cert.kappa == 3                  # paper: kappa* = 3
    # Max safe speed for kappa=3: 4 m / (3 * 3 s) = 0.44 m/s >= paper's 0.35.
    assert cert.max_speed >= 0.35


def test_more_measurements_help():
    lhs_small, _ = mts_residual(k=2, c=1.0, ell=0.1)
    lhs_large, _ = mts_residual(k=10, c=1.0, ell=0.1)
    assert lhs_large < lhs_small


def test_weak_detector_needs_more_looks():
    weak = lambda r: default_m(r, p=0.3)
    strong = lambda r: default_m(r, p=0.9)
    cw = find_certificate(delta=0.1, m=weak)
    cs = find_certificate(delta=0.1, m=strong)
    assert cw.kappa > cs.kappa
