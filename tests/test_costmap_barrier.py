import numpy as np

from core_safety.grounding.costmap import SemanticCostmap
from core_safety.grounding.barrier import SDFBarrier


def _costmap_with_wall():
    cm = SemanticCostmap(0, 10, 0, 10, resolution=0.2, tau=0.5)
    ys = np.linspace(0.1, 9.9, 200)
    wall = np.stack([np.full_like(ys, 5.0), ys], axis=1)
    open_space = np.stack([np.full_like(ys, 2.0), ys], axis=1)
    cm.add_points(safe_pts=open_space, unsafe_pts=wall)
    return cm


def test_p_safe_and_threshold():
    cm = _costmap_with_wall()
    ix, iy = cm.state_to_cell(5.0, 5.0)
    assert not cm.safe_grid()[ix, iy]
    ix, iy = cm.state_to_cell(2.0, 5.0)
    assert cm.safe_grid()[ix, iy]
    # Unobserved cells default to safe.
    ix, iy = cm.state_to_cell(9.0, 9.0)
    assert cm.safe_grid()[ix, iy]


def test_mixed_votes_majority():
    cm = SemanticCostmap(0, 2, 0, 2, resolution=1.0, tau=0.5)
    pt = np.array([[0.5, 0.5]])
    cm.add_points(np.repeat(pt, 3, axis=0), np.repeat(pt, 1, axis=0))
    assert cm.p_safe()[0, 0] == 0.75
    assert cm.safe_grid()[0, 0]


def test_sdf_sign_and_gradient():
    cm = _costmap_with_wall()
    b = SDFBarrier(cm.safe_grid(), cm.x_min, cm.y_min, cm.res)
    assert b.h(2.0, 5.0) > 0            # in safe region
    assert b.h(5.0, 5.0) <= 0           # on the unsafe wall
    # Approaching the wall from the left: h decreases.
    assert b.h(2.0, 5.0) > b.h(4.0, 5.0)
    # Gradient points away from the wall (negative x direction).
    g = b.grad_h(4.0, 5.0)
    assert g[0] < 0
    assert abs(np.linalg.norm(g) - 1.0) < 1e-6


def test_all_safe_barrier_inactive():
    grid = np.ones((10, 10), dtype=bool)
    b = SDFBarrier(grid, 0, 0, 0.2)
    assert np.isinf(b.h(1.0, 1.0))
