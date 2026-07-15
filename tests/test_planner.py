import numpy as np

from core_safety.grounding.costmap import SemanticCostmap
from core_safety.control.planner import plan_path


def test_astar_routes_around_wall_with_gap_at_edge():
    cm = SemanticCostmap(0, 10, 0, 10, resolution=0.5)
    ys = np.linspace(0.25, 7.75, 60)              # wall x=5, gap at top
    cm.add_points(safe_pts=np.zeros((0, 2)),
                  unsafe_pts=np.stack([np.full_like(ys, 5.0), ys], 1))
    path = plan_path(cm, (1.0, 5.0), (9.0, 5.0))
    assert path is not None
    ys_at_wall = [y for x, y in path if 4.0 <= x <= 6.0]
    assert ys_at_wall and min(ys_at_wall) > 7.5    # goes through the gap


def test_astar_direct_when_clear():
    cm = SemanticCostmap(0, 10, 0, 10, resolution=0.5)
    path = plan_path(cm, (1.0, 5.0), (9.0, 5.0))
    assert path is not None and len(path) >= 2
