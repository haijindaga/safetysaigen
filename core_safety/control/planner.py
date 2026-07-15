"""Grid A* planner over the semantic costmap.

The safety filter is a projection, not a planner: faced with a concave
unsafe region it stalls at the boundary. Planning the nominal path on the
costmap's safe grid (unobserved = optimistically traversable, unsafe cells
inflated by one cell) lets the robot route AROUND grounded constraints
while the CBF remains the safety backstop. The paper leaves the nominal
planner open ("agnostic to the robot's planner"), so this stays outside
the reproduction-faithful core.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

_NBRS = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
         (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]


def _astar(grid: np.ndarray, start: tuple[int, int],
           goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    nx, ny = grid.shape
    if not grid[start]:
        # Allow starting from an (inflated-)unsafe cell: escape is the plan.
        pass
    h = lambda c: np.hypot(c[0] - goal[0], c[1] - goal[1])
    open_q = [(h(start), 0.0, start)]
    came: dict = {start: None}
    g = {start: 0.0}
    while open_q:
        _, gc, cur = heapq.heappop(open_q)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if gc > g.get(cur, np.inf):
            continue
        for dx, dy, w in _NBRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or not grid[nb]:
                continue
            ng = gc + w
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(open_q, (ng + h(nb), ng, nb))
    return None


def plan_path(costmap, start_xy, goal_xy, inflate_cells: int = 1,
              decimate: int = 3) -> list[tuple[float, float]] | None:
    """World-frame waypoint list from start to goal, or None if no path."""
    safe = costmap.safe_grid()
    if inflate_cells > 0:
        safe = ~ndimage.binary_dilation(~safe, iterations=inflate_cells)
    start = costmap.state_to_cell(*start_xy)
    goal = costmap.state_to_cell(*goal_xy)
    if not safe[goal]:
        # Goal sits in (inflated) unsafe space: aim at nearest safe cell.
        idx = np.argwhere(safe)
        if len(idx) == 0:
            return None
        d = np.linalg.norm(idx - np.array(goal), axis=1)
        goal = tuple(idx[int(np.argmin(d))])
    cells = _astar(safe, start, goal)
    if cells is None:
        return None
    xs, ys = costmap.cell_centers()
    cells = cells[::decimate] + [cells[-1]]
    return [(float(xs[i]), float(ys[j])) for i, j in cells]
