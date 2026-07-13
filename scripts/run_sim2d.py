"""Run one 2D closed-loop episode and (optionally) plot it.

Usage:
  python scripts/run_sim2d.py --scenario cone_barrier --method core --plot
  python scripts/run_sim2d.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core_safety.sim2d.scenarios import SCENARIOS, make_scenario
from core_safety.sim2d.runner import run_episode
from core_safety.sim2d.world import Cylinder, Box, GroundPatch


def plot_episode(scenario, result, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle, Rectangle as MplRect

    fig, ax = plt.subplots(figsize=(9, 6))
    x_min, x_max, y_min, y_max = scenario.workspace

    for p in scenario.world.ground_patches:
        color = {"sidewalk": "#d9d0c1", "grass": "#b8d8a8",
                 "road": "#c0c0c0"}.get(p.cls, "#eeeeee")
        ax.add_patch(MplRect((p.x_min, p.y_min), p.x_max - p.x_min,
                             p.y_max - p.y_min, facecolor=color,
                             edgecolor="none", zorder=0))
        ax.annotate(p.cls, ((p.x_min + p.x_max) / 2, p.y_max - 0.3),
                    fontsize=7, color="gray", ha="center")
    if scenario.world.default_ground == "grass":
        ax.set_facecolor("#b8d8a8")

    for o in scenario.world.objects:
        if isinstance(o, Cylinder):
            ax.add_patch(MplCircle((o.cx, o.cy), o.radius, color="#c44",
                                   zorder=3))
            ax.annotate(o.cls, (o.cx, o.cy + o.radius + 0.15), fontsize=7,
                        ha="center", zorder=4)
        elif isinstance(o, Box):
            ax.add_patch(MplRect((o.x_min, o.y_min), o.wx, o.wy,
                                 color="#a66", zorder=3))
            ax.annotate(o.cls, (o.cx, o.y_max + 0.15), fontsize=7,
                        ha="center", zorder=4)

    # GT unsafe regions (evaluation only): sample membership on a grid.
    xs = np.linspace(x_min, x_max, 220)
    ys = np.linspace(y_min, y_max, 160)
    for label, fn in scenario.world.gt_unsafe_regions:
        mask = np.array([[fn(x, y) for x in xs] for y in ys])
        ax.contourf(xs, ys, mask.astype(float), levels=[0.5, 1.5],
                    colors=["red"], alpha=0.15, zorder=1)

    traj = result.trajectory
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, zorder=5, label="trajectory")
    ax.plot(*traj[0, :2], "go", ms=9, zorder=6, label="start")
    ax.plot(*scenario.goal, "r*", ms=14, zorder=6, label="goal (nominal)")

    status = "SUCCESS" if result.success else "FAILURE"
    ax.set_title(f"{scenario.name} [{scenario.kind}] — {result.method}: {status}"
                 f"  (violated={result.violated}, reached={result.reached_goal},"
                 f" min h={result.min_h:.2f})")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal"); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"plot saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cone_barrier",
                    choices=sorted(SCENARIOS.keys()))
    ap.add_argument("--method", default="core",
                    choices=["core", "oracle", "nocontext", "geometric"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, fac in sorted(SCENARIOS.items()):
            s = fac()
            print(f"{name:22s} [{s.kind:6s}] {s.description}")
        return

    scenario = make_scenario(args.scenario)
    result = run_episode(scenario, method=args.method, record=True)
    print(f"scenario={result.scenario} kind={result.kind} method={result.method}")
    print(f"success={result.success} violated={result.violated}"
          f" ({result.violation_label}) reached_goal={result.reached_goal}"
          f" steps={result.steps} min_h={result.min_h:.3f}")

    if args.plot:
        out = Path(__file__).resolve().parents[1] / "results"
        out.mkdir(exist_ok=True)
        plot_episode(scenario, result,
                     out / f"{args.scenario}_{args.method}.png")


if __name__ == "__main__":
    main()
