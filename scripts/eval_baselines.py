"""Full evaluation matrix: all scenarios x all methods x N seeds (Table I).

Usage:
  python scripts/eval_baselines.py --repeats 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core_safety.sim2d.scenarios import SCENARIOS, make_scenario
from core_safety.sim2d.runner import run_episode

METHODS = ["core", "nocontext", "geometric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="uniform start-position jitter [m]")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default="results/eval_baselines.json")
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for method in args.methods:
        for name in sorted(SCENARIOS.keys()):
            for rep in range(args.repeats):
                rng = np.random.default_rng(1000 * rep + hash(name) % 1000)
                scenario = make_scenario(name)
                r = run_episode(scenario, method=method,
                                start_jitter=args.jitter, rng=rng)
                rows.append({
                    "method": method, "scenario": name, "kind": r.kind,
                    "rep": rep, "success": r.success, "violated": r.violated,
                    "violation": r.violation_label,
                    "reached_goal": r.reached_goal, "steps": r.steps,
                    "min_h": r.min_h if np.isfinite(r.min_h) else None,
                })
                print(f"[{time.time()-t0:6.1f}s] {method:10s} {name:22s} "
                      f"rep{rep} -> {'OK ' if r.success else 'FAIL'}"
                      f"{' (violated: ' + str(r.violation_label) + ')' if r.violated else ''}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    # ---- Table I style summary -------------------------------------------
    print("\n=== Summary (Table I style) ===")
    print(f"{'Method':<12}{'Total':>8}{'Safe':>8}{'Unsafe':>8}")
    for method in args.methods:
        sel = [r for r in rows if r["method"] == method]
        safe = [r for r in sel if r["kind"] == "safe"]
        unsafe = [r for r in sel if r["kind"] == "unsafe"]
        pct = lambda rs: 100.0 * sum(r["success"] for r in rs) / max(len(rs), 1)
        print(f"{method:<12}{pct(sel):>7.1f}%{pct(safe):>7.1f}%{pct(unsafe):>7.1f}%")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
