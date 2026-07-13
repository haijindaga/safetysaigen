"""Generate a labeled VLM-evaluation dataset from the 2D simulator.

Renders observations from several poses in each scenario and writes the
expected predicate labels (from the contextual rulebook). This gives an
immediate, self-contained dataset for scripts/eval_vlm_ollama.py; replace
or extend it later with Isaac Sim screenshots or real photos.

Usage:
    python scripts/make_vlm_dataset.py --out assets/vlm_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core_safety.reasoning.vlm_client import RuleBasedVLM, DEFAULT_RULEBOOK
from core_safety.sim2d.camera import RaycastCamera
from core_safety.sim2d.runner import _visible_classes
from core_safety.sim2d.scenarios import SCENARIOS, make_scenario

POSES = [(0.0, 0.0, 0.0), (2.0, 0.5, 0.0), (3.5, -0.5, 0.1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/vlm_eval")
    args = ap.parse_args()

    from PIL import Image

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    camera = RaycastCamera()
    oracle = RuleBasedVLM(DEFAULT_RULEBOOK, contextual=True)

    labels = {}
    for name in sorted(SCENARIOS.keys()):
        scenario = make_scenario(name)
        sx, sy, sth = scenario.start
        for i, (dx, dy, dth) in enumerate(POSES):
            pose = np.array([sx + dx, sy + dy, sth + dth])
            rgb, depth, lab = camera.render(scenario.world, pose)
            visible = _visible_classes(lab, scenario.world)
            gt = oracle.infer(rgb, visible_classes=visible,
                              instance_counts=scenario.world.instance_counts())
            fname = f"{name}_{i}.png"
            Image.fromarray(rgb).save(out / "images" / fname)
            labels[fname] = {"safe": [str(p) for p in gt.safe],
                             "unsafe": [str(p) for p in gt.unsafe]}
            print(f"{fname}: safe={labels[fname]['safe']} "
                  f"unsafe={labels[fname]['unsafe']}")

    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"\n{len(labels)} labeled images -> {out}")


if __name__ == "__main__":
    main()
