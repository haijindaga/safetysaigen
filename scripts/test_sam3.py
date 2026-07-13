"""Self-test for the SAM3 segmenter (run on the Ubuntu GPU machine).

Downloads facebook/sam3 on first run, segments a synthetic scene rendered
by the 2D simulator, and reports per-class mask coverage vs ground truth.

Usage:
    python scripts/test_sam3.py
    python scripts/test_sam3.py --device cpu     # if VRAM is tight
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--image", default=None,
                    help="optional real image path (else synthetic scene)")
    ap.add_argument("--classes", nargs="+",
                    default=["traffic cone", "floor", "person"])
    args = ap.parse_args()

    from core_safety.grounding.sam3_segmenter import SAM3Segmenter

    if args.image:
        from PIL import Image
        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        gt = None
    else:
        from core_safety.sim2d.camera import RaycastCamera
        from core_safety.sim2d.scenarios import make_scenario
        scenario = make_scenario("cone_barrier")
        camera = RaycastCamera()
        rgb, depth, labels = camera.render(scenario.world,
                                           np.array([3.0, 0.3, 0.0]))
        gt = (labels, scenario.world)

    print("loading SAM3 (first run downloads ~3.4 GB)...")
    t0 = time.time()
    seg = SAM3Segmenter(device=args.device)
    print(f"loaded in {time.time()-t0:.1f}s on {seg.device}")

    t0 = time.time()
    masks = seg.segment(rgb, args.classes)
    dt = time.time() - t0
    h, w = rgb.shape[:2]
    print(f"segmented {len(args.classes)} classes in {dt:.2f}s")
    for cls, m in masks.items():
        print(f"  {cls:15s}: {m.sum():7d} px ({100.0*m.sum()/(h*w):5.1f}%)")

    if gt is not None:
        labels, world = gt
        ids = world.class_ids()
        cone_gt = labels == ids["cone"]
        cone_pred = masks.get("traffic cone", np.zeros_like(cone_gt))
        inter = (cone_gt & cone_pred).sum()
        union = (cone_gt | cone_pred).sum()
        if union:
            print(f"\ncone IoU vs ground truth: {inter/union:.2f} "
                  "(synthetic render — treat as smoke test, not a benchmark)")


if __name__ == "__main__":
    main()
