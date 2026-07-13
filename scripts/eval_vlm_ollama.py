"""Evaluate VLM contextual safety reasoning via Ollama (Table II / III).

Runs the paper's system prompt against a folder of test images and compares
predicted predicates to per-image labels. Run this ON THE UBUNTU MACHINE
(Ollama at http://localhost:11434).

Dataset layout:
    assets/vlm_eval/
        images/xxx.png
        labels.json    # {"xxx.png": {"safe": ["ON(floor)"],
                       #              "unsafe": ["AROUND(wet_floor_sign)"]}}

Usage:
    python scripts/eval_vlm_ollama.py --data assets/vlm_eval --model gemma3:27b
    python scripts/eval_vlm_ollama.py --data assets/vlm_eval --smoke  # 3 images

A prediction is counted correct (per the paper's protocol) if the predicted
predicate set matches the labeled set; we report safe-set and unsafe-set
accuracy separately, plus latency and parse failures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core_safety.predicates import parse_predicates
from core_safety.reasoning.vlm_client import OllamaVLM


def load_dataset(root: Path):
    labels = json.loads((root / "labels.json").read_text())
    items = []
    for name, lab in sorted(labels.items()):
        img_path = root / "images" / name
        if img_path.exists():
            items.append((img_path, set(map(str, parse_predicates(", ".join(lab.get("safe", []))))),
                          set(map(str, parse_predicates(", ".join(lab.get("unsafe", [])))))))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="assets/vlm_eval")
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--smoke", action="store_true", help="only 3 images")
    ap.add_argument("--out", default="results/vlm_eval.json")
    args = ap.parse_args()

    from PIL import Image

    root = Path(args.data)
    items = load_dataset(root)
    if args.smoke:
        items = items[:3]
    if not items:
        print(f"No labeled images found under {root}. See docstring for layout.")
        return

    vlm = OllamaVLM(model=args.model, host=args.host)
    rows = []
    for img_path, gt_safe, gt_unsafe in items:
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        t0 = time.time()
        try:
            pred = vlm.infer(rgb)
            parsed = True
            p_safe = {str(p) for p in pred.safe}
            p_unsafe = {str(p) for p in pred.unsafe}
        except Exception as e:
            parsed = False
            p_safe, p_unsafe = set(), set()
            print(f"  parse failure on {img_path.name}: {e}")
        latency = time.time() - t0
        rows.append({
            "image": img_path.name, "parsed": parsed, "latency": latency,
            "safe_correct": p_safe == gt_safe,
            "unsafe_correct": p_unsafe == gt_unsafe,
            "pred_safe": sorted(p_safe), "pred_unsafe": sorted(p_unsafe),
            "gt_safe": sorted(gt_safe), "gt_unsafe": sorted(gt_unsafe),
            "raw": vlm.last_raw,
        })
        print(f"{img_path.name}: safe={'OK' if rows[-1]['safe_correct'] else 'NG'} "
              f"unsafe={'OK' if rows[-1]['unsafe_correct'] else 'NG'} "
              f"({latency:.1f}s)")

    n = len(rows)
    safe_acc = 100.0 * sum(r["safe_correct"] for r in rows) / n
    unsafe_acc = 100.0 * sum(r["unsafe_correct"] for r in rows) / n
    lat = np.mean([r["latency"] for r in rows])
    print(f"\nModel {args.model}: safe {safe_acc:.1f}% | unsafe {unsafe_acc:.1f}%"
          f" | latency {lat:.1f}s | parse failures "
          f"{sum(not r['parsed'] for r in rows)}/{n}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
