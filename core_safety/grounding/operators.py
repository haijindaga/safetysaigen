"""Spatial operators: predicate -> pixel set (Sec. IV-B / Appendix A-A).

  ON(class):      pixels of the class (reserved for traversable surfaces).
  NEAR(class):    pixels of the class (identical implementation to ON).
  AROUND(class):  morphological dilation of the class mask
                  (paper: kernel size 50 px).
  BETWEEN(class): convex hull around the instances of the class.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..predicates import Predicate

AROUND_KERNEL_PX = 50


def predicate_to_mask(pred: Predicate, seg: dict[str, np.ndarray],
                      around_kernel_px: int = AROUND_KERNEL_PX) -> np.ndarray:
    """Compute the pixel set of a predicate from class segmentation masks."""
    base = None
    for name, mask in seg.items():
        if name.lower() == pred.cls:
            base = mask.astype(np.uint8)
            break
    if base is None or not base.any():
        shape = next(iter(seg.values())).shape if seg else (0, 0)
        return np.zeros(shape, dtype=bool)

    if pred.op in ("ON", "NEAR"):
        return base.astype(bool)

    if pred.op == "AROUND":
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (around_kernel_px, around_kernel_px))
        return cv2.dilate(base, k).astype(bool)

    if pred.op == "BETWEEN":
        pts = cv2.findNonZero(base)
        if pts is None or len(pts) < 3:
            return base.astype(bool)
        hull = cv2.convexHull(pts)
        out = np.zeros_like(base)
        cv2.fillConvexPoly(out, hull, 1)
        return out.astype(bool)

    raise ValueError(f"Unknown operator {pred.op}")
