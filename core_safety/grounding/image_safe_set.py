"""Image-space safe set construction (Eq. 2).

    I_safe = ( U_j I_safe_j )  \\  ( U_k I_unsafe_k )

Returns both the safe partition and the unsafe union so downstream code can
accumulate per-cell safe/unsafe observation counts (Eq. 3).
"""
from __future__ import annotations

import numpy as np

from ..predicates import SafetyConstraints
from .operators import predicate_to_mask


def build_image_safe_set(constraints: SafetyConstraints,
                         seg: dict[str, np.ndarray],
                         around_kernel_px: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (safe_mask, unsafe_mask), each HxW bool.

    safe_mask is Eq. 2 exactly; unsafe_mask is the union of unsafe pixel
    sets. Pixels in neither set are unobserved (no vote either way).
    """
    if not seg:
        raise ValueError("empty segmentation")
    shape = next(iter(seg.values())).shape
    safe_union = np.zeros(shape, dtype=bool)
    unsafe_union = np.zeros(shape, dtype=bool)
    for p in constraints.safe:
        safe_union |= predicate_to_mask(p, seg, around_kernel_px)
    for p in constraints.unsafe:
        unsafe_union |= predicate_to_mask(p, seg, around_kernel_px)
    safe_mask = safe_union & ~unsafe_union
    return safe_mask, unsafe_union
