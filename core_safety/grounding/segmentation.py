"""Open-vocabulary segmentation front-end.

The paper uses SAM 3 ("Segment Anything with Concepts") to localize each
semantic class named by the VLM. Implementations:

  - GroundTruthSegmenter: uses simulator-provided semantic segmentation
    (2D sim on Windows, or Isaac Sim's semantic ground truth). Lets us
    verify grounding/control in isolation from perception errors.
  - SAM3Segmenter (in sam3_segmenter.py): real SAM3, Ubuntu/GPU only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    @abstractmethod
    def segment(self, rgb: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
        """Return {class_name: HxW bool mask} for each requested class.

        Classes that are not found map to an all-False mask.
        """


class GroundTruthSegmenter(Segmenter):
    """Wraps a ground-truth label image (HxW int) + id->name mapping."""

    def __init__(self):
        self.label_img: np.ndarray | None = None
        self.id_to_name: dict[int, str] = {}

    def update(self, label_img: np.ndarray, id_to_name: dict[int, str]):
        self.label_img = label_img
        self.id_to_name = {i: n.lower() for i, n in id_to_name.items()}

    def segment(self, rgb: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
        assert self.label_img is not None, "call update() first"
        h, w = self.label_img.shape
        name_to_ids: dict[str, list[int]] = {}
        for i, n in self.id_to_name.items():
            name_to_ids.setdefault(n, []).append(i)
        out: dict[str, np.ndarray] = {}
        for cls in classes:
            c = cls.lower()
            mask = np.zeros((h, w), dtype=bool)
            for i in name_to_ids.get(c, []):
                mask |= self.label_img == i
            out[cls] = mask
        return out
