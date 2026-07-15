"""VLM clients for the Contextual Safety Reasoning module.

Two implementations behind one interface:
  - OllamaVLM:    the real thing (paper setup: 4-bit Gemma 3 27B via Ollama).
                  Runs on the Ubuntu machine (http://localhost:11434).
  - RuleBasedVLM: deterministic stand-in for Windows development and for the
                  Oracle / No-Context baselines. Maps visible semantic
                  classes to predicates with the same contextual logic the
                  VLM is prompted to apply.
"""
from __future__ import annotations

import base64
import io
import json
import time
from abc import ABC, abstractmethod

import numpy as np

from ..predicates import Predicate, SafetyConstraints, parse_vlm_output
from .prompt import SYSTEM_PROMPT


class VLMClient(ABC):
    """Produces contextual safety constraints from an RGB observation."""

    @abstractmethod
    def infer(self, rgb: np.ndarray, visible_classes: list[str] | None = None) -> SafetyConstraints:
        """rgb: HxWx3 uint8. visible_classes: only used by rule-based stand-ins."""


class OllamaVLM(VLMClient):
    """Queries a VLM served by Ollama (paper: gemma3:27b, 4-bit quantized).

    Uses the native /api/chat endpoint with an image attachment.
    """

    def __init__(self, model: str = "gemma3:27b",
                 host: str = "http://localhost:11434",
                 timeout: float = 600.0,
                 temperature: float = 0.0,
                 num_gpu: int | None = None):
        """num_gpu: Ollama option = number of layers offloaded to the GPU.
        Pass 0 to run the model CPU-only — required when the GPU is shared
        with Isaac Sim / SAM3 on an 8 GB card. None = server default."""
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_gpu = num_gpu
        self.last_latency: float | None = None
        self.last_raw: str | None = None

    @staticmethod
    def _encode_image(rgb: np.ndarray) -> str:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def infer(self, rgb: np.ndarray, visible_classes: list[str] | None = None,
              **_ignored) -> SafetyConstraints:
        import requests
        options: dict = {"temperature": self.temperature}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": "Analyze this image and output the safety JSON.",
                 "images": [self._encode_image(rgb)]},
            ],
        }
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        self.last_latency = time.time() - t0
        self.last_raw = r.json()["message"]["content"]
        return parse_vlm_output(self.last_raw)


# --------------------------------------------------------------------------
# Rule-based stand-in (Windows development, Oracle baseline)
# --------------------------------------------------------------------------

# Default contextual rulebook mirroring the paper's scenarios:
#   navigable surfaces -> ON(...) safe;   non-navigable surfaces -> ON(...) unsafe
#   hazard indicators / social entities -> AROUND(...) unsafe (+ NEAR)
#   barrier-forming objects -> BETWEEN(...) unsafe (+ NEAR)
#   any other solid object -> NEAR(...) unsafe
DEFAULT_RULEBOOK = {
    "navigable": {"floor", "sidewalk", "hallway", "road"},
    "non_navigable": {"grass", "flower_bed", "wet_area"},
    "hazard_around": {"wet_floor_sign", "forklift", "person", "hazard_sign"},
    "barrier_between": {"cone", "traffic_cone"},
}


class RuleBasedVLM(VLMClient):
    """Deterministic contextual reasoner over ground-truth visible classes.

    Serves two roles:
      - development stand-in for the VLM on machines without Ollama;
      - the Oracle / No-Context baselines (via `contextual=False`, the
        rulebook degrades to generic pre-deployment rules, cf. Sec. V).
    """

    def __init__(self, rulebook: dict | None = None, contextual: bool = True,
                 min_between_instances: int = 2):
        self.rulebook = rulebook or DEFAULT_RULEBOOK
        self.contextual = contextual
        self.min_between_instances = min_between_instances

    def infer(self, rgb: np.ndarray, visible_classes: list[str] | None = None,
              instance_counts: dict[str, int] | None = None) -> SafetyConstraints:
        if visible_classes is None:
            raise ValueError("RuleBasedVLM requires ground-truth visible_classes")
        rb = self.rulebook
        safe: list[Predicate] = []
        unsafe: list[Predicate] = []
        notes: list[str] = []
        for cls in visible_classes:
            c = cls.lower()
            if c in rb["navigable"]:
                safe.append(Predicate("ON", c))
            elif c in rb["non_navigable"]:
                if self.contextual:
                    unsafe.append(Predicate("ON", c))
                    notes.append(f"{c} is not intended for travel")
                else:
                    # Generic rules cannot tell navigable from non-navigable
                    # terrain: both look traversable.
                    safe.append(Predicate("ON", c))
            elif self.contextual and c in rb["barrier_between"]:
                n = (instance_counts or {}).get(cls, self.min_between_instances)
                if n >= self.min_between_instances:
                    unsafe.append(Predicate("BETWEEN", c))
                    notes.append(f"{c}s arranged to block a path")
                unsafe.append(Predicate("NEAR", c))
            elif self.contextual and c in rb["hazard_around"]:
                unsafe.append(Predicate("AROUND", c))
                unsafe.append(Predicate("NEAR", c))
                notes.append(f"vicinity of {c} is dangerous")
            else:
                unsafe.append(Predicate("NEAR", c))
        return SafetyConstraints(
            safety_logic="; ".join(notes) or "collision avoidance only",
            classes=[c.lower() for c in visible_classes],
            unsafe=unsafe, safe=safe,
        )
