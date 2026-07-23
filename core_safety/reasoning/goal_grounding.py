"""Language-goal grounding: mission target -> VA bbox -> world goal.

Implements the sparse-grounding idea of LaViRA / Uni-LaViRA (Ding et al.,
arXiv:2510.19655 / 2605.27582): instead of segmenting the goal object, a
multimodal LLM eyeballs a single bounding box in the CURRENT view ("Vision
Action"), and the box's bottom-center pixel plus registered depth is
back-projected into one world-frame goal point. The nominal planner (A*)
drives there; the CBF layer still has the final word on every command, so
a hallucinated box can waste time but never cause a collision.

Design notes (own implementation, no upstream code):
  - The VA can only box what is VISIBLE. If the target is not in view it
    must say so (`visible: false`); exploration toward unseen space stays
    the job of the extended-mode behaviors (INVESTIGATE / STOP_AND_SCAN).
  - Coordinates are requested normalized to [0, 1000] regardless of image
    size: unlike Qwen, gemma-class models have no fixed bbox convention,
    so pinning the range in the prompt removes the pixel-vs-normalized
    ambiguity. Values are de-normalized and clamped here.
  - Guards against hallucination: `visible` self-report, finite-depth
    check on the contact point, and a height gate (the bottom of a real,
    grounded object must sit near the floor; a box painted onto a wall
    or a shelf projects too high and is rejected).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from ..grounding.projection import PinholeCamera

VA_SYSTEM_PROMPT = """You are the visual grounding module of a mobile robot.
You receive one camera image and the description of a navigation target.
Decide whether the described target is ACTUALLY VISIBLE in this image.

Rules:
- Only report the target visible if you can point at it in THIS image.
  Never guess a location for something you cannot see.
- Report the tightest bounding box around the target.
- Coordinates MUST be normalized to the range [0, 1000] on both axes:
  [x1, y1, x2, y2], where (0,0) is the top-left corner and
  (1000,1000) the bottom-right corner of the image.

Answer with a single flat JSON object and nothing else:
{
"visible": true or false,
"bbox_2d": [x1, y1, x2, y2] or null,
"reasoning": "one short sentence"
}"""


def va_user_prompt(target: str) -> str:
    """User message accompanying the image for one VA query."""
    return (f'Navigation target: "{target}".\n'
            "Is this target visible in the image? Output the JSON.")


@dataclass
class VAResult:
    visible: bool
    bbox_px: tuple[int, int, int, int] | None   # pixel coords, clamped
    reasoning: str = ""


def parse_va_output(text: str, width: int, height: int) -> VAResult:
    """Parse a VA JSON reply and de-normalize [0,1000] -> pixel coords.

    Tolerates markdown fences and surrounding prose (same policy as
    parse_vlm_output). Any malformed reply degrades to `not visible` —
    goal grounding is best-effort and must never raise into the loop.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return VAResult(False, None, "no JSON in VA output")
    try:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
    except json.JSONDecodeError as e:
        return VAResult(False, None, f"bad JSON: {e}")
    reasoning = str(obj.get("reasoning", ""))
    visible = obj.get("visible")
    bbox = obj.get("bbox_2d")
    if visible is not True or not isinstance(bbox, (list, tuple)) \
            or len(bbox) != 4:
        return VAResult(False, None, reasoning)
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return VAResult(False, None, reasoning)
    if x2 <= x1 or y2 <= y1:
        return VAResult(False, None, reasoning)
    px = (int(np.clip(x1 / 1000 * width, 0, width - 1)),
          int(np.clip(y1 / 1000 * height, 0, height - 1)),
          int(np.clip(x2 / 1000 * width, 0, width - 1)),
          int(np.clip(y2 / 1000 * height, 0, height - 1)))
    return VAResult(True, px, reasoning)


def bbox_to_goal(bbox_px: tuple[int, int, int, int], depth: np.ndarray,
                 cam: PinholeCamera, robot_pose: np.ndarray,
                 min_range: float = 0.15, max_range: float = 6.0,
                 max_contact_height: float = 0.8,
                 window: int = 7) -> np.ndarray | None:
    """Bottom-center of the box + depth -> world (x, y) goal, or None.

    The bottom edge of the box is where the object meets the floor, so its
    depth is the object's ground distance (top pixels may see past a short
    object). Depth is the median of finite values in a small window around
    the contact pixel. Returns None when the depth is missing/out of range
    or the contact point sits higher than `max_contact_height` above the
    ground (a box on a wall or shelf — not a drivable goal).
    """
    x1, y1, x2, y2 = bbox_px
    u = int((x1 + x2) / 2)
    v = int(y2)
    h, w = depth.shape[:2]
    r = window // 2
    patch = depth[max(0, v - r):min(h, v + r + 1),
                  max(0, u - r):min(w, u + r + 1)]
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return None
    d = float(np.median(finite))
    if not (min_range <= d <= max_range):
        return None
    # Height of the contact point above ground (camera y points down):
    z_world = cam.mount_height - d * (v - cam.cy) / cam.fy
    if z_world > max_contact_height:
        return None
    # Same pinhole geometry as pixels_to_world, for a single pixel.
    xc = d * (u - cam.cx) / cam.fx        # camera x: right
    x, y, th = robot_pose
    fwd = np.array([np.cos(th), np.sin(th)])
    right = np.array([np.sin(th), -np.cos(th)])
    return np.array([x, y]) + d * fwd + xc * right


class OllamaVA:
    """One-shot VA query against an Ollama-served multimodal model.

    Kept separate from OllamaVLM: different system prompt, different JSON
    contract, and callers piggyback it on the thinking-layer cycle.
    """

    def __init__(self, model: str = "gemma3:27b",
                 host: str = "http://localhost:11434",
                 timeout: float = 600.0, num_gpu: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.last_raw: str | None = None

    def locate(self, rgb: np.ndarray, target: str) -> VAResult:
        import requests
        from .vlm_client import OllamaVLM
        options: dict = {"temperature": 0.0}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": VA_SYSTEM_PROMPT},
                {"role": "user", "content": va_user_prompt(target),
                 "images": [OllamaVLM._encode_image(rgb)]},
            ],
        }
        r = requests.post(f"{self.host}/api/chat", json=payload,
                          timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        self.last_raw = r.json()["message"]["content"]
        h, w = rgb.shape[:2]
        return parse_va_output(self.last_raw, w, h)
