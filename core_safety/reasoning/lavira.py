"""LaViRA-mode reasoning, single-call variant ("B").

Derived from the Uni-LaViRA real-robot pipeline (Ding et al.,
arXiv:2510.19655 / 2605.27582; prompt wording our own — the upstream code
is CC BY-NC-SA and nothing is copied verbatim). Upstream splits each cycle
into LA (panorama -> direction) and VA (post-turn view -> bbox); with a
mostly-CPU local model every extra call costs minutes, so this variant
merges the cycle into ONE call:

  4 panoramic views (rgb + depth + pose stored per view)
    -> updated markdown TODO + NAVIGATE/STOP + chosen view + bbox in it
    -> bbox bottom-center + the STORED depth/pose of that view -> subgoal
    -> A* + CBF drive there (no physical re-turn, no second VLM call).

Upstream properties kept: markdown TODO memory rewritten every cycle,
"Step i" navigation history, the max<=1000 bbox normalization heuristic,
and the always-move principle — no bbox -> view center pixel, no valid
depth -> 1.0 m along the view direction. Collision safety is the CBF's.

Additions beyond upstream (both odometry facts, not model guesses),
motivated by observed direction loss between slow cycles:
  - an ORIENTATION line telling the model how far it moved and where its
    previous subgoal now lies relative to the CURRENT heading;
  - the chosen direction recorded in each history step.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from ..grounding.projection import PinholeCamera

# View/direction layout, counter-clockwise from the current heading.
DIRECTIONS: dict[str, float] = {
    "front": 0.0,
    "left": np.pi / 2,
    "back": np.pi,
    "right": -np.pi / 2,
}
VIEW_ORDER = ("front", "left", "back", "right")

_VIEW_LIST = ("1=front(0 deg), 2=left(+90 deg), 3=back(180 deg), "
              "4=right(-90 deg)")

_SECTORS = ("front", "front-left", "left", "back-left",
            "back", "back-right", "right", "front-right")


def relative_sector(bearing: float) -> str:
    """Bucket an ego-relative bearing [rad] into one of 8 sector names."""
    idx = int(np.round((bearing % (2 * np.pi)) / (np.pi / 4))) % 8
    return _SECTORS[idx]


def orientation_text(moved_m: float, prev_target: str,
                     prev_sector: str | None) -> str:
    """Odometry-derived continuity line for the prompt (never guessed)."""
    if prev_sector is None:
        return ("ORIENTATION: this is the first cycle. The panorama below "
                "is relative to your CURRENT heading.")
    return (f"ORIENTATION: since the previous panorama you moved "
            f"{moved_m:.1f} m. Your previous subgoal ('{prev_target}') now "
            f"lies to your {prev_sector}. The panorama below is relative "
            "to your CURRENT heading, not the previous one.")


def history_text(steps: list[dict]) -> str:
    """Navigation-history block: 'Step i: <description> -> <target>'."""
    out = "Navigation History:\n"
    if steps:
        for i, s in enumerate(steps, 1):
            out += (f"Step {i}: {s.get('description', 'Unknown')} "
                    f"-> {s.get('target', 'Unknown')}\n")
    else:
        out += "No history yet.\n"
    return out


def plan_prompt(instruction: str, todo: str, steps: list[dict],
                orientation: str, width: int, height: int) -> str:
    """The single per-cycle prompt (text part; 4 images attached)."""
    return f"""Navigation Task: "{instruction}"

{history_text(steps)}
{orientation}

**ROLE**: You are the navigator of a mobile robot: strategy and visual
grounding in a single step.
**MISSION**: "{instruction}"

**Current TODO List** (create it now if empty):
{todo or '(empty)'}

The 4 attached images are a panorama from your current position, in this
order: {_VIEW_LIST}.

**Your tasks**:
1. Update the TODO list. Markdown, one item per line:
   "- [ ] pending" / "- [x] done".
2. Decide the action:
   - "STOP" only when the mission is complete (you are AT the target).
   - "NAVIGATE" otherwise.
3. If NAVIGATE: choose the ONE image whose direction best advances the
   mission, and put a bounding box around ONE concrete thing that is
   ACTUALLY VISIBLE in that image for the robot to drive toward — an
   object, a floor spot, the free floor at the edge of an obstacle row.
   Never box an abstract idea ("the path", "beyond the wall") and never
   box something you cannot see. If the mission target itself is visible
   in any image, always choose it.

The bbox coordinates are pixels [0..{width}, 0..{height}] in the CHOSEN
image only.

**Reply with one JSON object** (no markdown fences, nothing else):
{{
    "progress_analysis": "...(30 words or fewer)...",
    "reasoning": "...(1-2 sentences)...",
    "updated_todo_list": "...",
    "action": "NAVIGATE" | "STOP",
    "view": 1 | 2 | 3 | 4,
    "target": "the visible thing you boxed",
    "bbox_2d": [x1, y1, x2, y2]
}}
"""


@dataclass
class PlanResult:
    ok: bool                      # False = model/parse failure (fallback)
    action: str                   # NAVIGATE | STOP
    view: str                     # key of DIRECTIONS
    bbox_px: tuple[int, int, int, int] | None   # pixels in the chosen view
    target: str
    updated_todo: str
    progress_analysis: str = ""
    reasoning: str = ""


def _extract_json(text: str) -> dict | None:
    """Fenced ```json block first, else outermost braces."""
    m = re.search(r"```(?:json)?\s*\n?((?:.|\n)*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            raw = text[s:e + 1]
    if raw is None:
        return None
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw.strip()))
    except json.JSONDecodeError:
        return None


def _parse_view(v) -> str:
    """Accept 1-4 (upstream-style index) or a direction name."""
    try:
        i = int(v)
        if 1 <= i <= 4:
            return VIEW_ORDER[i - 1]
    except (TypeError, ValueError):
        pass
    name = str(v).strip().lower()
    return name if name in DIRECTIONS else "front"


def parse_plan_output(text: str, width: int, height: int,
                      prev_todo: str = "") -> PlanResult:
    """Parse the combined reply; degrade to NAVIGATE / front / no bbox.

    The fallback still moves (view center -> ~straight ahead): uncertainty
    means "move and look again", never "freeze"."""
    fallback = PlanResult(False, "NAVIGATE", "front", None, "open space",
                          prev_todo, "Fallback", "Model error")
    obj = _extract_json(text)
    if obj is None:
        return fallback
    action = str(obj.get("action", "")).strip().upper()
    if action not in ("NAVIGATE", "STOP"):
        action = "NAVIGATE"
    bbox_px = None
    b = obj.get("bbox_2d")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            b = [float(v) for v in b]
            if max(b) <= 1000:                    # [0,1000] -> pixels
                b = [b[0] / 1000 * width, b[1] / 1000 * height,
                     b[2] / 1000 * width, b[3] / 1000 * height]
            bbox_px = (int(max(0, min(width, b[0]))),
                       int(max(0, min(height, b[1]))),
                       int(max(0, min(width, b[2]))),
                       int(max(0, min(height, b[3]))))
        except (TypeError, ValueError):
            bbox_px = None
    return PlanResult(
        ok=True, action=action,
        view=_parse_view(obj.get("view")),
        bbox_px=bbox_px,
        target=str(obj.get("target", "")).strip() or "open space",
        updated_todo=str(obj.get("updated_todo_list", "")).strip()
        or prev_todo,
        progress_analysis=str(obj.get("progress_analysis", "")),
        reasoning=str(obj.get("reasoning", "")),
    )


# --------------------------------------------------------------------------
# RA grounding: nav pixel -> world goal (always succeeds)
# --------------------------------------------------------------------------

def nav_pixel(bbox_px: tuple[int, int, int, int] | None,
              width: int, height: int) -> tuple[int, int]:
    """Navigation pixel: bbox bottom-center, else the image center."""
    if bbox_px is not None:
        x1, y1, x2, y2 = bbox_px
        if x2 > x1 and y2 > y1:
            return int((x1 + x2) / 2), int(y2)
    return width // 2, height // 2


def pixel_to_goal(u: int, v: int, depth: np.ndarray | None,
                  cam: PinholeCamera, view_pose: np.ndarray,
                  window: int = 7) -> np.ndarray:
    """Project one pixel + median window depth into a world (x, y) goal.

    view_pose is the robot pose AT CAPTURE TIME of the chosen view (the
    subgoal is grounded from the stored panorama frame, not a re-capture).
    Never fails: with no valid depth the goal defaults to 1.0 m along the
    view direction — a wrong-but-moving guess beats freezing; the CBF
    layer owns collision safety."""
    d = None
    if depth is not None and depth.size:
        h, w = depth.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        r = window // 2
        patch = depth[max(0, v - r):min(h, v + r + 1),
                      max(0, u - r):min(w, u + r + 1)]
        finite = patch[np.isfinite(patch) & (patch > 0)]
        if finite.size:
            d = float(np.median(finite))
    x, y, th = view_pose
    fwd = np.array([np.cos(th), np.sin(th)])
    if d is None:
        return np.array([x, y]) + 1.0 * fwd
    xc = d * (u - cam.cx) / cam.fx            # camera x: right
    right = np.array([np.sin(th), -np.cos(th)])
    return np.array([x, y]) + d * fwd + xc * right


# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------

class OllamaLaViRA:
    def __init__(self, model: str = "gemma3:27b",
                 host: str = "http://localhost:11434",
                 timeout: float = 600.0, num_gpu: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.last_raw: str | None = None

    def plan(self, views: list[np.ndarray], instruction: str, todo: str,
             steps: list[dict], orientation: str) -> PlanResult:
        """One combined LA+VA decision from the 4 panorama views."""
        import requests
        from .vlm_client import OllamaVLM
        h, w = views[0].shape[:2]
        options: dict = {"temperature": 0.0}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [{"role": "user",
                          "content": plan_prompt(instruction, todo, steps,
                                                 orientation, w, h),
                          "images": [OllamaVLM._encode_image(v)
                                     for v in views]}],
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload,
                              timeout=self.timeout)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Ollama HTTP {r.status_code}: {r.text[:300]}")
            self.last_raw = r.json()["message"]["content"]
            return parse_plan_output(self.last_raw, w, h, prev_todo=todo)
        except Exception as e:        # network/model error -> keep moving
            print(f"[lavira] plan call failed: {e}")
            return parse_plan_output("", w, h, prev_todo=todo)
