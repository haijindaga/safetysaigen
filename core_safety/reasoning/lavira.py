"""LaViRA-mode reasoning: faithful port of the Uni-LaViRA LA/VA contract.

Structure, JSON fields, fallbacks, and control flow mirror the upstream
real-robot pipeline (Ding et al., arXiv:2510.19655 / 2605.27582) as closely
as our platform allows; the prompt WORDING is our own (the upstream code is
CC BY-NC-SA, so nothing is copied verbatim). Platform adaptations, kept
deliberately minimal:

  - 4 panoramic views at 90-deg increments (the habitat variant's layout;
    the real robot sweeps 7 views with its arms, which we cannot do) ->
    directions are front / left / back / right.
  - RA is our A* + CBF stack instead of iPlanner. The CBF arbitrates every
    command; LaViRA itself ships no safety layer.

The key upstream property this port preserves: EVERY cycle produces a
motion target. No bbox -> use the image center pixel; no valid depth ->
assume 1.0 m straight ahead. Uncertainty means "move and look again",
never "freeze". (Collision safety is the CBF's job, not the planner's.)

Memory, as upstream: a markdown TODO checklist created once from the
starting panorama and rewritten by every strategic call, plus a
"Step i: <progress> -> <target>" navigation history.
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

_VIEW_LIST = "front(0 deg), left(+90 deg), back(180 deg), right(-90 deg)"


# --------------------------------------------------------------------------
# Prompt builders (own wording, upstream structure)
# --------------------------------------------------------------------------

def initial_todo_prompt(instruction: str) -> str:
    """Leading text of the one-shot initial TODO generation (before frames)."""
    return (f'Instruction: "{instruction}"\n\n'
            "The attached images are panoramic views from the starting "
            f"position, in this order: {_VIEW_LIST}.")


def initial_todo_suffix() -> str:
    """Trailing directive of the initial TODO generation (after frames)."""
    return ("Write a dynamic checklist of steps to carry out the "
            "instruction.\n"
            "Markdown format, one item per line: - [ ] step description\n"
            "Return ONLY the checklist, no JSON, no other text.")


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


def strategic_task_text(instruction: str, steps: list[dict]) -> str:
    """Task + history text placed BEFORE the panorama frames (LA call)."""
    return (f'Navigation Task: "{instruction}"\n\n'
            f"History:\n{history_text(steps)}")


def strategic_prompt(instruction: str, todo: str) -> str:
    """Strategic decision body placed AFTER the panorama frames (LA call)."""
    return f"""
**ROLE**: You are the robot's strategic navigator.
**MISSION**: "{instruction}"

**Current TODO List**:
{todo}

**Available directions**: {_VIEW_LIST}
(the images are attached in that order)

**Your tasks**:
1. Update the TODO list: mark finished items [x].
2. Pick the next action:
   - NAVIGATE while navigation steps remain
   - STOP once every step is finished

**Reply with one JSON object** (no markdown, nothing else):
{{
    "progress_analysis": "...(30 words or fewer)...",
    "reasoning": "...(1-2 sentences)...",
    "updated_todo_list": "...",
    "action": "NAVIGATE" | "STOP",
    "turn_direction": "front|left|back|right",
    "expected_landmark": "..."
}}
"""


def tactical_prompt(instruction: str, progress: str, target: str,
                    width: int, height: int) -> str:
    """Tactical bbox body placed after the current-view image (VA call)."""
    return f"""
**ROLE**: You are the robot's tactical eyes.
**MISSION**: "{instruction}"
**PROGRESS**: "{progress}"
**CURRENT TARGET**: "{target}"

Find the target described above in the image.
Coordinates are pixels in [0..{width}, 0..{height}].

JSON:
{{
    "visual_check": "I see ...",
    "action": "NAVIGATE" | "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "object name",
    "stop_reasoning": "..."
}}
"""


# --------------------------------------------------------------------------
# Output parsing (upstream fallbacks preserved)
# --------------------------------------------------------------------------

@dataclass
class LAResult:
    ok: bool                      # False = model/parse failure (fallback)
    action: str                   # NAVIGATE | STOP
    direction: str                # key of DIRECTIONS
    expected_landmark: str
    updated_todo: str
    progress_analysis: str = ""
    reasoning: str = ""


def _extract_json(text: str) -> dict | None:
    """Fenced ```json block first, else outermost braces (upstream order)."""
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


def parse_la_output(text: str, prev_todo: str = "") -> LAResult:
    """Parse the strategic reply; upstream fallback = NAVIGATE / right /
    'open space' with the TODO list left unchanged."""
    fallback = LAResult(False, "NAVIGATE", "right", "open space", prev_todo,
                        "Fallback", "Model error")
    obj = _extract_json(text)
    if obj is None:
        return fallback
    action = str(obj.get("action", "")).strip().upper()
    if action not in ("NAVIGATE", "STOP"):
        action = "NAVIGATE"
    direction = str(obj.get("turn_direction", "")).strip().lower()
    if direction not in DIRECTIONS:
        direction = "right"
    todo = str(obj.get("updated_todo_list", "")).strip() or prev_todo
    return LAResult(
        ok=True, action=action, direction=direction,
        expected_landmark=str(obj.get("expected_landmark", "")).strip()
        or "open space",
        updated_todo=todo,
        progress_analysis=str(obj.get("progress_analysis", "")),
        reasoning=str(obj.get("reasoning", "")),
    )


@dataclass
class VAResult:
    action: str                          # NAVIGATE | STOP
    bbox_px: tuple[int, int, int, int] | None
    target: str = ""
    stop_reasoning: str = ""
    visual_check: str = ""


def parse_va_output(text: str, width: int, height: int) -> VAResult:
    """Parse the tactical reply and normalise the bbox to pixels.

    Upstream heuristic kept as-is: coordinates whose maximum is <= 1000
    are treated as Qwen-style [0, 1000]-normalized and rescaled. A missing
    or malformed bbox is NOT an error: the caller falls back to the image
    center (the robot still moves)."""
    obj = _extract_json(text)
    if obj is None:
        return VAResult("NAVIGATE", None)
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
    return VAResult(action=action, bbox_px=bbox_px,
                    target=str(obj.get("target", "")),
                    stop_reasoning=str(obj.get("stop_reasoning", "")),
                    visual_check=str(obj.get("visual_check", "")))


# --------------------------------------------------------------------------
# RA grounding: nav pixel -> world goal (always succeeds, as upstream)
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
                  cam: PinholeCamera, robot_pose: np.ndarray,
                  window: int = 7) -> np.ndarray:
    """Project one pixel + median window depth into a world (x, y) goal.

    Never fails: with no valid depth the goal defaults to 1.0 m straight
    ahead (upstream behaviour) — a wrong-but-moving guess beats freezing,
    and the CBF layer owns collision safety anyway."""
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
    x, y, th = robot_pose
    fwd = np.array([np.cos(th), np.sin(th)])
    if d is None:
        return np.array([x, y]) + 1.0 * fwd
    xc = d * (u - cam.cx) / cam.fx            # camera x: right
    right = np.array([np.sin(th), -np.cos(th)])
    return np.array([x, y]) + d * fwd + xc * right


# --------------------------------------------------------------------------
# Ollama client (LA + VA share one model, as upstream's llama-server)
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

    def _chat(self, text: str, images: list[np.ndarray],
              temperature: float = 0.0) -> str:
        import requests
        from .vlm_client import OllamaVLM
        options: dict = {"temperature": temperature}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [{"role": "user", "content": text,
                          "images": [OllamaVLM._encode_image(i)
                                     for i in images]}],
        }
        r = requests.post(f"{self.host}/api/chat", json=payload,
                          timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        self.last_raw = r.json()["message"]["content"]
        return self.last_raw

    def initial_todo(self, views: list[np.ndarray], instruction: str) -> str:
        """One-shot initial checklist from the starting panorama."""
        text = (initial_todo_prompt(instruction) + "\n\n"
                + initial_todo_suffix())
        return self._chat(text, views, temperature=0.1).strip()

    def decide(self, views: list[np.ndarray], instruction: str,
               todo: str, steps: list[dict]) -> LAResult:
        """Strategic decision (LA): direction / STOP + TODO rewrite."""
        text = (strategic_task_text(instruction, steps) + "\n"
                + strategic_prompt(instruction, todo))
        try:
            return parse_la_output(self._chat(text, views), prev_todo=todo)
        except Exception:            # network/model error -> upstream fallback
            return parse_la_output("", prev_todo=todo)

    def locate(self, rgb: np.ndarray, instruction: str, progress: str,
               target: str) -> VAResult:
        """Tactical grounding (VA): STOP or a bbox of the current target."""
        h, w = rgb.shape[:2]
        text = tactical_prompt(instruction, progress, target, w, h)
        try:
            return parse_va_output(self._chat(text, [rgb]), w, h)
        except Exception:            # no reply -> no bbox; caller uses center
            return VAResult("NAVIGATE", None)
