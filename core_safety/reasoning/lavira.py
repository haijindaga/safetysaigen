"""LaViRA-mode strategic layer (LA): panorama -> TODO checklist + direction.

Own implementation of the Language Action stage of LaViRA / Uni-LaViRA
(Ding et al., arXiv:2510.19655 / 2605.27582), no upstream code copied.
The full mode is a three-stage translation, run as a repeated cycle:

  LA (this module):  4 panoramic views (90-deg increments, matching the
                     camera's 90-deg FOV) + mission + markdown TODO list +
                     history -> updated TODO, NAVIGATE/STOP, one direction,
                     and ONE landmark that is VISIBLE in the chosen view.
  VA (goal_grounding): boxes that landmark in the current view; bbox
                     bottom-center + depth -> one world-frame subgoal.
  RA (existing stack): A* drives to the subgoal, CBF-QP arbitrates every
                     command. Safety layers are untouched by this mode.

The point of LA: a goal that is NOT visible (e.g. hidden behind an obstacle
row) becomes a chain of visible intermediate landmarks, instead of the
undirected STOP_AND_SCAN/INVESTIGATE wandering of extended mode.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

# View/direction layout, counter-clockwise from the current heading. The
# panorama is captured in this order; "left" is +90 deg (CCW positive).
DIRECTIONS: dict[str, float] = {
    "front": 0.0,
    "left": np.pi / 2,
    "back": np.pi,
    "right": -np.pi / 2,
}
VIEW_ORDER = ("front", "left", "back", "right")

LA_ACTIONS = ("NAVIGATE", "STOP")

LA_SYSTEM_PROMPT = """You are the strategic navigator of a mobile robot.
Each cycle you receive 4 panoramic views taken from the robot's current
position (in this order: 1=front 0 deg, 2=left +90 deg, 3=back 180 deg,
4=right -90 deg), the MISSION, your own TODO checklist from the previous
cycle, and the recent navigation history.

Your tasks:
1. Update the TODO checklist. Markdown format, one item per line:
   "- [ ] pending step" / "- [x] completed step". Keep it short and
   concrete; refine it when the scene reveals new information (e.g. an
   obstacle row that must be bypassed becomes its own step).
2. Decide the next action:
   - "NAVIGATE": keep going. Pick the ONE view direction that best
     advances the mission, and name ONE landmark that is ACTUALLY VISIBLE
     in that view for the robot to drive toward. If the mission target is
     not visible in any view, pick an intermediate landmark that will
     reveal new space (e.g. "the open floor beyond the right end of the
     shelf row"). Never name something you cannot see.
   - "STOP": the mission is complete (the robot is at the target).

Respond with a single flat JSON object and nothing else:
{
"updated_todo": "- [x] ...\\n- [ ] ...",
"action": "NAVIGATE" or "STOP",
"direction": "front" | "left" | "back" | "right",
"expected_landmark": "short description of one visible landmark",
"reasoning": "one or two sentences"
}"""


def la_user_prompt(mission: str, todo: str, history: list[str]) -> str:
    """User message accompanying the 4 panorama images (in VIEW_ORDER)."""
    hist = "\n".join(f"- {h}" for h in history[-6:]) if history else "(none)"
    return (f'MISSION: "{mission}"\n\n'
            f"CURRENT TODO LIST:\n{todo or '(first cycle: create it)'}\n\n"
            f"NAVIGATION HISTORY (most recent last):\n{hist}\n\n"
            "The 4 images are attached in order: 1=front(0 deg), "
            "2=left(+90 deg), 3=back(180 deg), 4=right(-90 deg).\n"
            "Update the TODO list and output the JSON.")


@dataclass
class LAResult:
    ok: bool                      # False = fallback (parse/model failure)
    action: str                   # NAVIGATE | STOP
    direction: str                # key of DIRECTIONS
    expected_landmark: str
    updated_todo: str
    reasoning: str = ""


def parse_la_output(text: str, prev_todo: str = "") -> LAResult:
    """Parse the LA JSON reply; degrade to a safe NAVIGATE/front fallback.

    LA failures must never raise into the control loop: the fallback keeps
    the previous TODO list and sends the robot to look front, where the
    VA "not visible" path or the next panorama recovers.
    """
    fallback = LAResult(False, "NAVIGATE", "front", "", prev_todo,
                        "LA output unusable")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return fallback
    try:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
    except json.JSONDecodeError:
        return fallback
    action = str(obj.get("action", "")).strip().upper()
    if action not in LA_ACTIONS:
        action = "NAVIGATE"
    direction = str(obj.get("direction", "")).strip().lower()
    if direction not in DIRECTIONS:
        direction = "front"
    todo = str(obj.get("updated_todo", "")).strip() or prev_todo
    return LAResult(
        ok=True,
        action=action,
        direction=direction,
        expected_landmark=str(obj.get("expected_landmark", "")).strip(),
        updated_todo=todo,
        reasoning=str(obj.get("reasoning", "")),
    )


class OllamaLA:
    """One LA decision per call: 4 panorama images -> LAResult."""

    def __init__(self, model: str = "gemma3:27b",
                 host: str = "http://localhost:11434",
                 timeout: float = 600.0, num_gpu: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.last_raw: str | None = None

    def decide(self, views: list[np.ndarray], mission: str,
               todo: str, history: list[str]) -> LAResult:
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
                {"role": "system", "content": LA_SYSTEM_PROMPT},
                {"role": "user",
                 "content": la_user_prompt(mission, todo, history),
                 "images": [OllamaVLM._encode_image(v) for v in views]},
            ],
        }
        r = requests.post(f"{self.host}/api/chat", json=payload,
                          timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        self.last_raw = r.json()["message"]["content"]
        return parse_la_output(self.last_raw, prev_todo=todo)
