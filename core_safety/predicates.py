"""Safety predicates: spatial operator + semantic class.

The VLM outputs constraints as two sets of predicates (safe / unsafe),
e.g. unsafe_regions = "NEAR(desk), AROUND(wet_floor_sign), BETWEEN(cone)".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

OPERATORS = ("ON", "NEAR", "AROUND", "BETWEEN")

_PRED_RE = re.compile(r"(ON|NEAR|AROUND|BETWEEN)\s*\(\s*([A-Za-z0-9_ \-]+?)\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class Predicate:
    op: str          # one of OPERATORS
    cls: str         # semantic class name, e.g. "wet_floor_sign"

    def __post_init__(self):
        object.__setattr__(self, "op", self.op.upper())
        object.__setattr__(self, "cls", self.cls.strip().lower().replace(" ", "_"))
        if self.op not in OPERATORS:
            raise ValueError(f"Unknown spatial operator: {self.op}")

    def __str__(self) -> str:
        return f"{self.op}({self.cls})"


BEHAVIORS = ("PROCEED", "SLOW", "STOP_AND_SCAN", "INVESTIGATE", "ASK_HUMAN")


@dataclass
class SafetyConstraints:
    """Structured output of the contextual safety reasoning module."""
    safety_logic: str = ""
    classes: list[str] = field(default_factory=list)
    unsafe: list[Predicate] = field(default_factory=list)
    safe: list[Predicate] = field(default_factory=list)
    # Extended-reasoning fields (None in faithful mode):
    behavior: str | None = None          # one of BEHAVIORS
    behavior_reason: str = ""
    message: str = ""                    # question/report for the human

    def all_classes(self) -> list[str]:
        """Classes referenced anywhere (for the segmentation front-end)."""
        seen: dict[str, None] = {}
        for c in self.classes:
            seen.setdefault(c.strip().lower().replace(" ", "_"))
        for p in self.safe + self.unsafe:
            seen.setdefault(p.cls)
        return list(seen)


def parse_predicates(text: str) -> list[Predicate]:
    """Parse a comma-separated predicate string like 'NEAR(a), AROUND(b)'."""
    if not text:
        return []
    return [Predicate(m.group(1), m.group(2)) for m in _PRED_RE.finditer(text)]


def parse_vlm_output(text: str) -> SafetyConstraints:
    """Parse the VLM's JSON response into SafetyConstraints.

    Robust to surrounding prose / markdown fences: extracts the first JSON
    object found in the text. Raises ValueError if no valid JSON is present
    (this counts as a reasoning failure, cf. Llava rows in Table II).
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("No JSON object in VLM output")
    raw = match.group(0)
    # Tolerate trailing commas, which some VLMs emit (and the paper's own
    # example JSON contains one).
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    obj = json.loads(raw)

    classes_field = obj.get("classes", "")
    if isinstance(classes_field, str):
        classes = [c.strip() for c in classes_field.split(",") if c.strip()]
    else:
        classes = [str(c) for c in classes_field]

    behavior = str(obj.get("behavior", "")).strip().upper() or None
    if behavior is not None and behavior not in BEHAVIORS:
        behavior = None
    return SafetyConstraints(
        safety_logic=str(obj.get("safety_logic", "")),
        classes=classes,
        unsafe=parse_predicates(str(obj.get("unsafe_regions", ""))),
        safe=parse_predicates(str(obj.get("safe_regions", ""))),
        behavior=behavior,
        behavior_reason=str(obj.get("behavior_reason", "")),
        message=str(obj.get("message", "")),
    )
