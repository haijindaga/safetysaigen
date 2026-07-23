"""LaViRA strategic layer (LA): prompt contract + output parsing."""
import numpy as np
import pytest

from core_safety.reasoning.lavira import (DIRECTIONS, LA_SYSTEM_PROMPT,
                                          VIEW_ORDER, la_user_prompt,
                                          parse_la_output)


def test_view_order_matches_directions():
    assert set(VIEW_ORDER) == set(DIRECTIONS)
    # Capture order must be strictly CCW so a constant-omega rotation
    # passes each capture heading exactly once.
    ccw = [DIRECTIONS[v] % (2 * np.pi) for v in VIEW_ORDER]
    assert ccw == sorted(ccw)


def test_direction_angles():
    assert DIRECTIONS["front"] == 0.0
    assert DIRECTIONS["left"] == pytest.approx(np.pi / 2)    # CCW positive
    assert DIRECTIONS["right"] == pytest.approx(-np.pi / 2)


def test_parse_full_output():
    r = parse_la_output(
        '{"updated_todo": "- [x] scan\\n- [ ] pass the cones",'
        ' "action": "NAVIGATE", "direction": "right",'
        ' "expected_landmark": "open floor past the cone row",'
        ' "reasoning": "goal hidden behind cones"}')
    assert r.ok and r.action == "NAVIGATE" and r.direction == "right"
    assert r.expected_landmark == "open floor past the cone row"
    assert "- [ ] pass the cones" in r.updated_todo


def test_parse_stop():
    r = parse_la_output('{"action": "stop", "updated_todo": "- [x] done"}')
    assert r.ok and r.action == "STOP"


def test_parse_bad_direction_falls_back_to_front():
    r = parse_la_output('{"action": "NAVIGATE", "direction": "north-west"}')
    assert r.ok and r.direction == "front"


def test_parse_garbage_keeps_previous_todo():
    r = parse_la_output("I think we should go left!",
                        prev_todo="- [ ] keep going")
    assert not r.ok
    assert r.action == "NAVIGATE" and r.direction == "front"
    assert r.updated_todo == "- [ ] keep going"


def test_parse_empty_todo_keeps_previous():
    r = parse_la_output('{"action": "NAVIGATE", "direction": "left",'
                        ' "updated_todo": ""}', prev_todo="- [ ] old")
    assert r.updated_todo == "- [ ] old"


def test_user_prompt_carries_state():
    p = la_user_prompt("reach the marker", "- [ ] step",
                       ["went left toward 'shelf'"])
    assert "reach the marker" in p
    assert "- [ ] step" in p
    assert "went left toward 'shelf'" in p
    # View labeling contract must match VIEW_ORDER (front/left/back/right).
    assert "1=front" in p and "2=left" in p


def test_system_prompt_mentions_all_directions():
    for d in DIRECTIONS:
        assert d in LA_SYSTEM_PROMPT
