"""LaViRA mode: LA/VA parsing, upstream fallbacks, and RA grounding."""
import numpy as np
import pytest

from core_safety.grounding.projection import PinholeCamera
from core_safety.reasoning.lavira import (DIRECTIONS, VIEW_ORDER,
                                          history_text, nav_pixel,
                                          parse_la_output, parse_va_output,
                                          pixel_to_goal, strategic_prompt)

W, H = 500, 320
CAM = PinholeCamera(fx=274.9, fy=376.7, cx=250.0, cy=160.0,
                    width=W, height=H, mount_height=0.25)


def test_view_order_matches_directions():
    assert set(VIEW_ORDER) == set(DIRECTIONS)
    # Capture order must be strictly CCW so a constant-omega rotation
    # passes each capture heading exactly once.
    ccw = [DIRECTIONS[v] % (2 * np.pi) for v in VIEW_ORDER]
    assert ccw == sorted(ccw)


# ---- LA (strategic) --------------------------------------------------------

def test_parse_la_full():
    r = parse_la_output(
        '{"progress_analysis": "cones ahead", "reasoning": "detour",'
        ' "updated_todo_list": "- [x] scan\\n- [ ] pass cones",'
        ' "action": "NAVIGATE", "turn_direction": "left",'
        ' "expected_landmark": "gap left of the cones"}')
    assert r.ok and r.action == "NAVIGATE" and r.direction == "left"
    assert r.expected_landmark == "gap left of the cones"
    assert "- [ ] pass cones" in r.updated_todo
    assert r.progress_analysis == "cones ahead"


def test_parse_la_stop():
    r = parse_la_output('{"action": "stop", "updated_todo_list": "- [x] a"}')
    assert r.ok and r.action == "STOP"


def test_parse_la_fenced_json():
    r = parse_la_output('Sure.\n```json\n{"action": "NAVIGATE",'
                        ' "turn_direction": "back"}\n```')
    assert r.ok and r.direction == "back"


def test_parse_la_upstream_fallback():
    # Upstream falls back to NAVIGATE / right / "open space", keeping the
    # previous TODO — the robot keeps moving even when the model babbles.
    r = parse_la_output("let me think about it...", prev_todo="- [ ] x")
    assert not r.ok
    assert r.action == "NAVIGATE" and r.direction == "right"
    assert r.expected_landmark == "open space"
    assert r.updated_todo == "- [ ] x"


def test_parse_la_bad_direction():
    r = parse_la_output('{"action": "NAVIGATE", "turn_direction": "up"}')
    assert r.ok and r.direction == "right"


def test_history_and_prompt_carry_state():
    steps = [{"description": "started", "target": "shelf"}]
    assert "Step 1: started -> shelf" in history_text(steps)
    assert "No history yet." in history_text([])
    p = strategic_prompt("reach the marker", "- [ ] step")
    assert "reach the marker" in p and "- [ ] step" in p
    for d in DIRECTIONS:
        assert d in p


# ---- VA (tactical) ---------------------------------------------------------

def test_parse_va_normalized_bbox():
    r = parse_va_output('{"action": "NAVIGATE",'
                        ' "bbox_2d": [100, 200, 300, 400],'
                        ' "target": "cone"}', W, H)
    # max <= 1000 -> treated as [0,1000]-normalized (upstream heuristic)
    assert r.bbox_px == (50, 64, 150, 128)
    assert r.target == "cone"


def test_parse_va_stop():
    r = parse_va_output('{"action": "STOP", "bbox_2d": null,'
                        ' "stop_reasoning": "arrived"}', W, H)
    assert r.action == "STOP" and r.bbox_px is None
    assert r.stop_reasoning == "arrived"


def test_parse_va_garbage_is_navigate_no_bbox():
    r = parse_va_output("cannot comply", W, H)
    assert r.action == "NAVIGATE" and r.bbox_px is None


# ---- RA grounding (always yields a goal) -----------------------------------

def test_nav_pixel_bbox_bottom_center():
    assert nav_pixel((100, 50, 200, 250), W, H) == (150, 250)


def test_nav_pixel_fallback_center():
    assert nav_pixel(None, W, H) == (W // 2, H // 2)
    assert nav_pixel((10, 10, 5, 5), W, H) == (W // 2, H // 2)  # degenerate


def test_pixel_to_goal_straight_ahead():
    depth = np.full((H, W), 3.0)
    g = pixel_to_goal(int(CAM.cx), int(CAM.cy), depth, CAM,
                      np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(g, [3.0, 0.0], atol=1e-6)


def test_pixel_to_goal_rotated():
    depth = np.full((H, W), 2.0)
    g = pixel_to_goal(int(CAM.cx), int(CAM.cy), depth, CAM,
                      np.array([1.0, -1.0, np.pi / 2]))
    np.testing.assert_allclose(g, [1.0, 1.0], atol=1e-6)


def test_pixel_to_goal_no_depth_is_one_meter_ahead():
    # Upstream fallback: missing depth -> (1.0, 0.0) in the robot frame.
    depth = np.full((H, W), np.nan)
    g = pixel_to_goal(250, 160, depth, CAM, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(g, [1.0, 0.0], atol=1e-6)
    g = pixel_to_goal(250, 160, None, CAM, np.array([2.0, 0.0, np.pi]))
    np.testing.assert_allclose(g, [1.0, 0.0], atol=1e-6)


def test_pixel_to_goal_median_ignores_nans():
    depth = np.full((H, W), np.nan)
    depth[158:163, 248:253] = 3.0
    g = pixel_to_goal(250, 160, depth, CAM, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(g[0], 3.0, atol=1e-6)
