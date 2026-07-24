"""LaViRA mode (single-call variant): parsing, orientation, RA grounding."""
import numpy as np
import pytest

from core_safety.grounding.projection import PinholeCamera
from core_safety.reasoning.lavira import (DIRECTIONS, VIEW_ORDER,
                                          history_text, nav_pixel,
                                          orientation_text,
                                          parse_plan_output, pixel_to_goal,
                                          plan_prompt, relative_sector)

W, H = 500, 320
CAM = PinholeCamera(fx=274.9, fy=376.7, cx=250.0, cy=160.0,
                    width=W, height=H, mount_height=0.25)


def test_view_order_matches_directions():
    assert set(VIEW_ORDER) == set(DIRECTIONS)
    # Capture order must be strictly CCW so a constant-omega rotation
    # passes each capture heading exactly once.
    ccw = [DIRECTIONS[v] % (2 * np.pi) for v in VIEW_ORDER]
    assert ccw == sorted(ccw)


# ---- plan parsing ----------------------------------------------------------

def test_parse_plan_full():
    r = parse_plan_output(
        '{"progress_analysis": "cones ahead", "reasoning": "detour",'
        ' "updated_todo_list": "- [x] scan\\n- [ ] pass cones",'
        ' "action": "NAVIGATE", "view": 2, "target": "floor left of cones",'
        ' "bbox_2d": [100, 200, 300, 400]}', W, H)
    assert r.ok and r.action == "NAVIGATE"
    assert r.view == "left"                       # view 2 = VIEW_ORDER[1]
    assert r.bbox_px == (50, 64, 150, 128)        # [0,1000] -> pixels
    assert r.target == "floor left of cones"
    assert "- [ ] pass cones" in r.updated_todo


def test_parse_plan_view_by_name():
    r = parse_plan_output('{"action": "NAVIGATE", "view": "back"}', W, H)
    assert r.ok and r.view == "back"


def test_parse_plan_stop():
    r = parse_plan_output('{"action": "stop", "view": 1,'
                          ' "updated_todo_list": "- [x] a"}', W, H)
    assert r.ok and r.action == "STOP"


def test_parse_plan_fenced_json():
    r = parse_plan_output('OK.\n```json\n{"action": "NAVIGATE",'
                          ' "view": 4}\n```', W, H)
    assert r.ok and r.view == "right"


def test_parse_plan_fallback_still_moves():
    # Garbage output degrades to NAVIGATE / front / no bbox: the caller
    # grounds the view-center pixel and the robot still moves.
    r = parse_plan_output("let me think...", W, H, prev_todo="- [ ] x")
    assert not r.ok
    assert r.action == "NAVIGATE" and r.view == "front"
    assert r.bbox_px is None
    assert r.updated_todo == "- [ ] x"


def test_parse_plan_bad_view_and_bbox():
    r = parse_plan_output('{"action": "NAVIGATE", "view": 9,'
                          ' "bbox_2d": ["a", 1, 2, 3]}', W, H)
    assert r.ok and r.view == "front" and r.bbox_px is None


# ---- prompt / memory -------------------------------------------------------

def test_history_and_prompt_carry_state():
    steps = [{"description": "started", "target": "went left toward 'shelf'"}]
    assert "Step 1: started -> went left toward 'shelf'" \
        in history_text(steps)
    assert "No history yet." in history_text([])
    p = plan_prompt("reach the marker", "- [ ] step", steps,
                    orientation_text(0.0, "", None), W, H)
    assert "reach the marker" in p and "- [ ] step" in p
    assert "first cycle" in p
    assert "1=front" in p and "4=right" in p


def test_orientation_text_continuity():
    t = orientation_text(3.2, "cone gap", "back-left")
    assert "3.2 m" in t and "cone gap" in t and "back-left" in t
    assert "CURRENT heading" in t


def test_relative_sector_buckets():
    assert relative_sector(0.0) == "front"
    assert relative_sector(np.pi / 2) == "left"
    assert relative_sector(-np.pi / 2) == "right"
    assert relative_sector(np.pi) == "back"
    assert relative_sector(3 * np.pi / 4) == "back-left"


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


def test_pixel_to_goal_uses_view_pose():
    # The subgoal is grounded from the CAPTURE-time pose of the chosen
    # view: a left-view bbox must project left of the panorama position
    # even though the robot has since rotated further.
    depth = np.full((H, W), 2.0)
    g = pixel_to_goal(int(CAM.cx), int(CAM.cy), depth, CAM,
                      np.array([1.0, -1.0, np.pi / 2]))
    np.testing.assert_allclose(g, [1.0, 1.0], atol=1e-6)


def test_pixel_to_goal_no_depth_is_one_meter_ahead():
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
