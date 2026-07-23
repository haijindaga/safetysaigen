"""Language-goal grounding (LaViRA-style VA): parsing + back-projection."""
import numpy as np
import pytest

from core_safety.grounding.projection import PinholeCamera
from core_safety.reasoning.goal_grounding import (VAResult, bbox_to_goal,
                                                  parse_va_output)

W, H = 500, 320
CAM = PinholeCamera(fx=274.9, fy=376.7, cx=250.0, cy=160.0,
                    width=W, height=H, mount_height=0.25)


# ---- parse_va_output -------------------------------------------------------

def test_parse_denormalizes_to_pixels():
    r = parse_va_output(
        '{"visible": true, "bbox_2d": [100, 200, 300, 400], '
        '"reasoning": "cone ahead"}', W, H)
    assert r.visible
    assert r.bbox_px == (50, 64, 150, 128)   # /1000 * (W, H)
    assert r.reasoning == "cone ahead"


def test_parse_not_visible():
    r = parse_va_output('{"visible": false, "bbox_2d": null, '
                        '"reasoning": "no marker in view"}', W, H)
    assert not r.visible and r.bbox_px is None


def test_parse_tolerates_fences_and_trailing_comma():
    r = parse_va_output(
        'Sure!\n```json\n{"visible": true, '
        '"bbox_2d": [0, 0, 1000, 1000], "reasoning": "",}\n```', W, H)
    assert r.visible
    assert r.bbox_px == (0, 0, W - 1, H - 1)   # clamped to image bounds


@pytest.mark.parametrize("text", [
    "I cannot see it.",                                   # no JSON at all
    '{"visible": true}',                                  # missing bbox
    '{"visible": true, "bbox_2d": [1, 2, 3]}',            # wrong arity
    '{"visible": true, "bbox_2d": [300, 300, 100, 400]}', # inverted box
    '{"visible": true, "bbox_2d": ["a", 0, 1, 2]}',       # non-numeric
])
def test_parse_degrades_to_not_visible(text):
    r = parse_va_output(text, W, H)
    assert isinstance(r, VAResult)
    assert not r.visible and r.bbox_px is None


# ---- bbox_to_goal ----------------------------------------------------------

def _flat_depth(d):
    return np.full((H, W), d, dtype=float)


def test_goal_straight_ahead():
    # Box centered on the optical axis, contact pixel at the principal
    # point: the goal must lie d meters along the heading.
    bbox = (240, 100, 260, int(CAM.cy))
    goal = bbox_to_goal(bbox, _flat_depth(3.0), CAM,
                        np.array([0.0, 0.0, 0.0]))
    assert goal is not None
    np.testing.assert_allclose(goal, [3.0, 0.0], atol=1e-6)


def test_goal_rotated_robot():
    # Same box, robot facing +y: the goal rotates with the heading.
    bbox = (240, 100, 260, int(CAM.cy))
    goal = bbox_to_goal(bbox, _flat_depth(2.0), CAM,
                        np.array([1.0, -1.0, np.pi / 2]))
    assert goal is not None
    np.testing.assert_allclose(goal, [1.0, 1.0], atol=1e-6)


def test_goal_lateral_offset_sign():
    # Contact pixel right of center (u > cx) -> goal to the robot's right
    # (negative world y when facing +x).
    u = int(CAM.cx) + 100
    bbox = (u - 10, 100, u + 10, int(CAM.cy))
    goal = bbox_to_goal(bbox, _flat_depth(3.0), CAM,
                        np.array([0.0, 0.0, 0.0]))
    assert goal is not None
    assert goal[1] < 0
    np.testing.assert_allclose(goal[1], -3.0 * 100 / CAM.fx, atol=1e-6)


def test_goal_rejects_missing_depth():
    assert bbox_to_goal((240, 100, 260, 160),
                        _flat_depth(np.nan), CAM,
                        np.array([0.0, 0.0, 0.0])) is None


def test_goal_rejects_out_of_range_depth():
    assert bbox_to_goal((240, 100, 260, 160), _flat_depth(9.0), CAM,
                        np.array([0.0, 0.0, 0.0]),
                        max_range=6.0) is None


def test_goal_rejects_high_contact_point():
    # Contact pixel well above the horizon at wall-like depth: the bottom
    # of the box floats high above ground (box on a wall, not drivable).
    bbox = (240, 10, 260, 40)     # bottom edge near the top of the image
    assert bbox_to_goal(bbox, _flat_depth(5.0), CAM,
                        np.array([0.0, 0.0, 0.0])) is None


def test_goal_depth_median_ignores_nans():
    depth = _flat_depth(np.nan)
    depth[158:163, 248:253] = 3.0        # a few valid pixels in the window
    goal = bbox_to_goal((240, 100, 260, 160), depth, CAM,
                        np.array([0.0, 0.0, 0.0]))
    assert goal is not None
    np.testing.assert_allclose(goal[0], 3.0, atol=1e-6)
