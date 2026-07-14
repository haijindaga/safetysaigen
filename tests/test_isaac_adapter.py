import numpy as np

from core_safety.isaac.adapter import (IsaacPlanarBase, camera_from_intrinsics,
                                       parse_semantic_frame, quat_to_yaw)


def test_quat_to_yaw():
    assert abs(quat_to_yaw(np.array([1.0, 0, 0, 0]))) < 1e-9
    # 90 deg about z: (w, x, y, z) = (cos45, 0, 0, sin45)
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert abs(quat_to_yaw(q) - np.pi / 2) < 1e-9


def test_planar_base_interface():
    base = IsaacPlanarBase(dt=1 / 60, v_max=0.3, holonomic=False)
    base.update_pose(np.array([1.0, 2.0, 0.03]),
                     np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]))
    assert np.allclose(base.state[:2], [1.0, 2.0])
    assert abs(base.state[2] - np.pi / 4) < 1e-9
    # g(theta) rotates body velocity into the world frame.
    u = base.clip_input(np.array([1.0, 0.5, 5.0]))
    assert u[1] == 0.0                        # non-holonomic
    assert abs(u[0]) <= 0.3 and abs(u[2]) <= 1.0
    v_world = base.g() @ np.array([0.3, 0, 0])
    assert np.allclose(v_world[:2],
                       0.3 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]))


def test_camera_from_intrinsics():
    K = np.array([[274.9, 0, 250.0], [0, 376.7, 160.0], [0, 0, 1]])
    cam = camera_from_intrinsics(K, 500, 320, mount_height=0.25)
    assert cam.fx == 274.9 and cam.cy == 160.0 and cam.mount_height == 0.25


def test_parse_semantic_frame_variants():
    data = np.array([[0, 1], [2, 2]], dtype=np.uint32)
    # New style: labels are dicts; keys may be strings.
    frame = {"data": data,
             "info": {"idToLabels": {"0": {"class": "BACKGROUND"},
                                     "1": {"class": "Cone"},
                                     "2": "floor"}}}
    labels, id_to_name = parse_semantic_frame(frame)
    assert labels.dtype == np.int32
    assert id_to_name == {0: "background", 1: "cone", 2: "floor"}
