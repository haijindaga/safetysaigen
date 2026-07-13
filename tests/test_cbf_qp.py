import numpy as np

from core_safety.control.cbf_qp import CBFSafetyFilter
from core_safety.control.dynamics import PlanarRobot


def test_passthrough_when_safe_direction():
    f = CBFSafetyFilter(v_max=1.0)
    robot = PlanarRobot(theta=0.0)
    u_nom = np.array([0.3, 0.0, 0.0])
    # Barrier far away, gradient pointing along +x (deeper into safety).
    u = f.filter(u_nom, h=5.0, grad_h_xy=np.array([1.0, 0.0]),
                 f=robot.f(), g=robot.g())
    assert np.allclose(u, u_nom)
    assert not f.last_active


def test_blocks_motion_into_unsafe():
    filt = CBFSafetyFilter(alpha_gain=0.25, v_max=1.0)
    robot = PlanarRobot(theta=0.0)
    # Robot at the boundary (h=0), unsafe region ahead (+x): grad points -x.
    u_nom = np.array([0.5, 0.0, 0.0])
    u = filt.filter(u_nom, h=0.0, grad_h_xy=np.array([-1.0, 0.0]),
                    f=robot.f(), g=robot.g())
    # CBF condition: grad.g(x)u >= -alpha(0) = 0  =>  -u_x >= 0.
    assert u[0] <= 1e-6
    assert filt.last_active


def test_forward_invariance_closed_loop():
    """Drive toward a wall at x=5 with h(x) = 5 - x; h must stay >= 0."""
    filt = CBFSafetyFilter(alpha_gain=0.25, v_max=1.0)
    robot = PlanarRobot(x=0.0, theta=0.0, v_max=1.0, dt=0.1)
    for _ in range(200):
        h = 5.0 - robot.state[0]
        u_nom = np.array([1.0, 0.0, 0.0])
        u = filt.filter(u_nom, h=h, grad_h_xy=np.array([-1.0, 0.0]),
                        f=robot.f(), g=robot.g())
        robot.step(u)
    assert 5.0 - robot.state[0] >= -1e-3     # forward invariant
    assert robot.state[0] > 3.0              # but made progress toward it


def test_minimal_deviation():
    """Sideways nominal motion parallel to the boundary is untouched."""
    filt = CBFSafetyFilter(v_max=1.0)
    robot = PlanarRobot(theta=0.0)
    u_nom = np.array([0.0, 0.4, 0.0])
    u = filt.filter(u_nom, h=0.0, grad_h_xy=np.array([-1.0, 0.0]),
                    f=robot.f(), g=robot.g())
    assert np.allclose(u, u_nom, atol=1e-6)
