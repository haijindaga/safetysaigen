import numpy as np

from core_safety.grounding.projection import PinholeCamera, pixels_to_world
from core_safety.sim2d.world import World, Cylinder, GroundPatch
from core_safety.sim2d.camera import RaycastCamera


def test_render_and_reproject_roundtrip():
    """Points rendered by the raycaster must reproject to their true world
    position through the pinhole model (consistency of the two codepaths)."""
    world = World(objects=[Cylinder("cone", 5.0, 0.0, 0.3, 1.0)])
    cam = RaycastCamera()
    pose = np.array([0.0, 0.0, 0.0])
    rgb, depth, labels = cam.render(world, pose)

    cone_id = world.class_ids()["cone"]
    mask = labels == cone_id
    assert mask.any(), "cone should be visible"

    pts = pixels_to_world(cam.cam, depth, mask, pose, min_range=0.5, max_range=10.0)
    assert len(pts) > 0
    # All projected points lie on the front surface of the cylinder.
    d = np.hypot(pts[:, 0] - 5.0, pts[:, 1] - 0.0)
    assert (d <= 0.3 + 0.05).all()
    assert (pts[:, 0] <= 5.0 + 1e-6).all()   # front side only


def test_ground_class_rendering():
    world = World(default_ground="grass",
                  ground_patches=[GroundPatch("sidewalk", -1, 20, -1, 1)])
    cam = RaycastCamera()
    rgb, depth, labels = cam.render(world, np.array([0.0, 0.0, 0.0]))
    ids = world.class_ids()
    assert (labels == ids["sidewalk"]).any()
    assert (labels == ids["grass"]).any()


def test_range_clipping():
    cam = PinholeCamera.paper_sim()
    depth = np.full((cam.height, cam.width), 2.0)   # everything at 2 m
    mask = np.ones_like(depth, dtype=bool)
    pts = pixels_to_world(cam, depth, mask, np.zeros(3),
                          min_range=3.0, max_range=7.0)
    assert len(pts) == 0                             # all clipped
