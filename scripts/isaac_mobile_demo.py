"""Isaac Sim closed-loop demo: Jetbot + cone line + contextual safety filter.

Run inside the Isaac environment (conda env_isaaclab), NOT the project venv:

    conda activate env_isaaclab
    cd core-safety
    export PYTHONPATH=$PWD:$PYTHONPATH
    python scripts/isaac_mobile_demo.py                    # GT seg + rulebook
    python scripts/isaac_mobile_demo.py --vlm ollama       # real VLM (async)
    python scripts/isaac_mobile_demo.py --segmenter sam3   # real SAM3
    python scripts/isaac_mobile_demo.py --headless         # no GUI window

Scene: ground plane ("floor") + a line of orange cylinders ("cone") between
the Jetbot and its goal. The nominal waypoint follower drives straight at
the goal; the safety filter must stop the robot at the cone line
(BETWEEN(cone) grounded into the costmap barrier).

Written against Isaac Sim >= 4.5 (isaacsim.* namespace) with fallbacks for
the older omni.isaac.* names. Camera annotators require rendering, which
also runs in --headless mode.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--vlm", choices=["rulebook", "ollama"], default="rulebook")
parser.add_argument("--segmenter", choices=["gt", "sam3"], default="gt")
parser.add_argument("--ollama-model", default="gemma3:27b")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--goal", type=float, nargs=2, default=[5.0, 0.0])
parser.add_argument("--perception-every", type=int, default=30,
                    help="physics steps between perception updates (GT mode)")
args = parser.parse_args()

# ---- Isaac boot (must precede any other isaac import) ----------------------
try:
    from isaacsim import SimulationApp
except ImportError:  # pre-4.5
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCylinder, VisualCuboid
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
        DifferentialController)
    from isaacsim.sensors.camera import Camera
except ImportError:  # pre-4.5 namespaces
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCylinder, VisualCuboid
    from omni.isaac.core.utils.semantics import add_update_semantics
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.wheeled_robots.robots import WheeledRobot
    from omni.isaac.wheeled_robots.controllers.differential_controller import (
        DifferentialController)
    from omni.isaac.sensor import Camera

import numpy as np

from core_safety.control.nominal import WaypointFollower
from core_safety.grounding.segmentation import GroundTruthSegmenter
from core_safety.isaac.adapter import (IsaacPlanarBase, camera_from_intrinsics,
                                       parse_semantic_frame)
from core_safety.pipeline import CorePipeline, CoreConfig
from core_safety.reasoning.vlm_client import RuleBasedVLM, OllamaVLM, DEFAULT_RULEBOOK

PHYSICS_DT = 1.0 / 60.0
V_MAX, OMEGA_MAX = 0.3, 1.0          # Jetbot-scale limits
CAM_RES = (500, 320)                 # (width, height), matches 2D sim scale

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
              rendering_dt=PHYSICS_DT)
ground = world.scene.add_default_ground_plane()
add_update_semantics(get_prim_at_path("/World/defaultGroundPlane"), "floor")

for i, y in enumerate(np.linspace(-1.2, 1.2, 5)):
    cone = world.scene.add(FixedCylinder(
        prim_path=f"/World/cone_{i}", name=f"cone_{i}",
        position=np.array([3.0, float(y), 0.2]),
        radius=0.08, height=0.4, color=np.array([1.0, 0.4, 0.0])))
    add_update_semantics(cone.prim, "cone")

# Goal marker (visual only, no semantics -> invisible to the safety stack)
world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal",
                             position=np.array([args.goal[0], args.goal[1], 0.05]),
                             scale=np.array([0.2, 0.2, 0.1]),
                             color=np.array([0.0, 1.0, 0.0])))

import omni.client

assets_root = get_assets_root_path()
jetbot_usd = None
for cand in ("/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",     # 5.x layout
             "/Isaac/Robots/Jetbot/jetbot.usd"):           # 4.x layout
    url = assets_root + cand
    result, _ = omni.client.stat(url)
    if result == omni.client.Result.OK:
        jetbot_usd = url
        break
if jetbot_usd is None:
    raise RuntimeError("Jetbot USD not found under the assets root; "
                       "check the asset paths in this script.")
robot = world.scene.add(WheeledRobot(
    prim_path="/World/jetbot", name="jetbot",
    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
    create_robot=True, usd_path=jetbot_usd,
    position=np.array([0.0, 0.0, 0.03])))
diff_controller = DifferentialController(name="diff", wheel_radius=0.03,
                                         wheel_base=0.1125)

# Orientation uses the wrapper's world-axes convention (x-forward); identity
# means "look along the chassis +x". If the image comes out pointing down or
# rotated on your Isaac version, adjust this quaternion.
camera = Camera(prim_path="/World/jetbot/chassis/core_cam",
                translation=np.array([0.08, 0.0, 0.25]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                resolution=CAM_RES, frequency=20)

world.reset()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_semantic_segmentation_to_frame()

# ---------------------------------------------------------------------------
# CORE pipeline
# ---------------------------------------------------------------------------
cfg = CoreConfig(min_range=0.4, max_range=6.0,     # Jetbot-scale (paper: 3-7 m)
                 v_max=V_MAX, omega_max=OMEGA_MAX,
                 perception_period=1)              # we gate perception ourselves

if args.vlm == "ollama":
    vlm = OllamaVLM(model=args.ollama_model)
else:
    vlm = RuleBasedVLM(DEFAULT_RULEBOOK, contextual=True)

gt_segmenter = GroundTruthSegmenter()
if args.segmenter == "sam3":
    from core_safety.grounding.sam3_segmenter import SAM3Segmenter
    segmenter = SAM3Segmenter()
else:
    segmenter = gt_segmenter

pin_cam = camera_from_intrinsics(camera.get_intrinsics_matrix(),
                                 CAM_RES[0], CAM_RES[1], mount_height=0.25)
pipeline = CorePipeline(vlm, segmenter, pin_cam,
                        workspace=(-2.0, 8.0, -4.0, 4.0), config=cfg)

base = IsaacPlanarBase(dt=PHYSICS_DT, v_max=V_MAX, omega_max=OMEGA_MAX,
                       holonomic=False)
nominal = WaypointFollower([tuple(args.goal)], v_max=V_MAX, omega_max=OMEGA_MAX)

# Async perception: the slow outer loop (VLM/SAM3) must not stall physics.
_lock = threading.Lock()
_latest = {}          # frame handed to the worker
_busy = threading.Event()


def _perception_worker():
    while simulation_app.is_running():
        _busy.wait(timeout=0.5)
        if not _busy.is_set():
            continue
        with _lock:
            frame = dict(_latest)
        try:
            gt_segmenter.update(frame["labels"], frame["id_to_name"])
            pipeline.update_perception(
                frame["rgb"], frame["depth"], frame["pose"],
                visible_classes=frame["classes"],
                instance_counts=frame["counts"])
        except Exception as e:      # keep last good barrier on any failure
            print(f"[perception] update failed: {e}")
        _busy.clear()


threading.Thread(target=_perception_worker, daemon=True).start()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
step = 0
while simulation_app.is_running() and step < args.steps:
    pos, quat = robot.get_world_pose()
    base.update_pose(pos, quat)

    if step % args.perception_every == 0 and not _busy.is_set():
        frame = camera.get_current_frame()
        rgba = frame.get("rgba")
        depth = frame.get("distance_to_image_plane")
        sem = frame.get("semantic_segmentation")
        if rgba is not None and depth is not None and sem is not None \
                and np.asarray(depth).size > 1:
            labels, id_to_name = parse_semantic_frame(sem)
            present = [id_to_name[i] for i in np.unique(labels)
                       if i in id_to_name and id_to_name[i] not in
                       ("background", "unlabelled", "unlabeled")]
            with _lock:
                _latest.update(
                    rgb=np.asarray(rgba)[:, :, :3].copy(),
                    depth=np.asarray(depth, dtype=float).copy(),
                    labels=labels, id_to_name=id_to_name,
                    classes=present,
                    counts={"cone": 5},
                    pose=base.state.copy())
            _busy.set()

    u_nom = nominal.compute(base.state)
    u_safe = pipeline.safe_control(u_nom, base)
    u_safe = base.clip_input(u_safe)
    action = diff_controller.forward(command=[float(u_safe[0]), float(u_safe[2])])
    robot.apply_wheel_actions(action)

    world.step(render=True)
    if step % 60 == 0:
        d_goal = float(np.linalg.norm(base.state[:2] - np.asarray(args.goal)))
        print(f"t={step*PHYSICS_DT:5.1f}s pos=({base.state[0]:+.2f},"
              f"{base.state[1]:+.2f}) h={pipeline.debug.h:6.2f} "
              f"filtered={pipeline.debug.filtered} d_goal={d_goal:.2f}")
        if d_goal < 0.25:
            print("reached goal")
            break
    step += 1

print("done — robot should be stopped at the cone line, not past it "
      "(cones are at x=3.0).")
simulation_app.close()
