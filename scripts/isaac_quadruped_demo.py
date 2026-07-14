"""Isaac Sim quadruped demo: Spot walking policy + contextual safety filter.

The flat-terrain locomotion policy (isaacsim.robot.policy.examples) consumes
a body-velocity command [vx, vy, yaw_rate] — exactly the interface the
safety filter acts on. CORE filters the command; the policy handles legs.

Run inside the Isaac environment (conda env_isaaclab):

    conda activate env_isaaclab
    cd <repo>
    export PYTHONPATH=$PWD:$PYTHONPATH
    python scripts/isaac_quadruped_demo.py                  # GT seg + rulebook
    python scripts/isaac_quadruped_demo.py --vlm ollama --segmenter sam3
    python scripts/isaac_quadruped_demo.py --headless

Scene: same cone-line scenario as the mobile demo, scaled up for Spot
(cones at x=4, goal at x=7). Success: Spot walks toward the goal and the
filter stops it short of the cone line.
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
parser.add_argument("--ollama-num-gpu", type=int, default=0,
                    help="Ollama GPU layers; 0 = CPU-only (default: the GPU "
                         "is needed by Isaac/SAM3 on an 8 GB card)")
parser.add_argument("--steps", type=int, default=6000, help="render steps (50 Hz)")
parser.add_argument("--goal", type=float, nargs=2, default=[7.0, 0.0])
parser.add_argument("--perception-every", type=int, default=25,
                    help="render steps between perception updates")
args = parser.parse_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCylinder, VisualCuboid
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.sensors.camera import Camera
    from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCylinder, VisualCuboid
    from omni.isaac.core.utils.semantics import add_update_semantics
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.sensor import Camera
    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy

import numpy as np

from core_safety.control.nominal import WaypointFollower
from core_safety.grounding.segmentation import GroundTruthSegmenter
from core_safety.isaac.adapter import (IsaacPlanarBase, camera_from_intrinsics,
                                       parse_semantic_frame)
from core_safety.pipeline import CorePipeline, CoreConfig
from core_safety.reasoning.vlm_client import RuleBasedVLM, OllamaVLM, DEFAULT_RULEBOOK

# Spot policy runs the physics loop at 500 Hz; we command at render rate.
PHYSICS_DT = 1.0 / 500.0
RENDER_DT = 1.0 / 50.0
V_MAX, OMEGA_MAX = 0.5, 1.0          # conservative Spot walk limits
CAM_RES = (500, 320)
CAM_HEIGHT = 0.6                     # camera above ground when mounted on body

world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
              rendering_dt=RENDER_DT)
world.scene.add_default_ground_plane()
add_update_semantics(get_prim_at_path("/World/defaultGroundPlane"), "floor")

for i, y in enumerate(np.linspace(-1.6, 1.6, 5)):
    cone = world.scene.add(FixedCylinder(
        prim_path=f"/World/cone_{i}", name=f"cone_{i}",
        position=np.array([4.0, float(y), 0.25]),
        radius=0.1, height=0.5, color=np.array([1.0, 0.4, 0.0])))
    add_update_semantics(cone.prim, "cone")

world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal",
                             position=np.array([args.goal[0], args.goal[1], 0.05]),
                             scale=np.array([0.2, 0.2, 0.1]),
                             color=np.array([0.0, 1.0, 0.0])))

spot = SpotFlatTerrainPolicy(prim_path="/World/spot", name="spot",
                             position=np.array([0.0, 0.0, 0.8]))

# Camera on the body, looking forward (world-axes convention: x-forward).
camera = Camera(prim_path="/World/spot/body/core_cam",
                translation=np.array([0.4, 0.0, 0.1]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                resolution=CAM_RES, frequency=20)

world.reset()
spot.initialize()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_semantic_segmentation_to_frame()

cfg = CoreConfig(min_range=0.8, max_range=7.0, v_max=V_MAX,
                 omega_max=OMEGA_MAX, perception_period=1)
vlm = (OllamaVLM(model=args.ollama_model, num_gpu=args.ollama_num_gpu) if args.vlm == "ollama"
       else RuleBasedVLM(DEFAULT_RULEBOOK, contextual=True))
gt_segmenter = GroundTruthSegmenter()
if args.segmenter == "sam3":
    from core_safety.grounding.sam3_segmenter import SAM3Segmenter
    segmenter = SAM3Segmenter()
else:
    segmenter = gt_segmenter

pin_cam = camera_from_intrinsics(camera.get_intrinsics_matrix(),
                                 CAM_RES[0], CAM_RES[1], mount_height=CAM_HEIGHT)
pipeline = CorePipeline(vlm, segmenter, pin_cam,
                        workspace=(-2.0, 10.0, -5.0, 5.0), config=cfg)
# Base command updates at render rate: use RENDER_DT for the discrete check.
base = IsaacPlanarBase(dt=RENDER_DT, v_max=V_MAX, omega_max=OMEGA_MAX,
                       holonomic=True)          # Spot policy accepts vy
nominal = WaypointFollower([tuple(args.goal)], v_max=V_MAX, omega_max=OMEGA_MAX)

# The policy must be stepped every PHYSICS step with the latest command.
latest_command = np.zeros(3)


def on_physics_step(step_size):
    spot.forward(step_size, latest_command)


world.add_physics_callback("spot_policy", on_physics_step)

# Async perception (slow outer loop).
_lock = threading.Lock()
_latest: dict = {}
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
            pipeline.update_perception(frame["rgb"], frame["depth"],
                                       frame["pose"],
                                       visible_classes=frame["classes"],
                                       instance_counts=frame["counts"])
        except Exception as e:
            print(f"[perception] update failed: {e}")
        _busy.clear()


threading.Thread(target=_perception_worker, daemon=True).start()

import os
import traceback

step = 0
try:
  while simulation_app.is_running() and step < args.steps:
    pos, quat = spot.robot.get_world_pose()
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
                    classes=present, counts={"cone": 5},
                    pose=base.state.copy())
            _busy.set()

    u_nom = nominal.compute(base.state)
    u_safe = base.clip_input(pipeline.safe_control(u_nom, base))
    latest_command = np.array([u_safe[0], u_safe[1], u_safe[2]])

    world.step(render=True)
    if step % 50 == 0:
        d_goal = float(np.linalg.norm(base.state[:2] - np.asarray(args.goal)))
        print(f"t={step*RENDER_DT:5.1f}s pos=({base.state[0]:+.2f},"
              f"{base.state[1]:+.2f}) h={pipeline.debug.h:6.2f} "
              f"filtered={pipeline.debug.filtered} d_goal={d_goal:.2f}")
        if d_goal < 0.3:
            print("reached goal")
            break
    step += 1
  print("done — Spot should be standing short of the cone line (x=4.0).")
except Exception:
    traceback.print_exc()

simulation_app.close()
os._exit(0)
