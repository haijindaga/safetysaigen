"""Isaac Sim mobile-manipulator demo: Ridgeback+Franka base under the filter.

EXPERIMENTAL — the Ridgeback+Franka asset drives its holonomic base through
three virtual joints (x/y prismatic + z revolute). We command those joints
in velocity mode; the Franka arm just holds its default pose. The safety
filter acts on the base velocity (vx, vy, omega) exactly as for the other
robots; filtering the end-effector is a documented future extension.

Run inside the Isaac environment (conda env_isaaclab):

    conda activate env_isaaclab
    cd <repo>
    export PYTHONPATH=$PWD:$PYTHONPATH
    python scripts/isaac_manipulator_demo.py [--headless]
                                             [--vlm ollama] [--segmenter sam3]

On startup the script prints the articulation's DOF names. If the three
base joints are not auto-detected, read that list and adjust BASE_JOINT_HINTS.
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
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
parser.add_argument("--perception-every", type=int, default=30)
args = parser.parse_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCylinder, VisualCuboid
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.sensors.camera import Camera
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCylinder, VisualCuboid
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.semantics import add_update_semantics
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.sensor import Camera

import numpy as np

from core_safety.control.nominal import WaypointFollower
from core_safety.grounding.segmentation import GroundTruthSegmenter
from core_safety.isaac.adapter import (IsaacPlanarBase, camera_from_intrinsics,
                                       parse_semantic_frame)
from core_safety.pipeline import CorePipeline, CoreConfig
from core_safety.reasoning.vlm_client import RuleBasedVLM, OllamaVLM, DEFAULT_RULEBOOK

PHYSICS_DT = 1.0 / 60.0
V_MAX, OMEGA_MAX = 0.4, 1.0
CAM_RES = (500, 320)
CAM_HEIGHT = 0.8
# Substrings used to find the virtual base joints in the articulation.
BASE_JOINT_HINTS = {"x": "prismatic_x", "y": "prismatic_y", "z": "revolute_z"}

world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
              rendering_dt=PHYSICS_DT)
world.scene.add_default_ground_plane()
add_update_semantics(get_prim_at_path("/World/defaultGroundPlane"), "floor")

for i, y in enumerate(np.linspace(-1.6, 1.6, 5)):
    cone = world.scene.add(FixedCylinder(
        prim_path=f"/World/cone_{i}", name=f"cone_{i}",
        position=np.array([3.5, float(y), 0.25]),
        radius=0.1, height=0.5, color=np.array([1.0, 0.4, 0.0])))
    add_update_semantics(cone.prim, "cone")

world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal",
                             position=np.array([args.goal[0], args.goal[1], 0.05]),
                             scale=np.array([0.2, 0.2, 0.1]),
                             color=np.array([0.0, 1.0, 0.0])))

import omni.client

assets_root = get_assets_root_path()
rf_usd = None
for cand in ("/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
             "/Isaac/Robots/RidgebackFranka/ridgeback_franka.usd"):
    url = assets_root + cand
    result, _ = omni.client.stat(url)
    if result == omni.client.Result.OK:
        rf_usd = url
        break
if rf_usd is None:
    raise RuntimeError("ridgeback_franka.usd not found; adjust asset paths.")
add_reference_to_stage(usd_path=rf_usd, prim_path="/World/ridgeback")
robot = world.scene.add(Robot(prim_path="/World/ridgeback", name="ridgeback"))

# Free-standing camera teleported to follow the base each step (robust to
# unknown prim trees inside the asset).
camera = Camera(prim_path="/World/follow_cam",
                position=np.array([0.4, 0.0, CAM_HEIGHT]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                resolution=CAM_RES, frequency=20)

world.reset()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_semantic_segmentation_to_frame()

dof_names = list(robot.dof_names)
print(f"[manipulator] articulation DOFs: {dof_names}")
base_idx = {}
for key, hint in BASE_JOINT_HINTS.items():
    matches = [i for i, n in enumerate(dof_names) if hint in n.lower()]
    if not matches:
        raise RuntimeError(f"base joint '{hint}' not found in {dof_names}; "
                           "adjust BASE_JOINT_HINTS.")
    base_idx[key] = matches[0]
idx = np.array([base_idx["x"], base_idx["y"], base_idx["z"]])
print(f"[manipulator] base joint indices: {base_idx}")

# Velocity mode for the three base DOFs only (arm keeps position control).
controller = robot.get_articulation_controller()
for i in idx:
    try:
        controller.switch_dof_control_mode(dof_index=int(i), mode="velocity")
    except Exception as e:
        print(f"[manipulator] switch_dof_control_mode failed for dof {i}: {e}")

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
                        workspace=(-2.0, 9.0, -5.0, 5.0), config=cfg)
base = IsaacPlanarBase(dt=PHYSICS_DT, v_max=V_MAX, omega_max=OMEGA_MAX,
                       holonomic=True)
nominal = WaypointFollower([tuple(args.goal)], v_max=V_MAX, omega_max=OMEGA_MAX)

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


def base_pose_from_joints() -> np.ndarray:
    q = robot.get_joint_positions()
    return np.array([q[idx[0]], q[idx[1]], q[idx[2]]])


step = 0
try:
  while simulation_app.is_running() and step < args.steps:
    x, y, th = base_pose_from_joints()
    base.state = np.array([x, y, th])

    # Follow camera: forward of the base, at yaw th (world-axes convention).
    cam_pos = np.array([x + 0.4 * np.cos(th), y + 0.4 * np.sin(th), CAM_HEIGHT])
    cam_quat = np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])
    camera.set_world_pose(cam_pos, cam_quat, camera_axes="world")

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
    # Virtual x/y joints act in the WORLD frame: rotate the body command.
    c, s = np.cos(th), np.sin(th)
    v_world = np.array([c * u_safe[0] - s * u_safe[1],
                        s * u_safe[0] + c * u_safe[1]])
    robot.apply_action(ArticulationAction(
        joint_velocities=np.array([v_world[0], v_world[1], u_safe[2]]),
        joint_indices=idx))

    world.step(render=True)
    if step % 60 == 0:
        d_goal = float(np.linalg.norm(base.state[:2] - np.asarray(args.goal)))
        print(f"t={step*PHYSICS_DT:5.1f}s pos=({x:+.2f},{y:+.2f}) "
              f"h={pipeline.debug.h:6.2f} filtered={pipeline.debug.filtered} "
              f"d_goal={d_goal:.2f}")
        if d_goal < 0.3:
            print("reached goal")
            break
    step += 1
  print("done — base should be stopped short of the cone line (x=3.5).")
except Exception:
    traceback.print_exc()

simulation_app.close()
os._exit(0)
