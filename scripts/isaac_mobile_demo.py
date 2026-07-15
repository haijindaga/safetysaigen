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
parser.add_argument("--ollama-num-gpu", type=int, default=0,
                    help="Ollama GPU layers; 0 = CPU-only (default: the GPU "
                         "is needed by Isaac/SAM3 on an 8 GB card)")
parser.add_argument("--scene", choices=["plain", "warehouse"], default="plain",
                    help="plain = gray ground + primitive cones; warehouse = "
                         "photoreal Isaac warehouse + real cone props "
                         "(use for real VLM/SAM3)")
parser.add_argument("--camera", choices=["own", "jetbot"], default="jetbot",
                    help="jetbot = the robot's built-in camera prim; "
                         "own = create a camera on the chassis")
parser.add_argument("--steps", type=int, default=30000,
                    help="60 Hz physics steps (30000 = 500 sim-seconds; "
                         "slow-VLM runs need the first inference to finish)")
parser.add_argument("--goal", type=float, nargs=2, default=[5.0, 0.0])
parser.add_argument("--perception-every", type=int, default=30,
                    help="physics steps between perception updates (GT mode)")
parser.add_argument("--no-wait-perception", action="store_true",
                    help="allow motion before the first barrier is grounded "
                         "(only sound when v_max satisfies the Theorem-1 "
                         "certificate for your perception latency)")
parser.add_argument("--max-barrier-age", type=float, default=0.0,
                    help="stop if the barrier is older than this many seconds "
                         "(0 = disabled); stop-start mode for slow perception")
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
from core_safety.telemetry import TelemetryWriter

PHYSICS_DT = 1.0 / 60.0
V_MAX, OMEGA_MAX = 0.3, 1.0          # Jetbot-scale limits
CAM_RES = (500, 320)                 # (width, height), matches 2D sim scale

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
import omni.client
from isaacsim.core.utils.stage import add_reference_to_stage

world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
              rendering_dt=PHYSICS_DT)


def _find_cone_prop(assets_root: str) -> str | None:
    """Search warehouse prop folders for a traffic-cone USD."""
    for folder in ("/Isaac/Environments/Simple_Warehouse/Props",
                   "/Isaac/Props/Warehouse", "/Isaac/Props"):
        result, entries = omni.client.list(assets_root + folder)
        if result != omni.client.Result.OK:
            continue
        for e in entries:
            name = e.relative_path
            if "cone" in name.lower() and name.lower().endswith(
                    (".usd", ".usda", ".usdz")):
                return assets_root + folder + "/" + name
    return None


if args.scene == "warehouse":
    from isaacsim.storage.native import get_assets_root_path as _garp
    _assets = _garp()
    add_reference_to_stage(
        usd_path=_assets + "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        prim_path="/World/warehouse")
    cone_usd = _find_cone_prop(_assets)
    print(f"[scene] warehouse loaded; cone prop: {cone_usd}")
    for i, y in enumerate(np.linspace(-1.2, 1.2, 5)):
        if cone_usd:
            add_reference_to_stage(usd_path=cone_usd,
                                   prim_path=f"/World/cone_{i}")
            prim = get_prim_at_path(f"/World/cone_{i}")
            from isaacsim.core.prims import XFormPrim as _X
            _X(f"/World/cone_{i}").set_world_poses(
                positions=np.array([[3.0, float(y), 0.0]]))
        else:
            world.scene.add(FixedCylinder(
                prim_path=f"/World/cone_{i}", name=f"cone_{i}",
                position=np.array([3.0, float(y), 0.2]),
                radius=0.08, height=0.4, color=np.array([1.0, 0.4, 0.0])))
            prim = get_prim_at_path(f"/World/cone_{i}")
        add_update_semantics(prim, "cone")
else:
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

JETBOT_CAM_PRIM = "/World/jetbot/chassis/rgb_camera/jetbot_camera"
if args.camera == "jetbot" and get_prim_at_path(JETBOT_CAM_PRIM).IsValid():
    # Wrap the robot's built-in camera (already positioned/oriented).
    camera = Camera(prim_path=JETBOT_CAM_PRIM, resolution=CAM_RES)
    print("[camera] using jetbot built-in camera")
else:
    # Own camera; identity orientation = look along chassis +x (world axes).
    camera = Camera(prim_path="/World/jetbot/chassis/core_cam",
                    translation=np.array([0.08, 0.0, 0.25]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    resolution=CAM_RES, frequency=20)
    print("[camera] using created core_cam")

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
    vlm = OllamaVLM(model=args.ollama_model, num_gpu=args.ollama_num_gpu)
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
_state = {"last_perception": None}   # wall time of last completed cycle


DEBUG_DIR = Path("results/isaac_debug").resolve()
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[debug] artifacts will be written to: {DEBUG_DIR}")
telemetry = TelemetryWriter(DEBUG_DIR)


def _dump_debug(frame):
    """Save RGB, VLM output, mask overlay, and costmap for inspection.
    Each artifact is saved independently so one failure cannot hide the rest."""
    import time as _time
    import cv2
    ts = _time.strftime("%H%M%S")
    try:
        rgb = np.ascontiguousarray(frame["rgb"][:, :, ::-1])
        cv2.imwrite(str(DEBUG_DIR / f"{ts}_rgb.png"), rgb)
    except Exception as e:
        print(f"[debug] rgb save failed: {e}")
    try:
        c = pipeline.debug.constraints
        txt = "no constraints" if c is None else (
            f"logic: {c.safety_logic}\nclasses: {c.classes}\n"
            f"safe: {[str(p) for p in c.safe]}\n"
            f"unsafe: {[str(p) for p in c.unsafe]}\n"
            f"raw:\n{getattr(vlm, 'last_raw', '')}")
        (DEBUG_DIR / f"{ts}_vlm.txt").write_text(txt, encoding="utf-8")
    except Exception as e:
        print(f"[debug] vlm save failed: {e}")
    try:
        if pipeline.debug.unsafe_mask is not None:
            ov = frame["rgb"].astype(np.float32)
            ov[pipeline.debug.safe_mask, 1] += 120     # green = safe
            ov[pipeline.debug.unsafe_mask, 0] += 120   # red   = unsafe
            ov = np.ascontiguousarray(np.clip(ov, 0, 255).astype(np.uint8)[:, :, ::-1])
            cv2.imwrite(str(DEBUG_DIR / f"{ts}_masks.png"), ov)
    except Exception as e:
        print(f"[debug] mask save failed: {e}")
    try:
        # Top-down costmap: what the barrier actually enforces right now.
        cm = pipeline.costmap
        img = np.full((cm.ny, cm.nx, 3), 128, dtype=np.uint8)   # gray unknown
        observed = (cm.n_safe + cm.n_unsafe).T > 0
        safe = cm.safe_grid().T
        img[observed & safe] = (0, 160, 0)
        img[observed & ~safe] = (0, 0, 200)                     # BGR red
        ix, iy = cm.state_to_cell(base.state[0], base.state[1])
        cv2.circle(img, (ix, iy), 2, (255, 200, 0), -1)          # robot
        img = cv2.resize(img[::-1], (cm.nx * 6, cm.ny * 6),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(DEBUG_DIR / f"{ts}_costmap.png"), img)
    except Exception as e:
        print(f"[debug] costmap save failed: {e}")
    print(f"[debug] saved {ts}_* to {DEBUG_DIR}")


def _perception_worker():
    while simulation_app.is_running():
        _busy.wait(timeout=0.5)
        if not _busy.is_set():
            continue
        with _lock:
            frame = dict(_latest)
        try:
            import time as _time
            t0 = _time.time()
            gt_segmenter.update(frame["labels"], frame["id_to_name"])
            pipeline.update_perception(
                frame["rgb"], frame["depth"], frame["pose"],
                visible_classes=frame["classes"],
                instance_counts=frame["counts"])
            _state["last_perception"] = _time.time()
            _state["cycle_s"] = _time.time() - t0
            print(f"[perception] cycle {_state['cycle_s']:.1f}s")
            _dump_debug(frame)
        except Exception as e:      # keep last good barrier on any failure
            print(f"[perception] update failed: {e}")
        _busy.clear()


threading.Thread(target=_perception_worker, daemon=True).start()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
import os
import traceback

step = 0
try:
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
    # Safe initialization: no motion until the first barrier is grounded
    # (Assumption 1). With slow perception, --max-barrier-age also enforces
    # the paper's stop-start regime instead of the continuous-motion
    # certificate, which our CPU-VLM latency cannot satisfy at this v_max.
    import time as _time
    if pipeline.barrier is None and not args.no_wait_perception:
        u_nom = np.zeros(3)
    elif (args.max_barrier_age > 0 and _state["last_perception"] is not None
          and _time.time() - _state["last_perception"] > args.max_barrier_age):
        u_nom = np.zeros(3)
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
        # Dashboard hook (optional viewer; harmless if nothing reads it).
        telemetry.write_status(
            t=step * PHYSICS_DT, x=float(base.state[0]), y=float(base.state[1]),
            h=(None if not np.isfinite(pipeline.debug.h) else float(pipeline.debug.h)),
            filtered=bool(pipeline.debug.filtered), d_goal=d_goal,
            cycle_s=_state.get("cycle_s"), vlm=args.vlm,
            model=(args.ollama_model if args.vlm == "ollama" else "-"),
            segmenter=args.segmenter, v_max=base.v_max,
            max_barrier_age=args.max_barrier_age,
            perception_every=args.perception_every)
        p = telemetry.read_params()
        if "v_max" in p:
            base.v_max = nominal.v_max = float(p["v_max"])
        if "max_barrier_age" in p:
            args.max_barrier_age = float(p["max_barrier_age"])
        if "perception_every" in p:
            args.perception_every = max(1, int(p["perception_every"]))
        if "tau" in p:
            pipeline.costmap.tau = float(p["tau"])
        if d_goal < 0.25:
            print("reached goal")
            break
    step += 1
  print("done — robot should be stopped at the cone line, not past it "
        "(cones are at x=3.0).")
except Exception:
    traceback.print_exc()   # show the real error before Isaac teardown noise

# Isaac Sim is known to segfault in omni.graph/syntheticdata teardown during
# interpreter finalization (harmless but alarming). Close the app, then skip
# the remaining atexit handlers.
simulation_app.close()
os._exit(0)
