"""
avoid_obstacles.py — PAROL6 + RealSense D435 real-time obstacle avoidance
==========================================================================

Run against real hardware:
    python src/avoid_obstacles.py

Run fully offline (mock camera + robot simulator):
    python src/avoid_obstacles.py --sim --mock

Run with webcam image and simulated robot:
    python src/avoid_obstacles.py --sim --mock webcam

Press 'q' in the preview window to quit.

──────────────────────────────────────────────────────────────────────────────
ROS 2 / MoveIt 2 CONFIGURATION REFERENCE  (deploy separately if using MoveIt)
──────────────────────────────────────────────────────────────────────────────

--- moveit_sensors.yaml ---

sensors:
  - point_cloud_sensor

point_cloud_sensor:
  sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
  point_cloud_topic: /camera/depth/color/points
  max_range: 2.0
  padding_offset: 0.01
  padding_scale: 1.0
  point_subsample: 1
  filtered_cloud_topic: /filtered_cloud

octomap_resolution: 0.02        # 2 cm voxels
octomap_frame: world
max_range: 2.0

--- robot_self_filter.yaml ---

self_filter:
  robot_description: robot_description
  sensor_frame: camera_depth_optical_frame
  subsample_value: 0.0
  min_sensor_dist: 0.01
  links:
    - {name: base_link, padding: 0.01, scale: 1.1}
    - {name: link_1,    padding: 0.01, scale: 1.1}
    - {name: link_2,    padding: 0.01, scale: 1.1}
    - {name: link_3,    padding: 0.01, scale: 1.1}
    - {name: link_4,    padding: 0.01, scale: 1.1}
    - {name: link_5,    padding: 0.01, scale: 1.1}
    - {name: link_6,    padding: 0.01, scale: 1.1}
    - {name: tool_link, padding: 0.01, scale: 1.1}

--- chomp_planning.yaml ---

planning_plugin: chomp_interface/CHOMPPlanner
request_adapters: >-
  default_planner_request_adapters/AddTimeParameterization
  default_planner_request_adapters/FixWorkspaceBounds
  default_planner_request_adapters/FixStartStateBounds
  default_planner_request_adapters/FixStartStateCollision
  default_planner_request_adapters/FixStartStatePathConstraints
chomp:
  learning_rate: 0.01
  ridge_factor: 0.01
  planning_time_limit: 10.0
  max_iterations: 200
  max_iterations_after_collision_free: 5
  smoothness_cost_weight: 0.1
  obstacle_cost_weight: 1.0
  dynamic_obstacle_cost_weight: 0.0
  use_hamiltonian_monte_carlo: false
  enable_failure_recovery: true
  max_recovery_attempts: 5
  trajectory_initialization_method: quintic-cubic-blend

# STOMP alternative — replace the chomp block above with:
# stomp:
#   num_timesteps: 60
#   num_iterations: 40
#   num_iterations_after_valid: 0
#   num_rollouts: 30
#   max_rollouts: 30
#   initialization_method: 1   # LINEAR_INTERPOLATION
#   control_cost_weight: 0.1
"""

import argparse
import asyncio
import time
from typing import Any

import cv2
import numpy as np

from parol6 import AsyncRobotClient

# ── Configuration ──────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 5001

COLOR_W, COLOR_H, FPS = 640, 480, 30
DEPTH_MM_TO_M = 0.001

# ROI within the depth frame used for obstacle detection.
# Covers the central workspace area visible to the camera.
ROI = (120, 60, 520, 420)   # (x1, y1, x2, y2)

# Velocity scaling thresholds
STOP_DISTANCE_M = 0.40   # below this → e-stop (disable robot)
SLOW_DISTANCE_M = 1.20   # above this → full speed
MIN_VELOCITY_SCALE = 0.0
MAX_VELOCITY_SCALE = 1.0

SEND_INTERVAL = 0.10   # seconds between robot commands


# ── Depth helpers ──────────────────────────────────────────────────────────────

def nearest_obstacle_m(depth_frame: Any, roi: tuple[int, int, int, int] | None = None) -> float:
    """
    Return the distance in metres to the nearest non-zero pixel in *depth_frame*.

    Parameters
    ----------
    depth_frame:
        Object with a ``.get_data()`` method returning a uint16 numpy array
        (values in millimetres), matching both the RealSense frame interface
        and ``mock_camera._DepthFrame``.
    roi:
        (x1, y1, x2, y2) crop applied before scanning.  ``None`` → full frame.

    Returns
    -------
    float
        Nearest obstacle distance in metres.  ``SLOW_DISTANCE_M`` is returned
        when no valid pixels are found (assume clear).
    """
    arr = np.asanyarray(depth_frame.get_data())
    if roi is not None:
        x1, y1, x2, y2 = roi
        arr = arr[y1:y2, x1:x2]
    valid = arr[arr > 0]
    if valid.size == 0:
        return SLOW_DISTANCE_M
    return float(np.min(valid)) * DEPTH_MM_TO_M


def compute_velocity_scale(nearest_dist: float) -> float:
    """
    Piecewise-linear ramp between ``STOP_DISTANCE_M`` and ``SLOW_DISTANCE_M``.

    Returns ``MIN_VELOCITY_SCALE`` at or below the stop threshold,
    ``MAX_VELOCITY_SCALE`` at or above the slow threshold.
    """
    if nearest_dist <= STOP_DISTANCE_M:
        return MIN_VELOCITY_SCALE
    if nearest_dist >= SLOW_DISTANCE_M:
        return MAX_VELOCITY_SCALE
    t = (nearest_dist - STOP_DISTANCE_M) / (SLOW_DISTANCE_M - STOP_DISTANCE_M)
    return float(np.clip(
        MIN_VELOCITY_SCALE + t * (MAX_VELOCITY_SCALE - MIN_VELOCITY_SCALE),
        MIN_VELOCITY_SCALE,
        MAX_VELOCITY_SCALE,
    ))


# ── Camera abstraction ─────────────────────────────────────────────────────────

def build_camera(mock_source: str | None):
    """Return a started camera object (real RealSense or mock)."""
    if mock_source is not None:
        from mock_camera import MockCamera
        cam = MockCamera(source=mock_source)
        cam.start()
        return cam
    return _RealSenseCamera()


class _RealSenseCamera:
    """Thin wrapper around the RealSense pipeline with the same interface as MockCamera."""

    def __init__(self) -> None:
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline: Any = None
        self._align: Any = None

    def start(self) -> None:
        rs = self._rs
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16,  FPS)
        pipeline.start(cfg)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        print("[INFO] RealSense started.")

    def read(self) -> tuple[np.ndarray | None, Any]:
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return np.asanyarray(color_frame.get_data()), depth_frame

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()


# ── Visualisation ──────────────────────────────────────────────────────────────

def _draw_overlay(
    bgr: np.ndarray,
    nearest_dist: float,
    velocity_scale: float,
    was_estopped: bool,
) -> np.ndarray:
    """Draw the ROI, distance readout, and velocity-scale bar onto *bgr* (copy)."""
    vis = bgr.copy()
    x1, y1, x2, y2 = ROI

    # ROI rectangle
    roi_color = (0, 0, 220) if velocity_scale == MIN_VELOCITY_SCALE else (0, 200, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), roi_color, 2)

    # Velocity-scale bar (right edge of ROI)
    bar_x = x2 + 12
    bar_top = y1
    bar_bot = y2
    bar_h = bar_bot - bar_top
    fill_h = int(velocity_scale * bar_h)
    cv2.rectangle(vis, (bar_x, bar_top), (bar_x + 18, bar_bot), (60, 60, 60), -1)
    bar_color = (0, 200, 0) if velocity_scale > 0.5 else (0, 100, 220) if velocity_scale > 0 else (0, 0, 200)
    cv2.rectangle(vis, (bar_x, bar_bot - fill_h), (bar_x + 18, bar_bot), bar_color, -1)
    cv2.rectangle(vis, (bar_x, bar_top), (bar_x + 18, bar_bot), (180, 180, 180), 1)

    # Text labels
    label = f"Nearest: {nearest_dist:.2f} m   Scale: {velocity_scale:.2f}"
    cv2.putText(vis, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if was_estopped:
        cv2.putText(vis, "E-STOP", (COLOR_W // 2 - 60, COLOR_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

    return vis


# ── Avoidance loop ─────────────────────────────────────────────────────────────

async def avoidance_loop(client: AsyncRobotClient, camera: Any) -> None:
    """
    Main control loop:
      - Reads depth frames continuously.
      - Computes nearest obstacle distance inside ROI.
      - Derives a velocity scale via ``compute_velocity_scale()``.
      - Disables (e-stops) the robot when scale == 0; re-enables when it recovers.
      - Displays a live preview window.
    """
    print("[INFO] Starting obstacle avoidance loop. Press 'q' to quit.")
    await asyncio.sleep(1.0)

    estopped = False
    last_send = 0.0

    try:
        while True:
            bgr, depth_frame = camera.read()
            if bgr is None or depth_frame is None:
                await asyncio.sleep(0)
                continue

            nearest = nearest_obstacle_m(depth_frame, roi=ROI)
            scale   = compute_velocity_scale(nearest)

            now = time.monotonic()
            if now - last_send >= SEND_INTERVAL:
                if scale == MIN_VELOCITY_SCALE and not estopped:
                    await client.disable()
                    estopped = True
                    print(f"[WARN] E-STOP — obstacle at {nearest:.2f} m")
                elif scale > MIN_VELOCITY_SCALE and estopped:
                    await client.enable()
                    estopped = False
                    print(f"[INFO] Resuming — nearest obstacle {nearest:.2f} m  scale={scale:.2f}")
                last_send = now

            vis = _draw_overlay(bgr, nearest, scale, estopped)
            cv2.imshow("PAROL6 Obstacle Avoidance", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

            await asyncio.sleep(0)

    finally:
        camera.stop()
        cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> int:
    async with AsyncRobotClient(host=HOST, port=PORT, timeout=3.0) as client:
        ready = await client.wait_for_server_ready(timeout=8.0)
        if not ready:
            print("[ERROR] PAROL6 server not ready. Is it running?")
            return 1
        print("[INFO] Connected to PAROL6 server.")

        if args.sim:
            await client.simulator_on()
            print("[INFO] Simulator mode ON — no real motion.")

        camera = build_camera(args.mock if args.mock is not None else None)
        await avoidance_loop(client, camera)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAROL6 real-time obstacle avoidance using a RealSense depth camera."
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Simulator mode: commands sent but robot does not move.",
    )
    parser.add_argument(
        "--mock", nargs="?", const="synthetic", metavar="SOURCE",
        help=(
            "Use mock camera instead of real RealSense. "
            "SOURCE: 'synthetic' (default), 'webcam', or 'file:PATH'."
        ),
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
