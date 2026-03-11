"""
PAROL6 Person Follower — Intel RealSense + HOG body + Haar face detection.

Behaviour
---------
- Detects people using OpenCV's HOG pedestrian detector.
- Detects faces using Haar cascade within the body region.
- Primary target: face center. Fallback: upper-body center (head area).
- Joint 1 (base) pans left/right to keep target horizontally centred.
- Joint 2 (shoulder) tilts up/down to keep face vertically centred.
- Follows the largest/closest person when multiple are detected.

Requirements
------------
    pip install pyrealsense2 opencv-python numpy

Usage
-----
Assumes a PAROL6 server is already running at 127.0.0.1:5001.
Start it with:  parol6 server --port 5001

    python my_script/person_follower.py
    python my_script/person_follower.py --sim  # simulator mode
"""

import argparse
import asyncio
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from parol6 import AsyncRobotClient

# ── Configuration ──────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 5001

COLOR_W, COLOR_H, FPS = 640, 480, 30
DEPTH_MM_TO_M = 0.001

# HOG body detection
HOG_WIN_STRIDE = (8, 8)
HOG_PADDING    = (4, 4)
HOG_SCALE      = 1.05

# Haar face detection
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 4
FACE_MIN_SIZE      = (40, 40)

# Tracking control — horizontal (base pan)
H_DEADZONE_PX  = 20
H_BASE_SPEED   = 12
H_MAX_SPEED    = 45

# Tracking control — vertical (shoulder tilt)
V_DEADZONE_PX  = 25
V_BASE_SPEED   = 8
V_MAX_SPEED    = 30

JOG_DURATION   = 0.12
SEND_INTERVAL  = 0.12

# PAROL6 joint indices  (0-5 = positive dir, 6-11 = negative dir for joints 1-6)
BASE_POS  = 0    # base rotate CCW (person right of centre → move CW)
BASE_NEG  = 6    # base rotate CW
TILT_UP   = 1    # joint 2 positive = tilt head up
TILT_DOWN = 7    # joint 2 negative = tilt head down


# ── Detectors ──────────────────────────────────────────────────────────────────

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_HAAR_PATH = "/home/teymur/.conda/envs/humaned/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_HAAR_PATH)


def best_body(bgr: np.ndarray) -> tuple | None:
    """Return the largest HOG body bounding box, or None."""
    rects, weights = _hog.detectMultiScale(
        bgr,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE,
    )
    if len(rects) == 0:
        return None
    # pick the rect with the greatest area (closest / largest person)
    areas = [w * h for (x, y, w, h) in rects]
    return tuple(rects[int(np.argmax(areas))])


def find_face_in_roi(bgr: np.ndarray, body: tuple) -> tuple | None:
    """
    Search for a face in the upper ~55 % of the body bounding box.
    Returns face bbox in full-frame coordinates, or None.
    """
    x, y, w, h = body
    roi_h = int(h * 0.55)
    roi = bgr[y: y + roi_h, x: x + w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    if len(faces) == 0:
        return None
    # pick the largest face in ROI
    areas = [fw * fh for (fx, fy, fw, fh) in faces]
    fx, fy, fw, fh = faces[int(np.argmax(areas))]
    return (x + fx, y + fy, fw, fh)   # convert to full-frame coords


def get_target(bgr: np.ndarray) -> tuple[tuple | None, tuple | None, str]:
    """
    Returns (body_box, target_box, mode).
    target_box is the region to centre on (face if found, else upper body).
    mode: 'face' | 'body' | 'none'
    """
    body = best_body(bgr)
    if body is None:
        return None, None, "none"

    face = find_face_in_roi(bgr, body)
    if face is not None:
        return body, face, "face"

    # Fallback: use upper-centre of body as face proxy
    x, y, w, h = body
    proxy_h = int(h * 0.30)
    proxy_y = y + int(h * 0.05)
    proxy = (x + w // 4, proxy_y, w // 2, proxy_h)
    return body, proxy, "body"


# ── RealSense ──────────────────────────────────────────────────────────────────

def start_realsense() -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, FPS)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align


def sample_depth(depth_frame, cx: int, cy: int, radius: int = 6) -> float:
    arr = np.asanyarray(depth_frame.get_data())
    h, w = arr.shape
    patch = arr[
        max(0, cy - radius):min(h, cy + radius),
        max(0, cx - radius):min(w, cx + radius),
    ]
    valid = patch[patch > 0]
    return float(np.median(valid)) * DEPTH_MM_TO_M if valid.size else 0.0


# ── Robot control ──────────────────────────────────────────────────────────────

def px_to_speed(error_px: int, base: int, cap: int) -> int:
    return min(base + int(abs(error_px) / 12), cap)


async def move_robot(client: AsyncRobotClient, error_x: int, error_y: int) -> None:
    """Send pan and/or tilt jog commands based on pixel errors."""
    tasks = []

    if abs(error_x) >= H_DEADZONE_PX:
        # person right of centre → rotate CW (BASE_NEG)
        joint = BASE_NEG if error_x > 0 else BASE_POS
        speed = px_to_speed(error_x, H_BASE_SPEED, H_MAX_SPEED)
        tasks.append(client.jog_joint(joint, speed, duration=JOG_DURATION))

    if abs(error_y) >= V_DEADZONE_PX:
        # face below centre → tilt down (TILT_DOWN)
        joint = TILT_DOWN if error_y > 0 else TILT_UP
        speed = px_to_speed(error_y, V_BASE_SPEED, V_MAX_SPEED)
        tasks.append(client.jog_joint(joint, speed, duration=JOG_DURATION))

    if tasks:
        await asyncio.gather(*tasks)


# ── Tracking loop ──────────────────────────────────────────────────────────────

async def tracking_loop(client: AsyncRobotClient) -> None:
    pipeline, align = start_realsense()
    print("[INFO] RealSense started. Press 'q' to quit.")
    await asyncio.sleep(1.0)

    cx_img = COLOR_W // 2
    cy_img = COLOR_H // 2
    last_send = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            body, target, mode = get_target(bgr)

            vis = bgr.copy()
            # crosshair
            cv2.line(vis, (cx_img, 0),       (cx_img, COLOR_H), (255, 255, 0), 1)
            cv2.line(vis, (0, cy_img),        (COLOR_W, cy_img), (255, 255, 0), 1)

            if body is not None:
                bx, by, bw, bh = body
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 200, 0), 1)

            if target is not None:
                tx, ty, tw, th = target
                tcx = tx + tw // 2
                tcy = ty + th // 2
                error_x = tcx - cx_img
                error_y = tcy - cy_img
                depth_m = sample_depth(depth_frame, tcx, tcy)

                color = (0, 255, 0) if mode == "face" else (0, 165, 255)
                cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), color, 2)
                cv2.circle(vis, (tcx, tcy), 5, color, -1)
                cv2.line(vis, (cx_img, cy_img), (tcx, tcy), color, 1)

                label = f"[{mode}] ex={error_x:+d} ey={error_y:+d} {depth_m:.2f}m"
                cv2.putText(vis, label, (tx, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                now = time.monotonic()
                if now - last_send >= SEND_INTERVAL:
                    await move_robot(client, error_x, error_y)
                    last_send = now
            else:
                cv2.putText(vis, "[no person]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("PAROL6 Person Follower", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

            await asyncio.sleep(0)

    finally:
        pipeline.stop()
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

        await tracking_loop(client)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the PAROL6 robot follow a person using RealSense depth camera."
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Enable simulator mode (commands sent but robot does not move).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
