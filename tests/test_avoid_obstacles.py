"""
Unit tests for avoid_obstacles.py.

All tests run without hardware — no RealSense, no PAROL6 server, no webcam.
Robot client calls are replaced with AsyncMock.
"""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import avoid_obstacles as ao
from mock_camera import MockCamera


# ── nearest_obstacle_m ─────────────────────────────────────────────────────────

class _FakeDepthFrame:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


def _frame(value_mm: int, shape: tuple = (480, 640)) -> _FakeDepthFrame:
    return _FakeDepthFrame(np.full(shape, value_mm, dtype=np.uint16))


def test_nearest_obstacle_full_frame_uniform():
    frame = _frame(1000)
    result = ao.nearest_obstacle_m(frame)
    assert abs(result - 1.0) < 0.01


def test_nearest_obstacle_converts_mm_to_metres():
    frame = _frame(500)
    assert abs(ao.nearest_obstacle_m(frame) - 0.5) < 0.01


def test_nearest_obstacle_returns_minimum_not_mean():
    """With a near pixel buried in far background, min is returned."""
    data = np.full((480, 640), 2000, dtype=np.uint16)
    data[240, 320] = 300   # single near pixel
    frame = _FakeDepthFrame(data)
    result = ao.nearest_obstacle_m(frame)
    assert abs(result - 0.3) < 0.01


def test_nearest_obstacle_all_zeros_returns_slow_distance():
    """Zero pixels are invalid — should fall back to SLOW_DISTANCE_M."""
    frame = _frame(0)
    assert ao.nearest_obstacle_m(frame) == ao.SLOW_DISTANCE_M


def test_nearest_obstacle_roi_clips_frame():
    """Pixels outside the ROI must not affect the result."""
    data = np.full((480, 640), 3000, dtype=np.uint16)
    # Place a near pixel OUTSIDE the ROI
    data[10, 10] = 100
    frame = _FakeDepthFrame(data)
    roi = (120, 60, 520, 420)
    result = ao.nearest_obstacle_m(frame, roi=roi)
    assert abs(result - 3.0) < 0.05


def test_nearest_obstacle_roi_detects_inside():
    """A near pixel INSIDE the ROI must be returned."""
    data = np.full((480, 640), 3000, dtype=np.uint16)
    data[240, 320] = 400   # inside default ROI
    frame = _FakeDepthFrame(data)
    roi = (120, 60, 520, 420)
    result = ao.nearest_obstacle_m(frame, roi=roi)
    assert abs(result - 0.4) < 0.01


def test_nearest_obstacle_no_roi_uses_full_frame():
    data = np.full((480, 640), 1500, dtype=np.uint16)
    data[5, 5] = 200
    frame = _FakeDepthFrame(data)
    result = ao.nearest_obstacle_m(frame, roi=None)
    assert abs(result - 0.2) < 0.01


# ── compute_velocity_scale ─────────────────────────────────────────────────────

def test_scale_at_stop_distance_is_min():
    assert ao.compute_velocity_scale(ao.STOP_DISTANCE_M) == ao.MIN_VELOCITY_SCALE


def test_scale_below_stop_distance_is_min():
    assert ao.compute_velocity_scale(0.0) == ao.MIN_VELOCITY_SCALE


def test_scale_at_slow_distance_is_max():
    assert ao.compute_velocity_scale(ao.SLOW_DISTANCE_M) == ao.MAX_VELOCITY_SCALE


def test_scale_above_slow_distance_is_max():
    assert ao.compute_velocity_scale(ao.SLOW_DISTANCE_M + 10.0) == ao.MAX_VELOCITY_SCALE


def test_scale_midpoint_is_between_min_and_max():
    mid = (ao.STOP_DISTANCE_M + ao.SLOW_DISTANCE_M) / 2
    scale = ao.compute_velocity_scale(mid)
    assert ao.MIN_VELOCITY_SCALE < scale < ao.MAX_VELOCITY_SCALE


def test_scale_is_monotonically_increasing():
    distances = np.linspace(ao.STOP_DISTANCE_M, ao.SLOW_DISTANCE_M, 20)
    scales = [ao.compute_velocity_scale(float(d)) for d in distances]
    assert all(scales[i] <= scales[i + 1] for i in range(len(scales) - 1))


def test_scale_never_exceeds_max():
    for d in [0.0, 0.1, 0.5, 1.0, 5.0, 100.0]:
        assert ao.compute_velocity_scale(d) <= ao.MAX_VELOCITY_SCALE


def test_scale_never_below_min():
    for d in [0.0, 0.1, 0.5, 1.0, 5.0, 100.0]:
        assert ao.compute_velocity_scale(d) >= ao.MIN_VELOCITY_SCALE


# ── build_camera ───────────────────────────────────────────────────────────────

def test_build_camera_mock_returns_mock_camera():
    cam = ao.build_camera("synthetic")
    assert isinstance(cam, MockCamera)
    cam.stop()


def test_build_camera_mock_webcam_returns_mock_camera():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        cam = ao.build_camera("webcam")
    assert isinstance(cam, MockCamera)
    cam.stop()


# ── avoidance_loop — robot commands ───────────────────────────────────────────

def _make_client() -> MagicMock:
    client = MagicMock()
    client.disable = AsyncMock(return_value=True)
    client.enable  = AsyncMock(return_value=True)
    return client


def _make_camera(depth_mm: int) -> MagicMock:
    """Camera that always returns a plain BGR frame and a depth frame at *depth_mm*."""
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_data = np.full((480, 640), depth_mm, dtype=np.uint16)
    depth_frame = _FakeDepthFrame(depth_data)

    camera = MagicMock()
    camera.read.return_value = (bgr, depth_frame)
    camera.stop = MagicMock()
    return camera


async def _run_one_tick(client, camera):
    """Drive avoidance_loop for a single iteration then quit via cv2 mock."""
    call_count = 0

    def _waitkey(_delay):
        nonlocal call_count
        call_count += 1
        return ord("q") if call_count >= 1 else 0

    with patch("cv2.imshow"), patch("cv2.waitKey", side_effect=_waitkey), \
         patch("cv2.destroyAllWindows"):
        await ao.avoidance_loop(client, camera)


def test_loop_estops_when_obstacle_too_close():
    """Depth below STOP threshold → disable() called."""
    client = _make_client()
    depth_mm = int(ao.STOP_DISTANCE_M * 1000 * 0.5)   # clearly inside stop zone
    camera = _make_camera(depth_mm)

    asyncio.run(_run_one_tick(client, camera))
    client.disable.assert_called()


def test_loop_does_not_estop_when_clear():
    """Depth well above SLOW threshold → disable() never called."""
    client = _make_client()
    depth_mm = int(ao.SLOW_DISTANCE_M * 1000 * 2)     # clearly outside slow zone
    camera = _make_camera(depth_mm)

    asyncio.run(_run_one_tick(client, camera))
    client.disable.assert_not_called()


def test_loop_stops_camera_on_exit():
    """camera.stop() must always be called (finally block)."""
    client = _make_client()
    camera = _make_camera(2000)

    asyncio.run(_run_one_tick(client, camera))
    camera.stop.assert_called_once()
