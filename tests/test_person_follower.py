"""
Unit tests for person_follower.py.

All tests run without hardware — no RealSense, no PAROL6 server, no webcam.
Robot client calls are replaced with AsyncMock.
"""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

import person_follower as pf
from mock_camera import MockCamera


# ── px_to_speed ────────────────────────────────────────────────────────────────

def test_px_to_speed_returns_base_for_small_error():
    # 5 px error → 5 // 12 = 0 added → exactly base
    assert pf.px_to_speed(5, base=12, cap=45) == 12


def test_px_to_speed_clamps_to_cap():
    assert pf.px_to_speed(9999, base=12, cap=45) == 45


def test_px_to_speed_increases_with_error():
    s1 = pf.px_to_speed(12, base=12, cap=45)
    s2 = pf.px_to_speed(120, base=12, cap=45)
    assert s2 > s1


def test_px_to_speed_symmetric():
    # sign of error should not matter — caller passes abs value but formula
    # uses abs(error_px), so negative and positive give the same speed
    assert pf.px_to_speed(-50, base=12, cap=45) == pf.px_to_speed(50, base=12, cap=45)


# ── sample_depth ───────────────────────────────────────────────────────────────

class _FakeDepthFrame:
    def __init__(self, data): self._data = data
    def get_data(self): return self._data


def test_sample_depth_returns_median_in_metres():
    data = np.full((480, 640), 2000, dtype=np.uint16)  # 2000 mm = 2.0 m
    frame = _FakeDepthFrame(data)
    result = pf.sample_depth(frame, cx=320, cy=240)
    assert abs(result - 2.0) < 0.01


def test_sample_depth_all_zeros_returns_zero():
    data = np.zeros((480, 640), dtype=np.uint16)
    frame = _FakeDepthFrame(data)
    assert pf.sample_depth(frame, cx=320, cy=240) == 0.0


def test_sample_depth_edge_coord_does_not_crash():
    data = np.full((480, 640), 1500, dtype=np.uint16)
    frame = _FakeDepthFrame(data)
    pf.sample_depth(frame, cx=0, cy=0)       # top-left corner
    pf.sample_depth(frame, cx=639, cy=479)   # bottom-right corner


def test_sample_depth_converts_mm_to_metres():
    data = np.full((480, 640), 1000, dtype=np.uint16)  # 1000 mm
    frame = _FakeDepthFrame(data)
    result = pf.sample_depth(frame, cx=320, cy=240)
    assert abs(result - 1.0) < 0.01


def test_sample_depth_ignores_zero_pixels():
    """Zeros are treated as invalid — only non-zero pixels contribute to median."""
    data = np.zeros((480, 640), dtype=np.uint16)
    # Put 1500 mm in the sample patch and zeros everywhere else
    data[234:246, 314:326] = 1500
    frame = _FakeDepthFrame(data)
    result = pf.sample_depth(frame, cx=320, cy=240, radius=6)
    assert abs(result - 1.5) < 0.01


# ── get_target (detection) ─────────────────────────────────────────────────────

def test_get_target_blank_frame_returns_none_mode():
    """A pure-black frame should never trigger HOG detection."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    body, target, mode = pf.get_target(blank)
    assert mode == "none"
    assert body is None
    assert target is None


def test_get_target_returns_three_tuple():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pf.get_target(blank)
    assert len(result) == 3


def test_get_target_mode_is_valid_string():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, _, mode = pf.get_target(blank)
    assert mode in ("face", "body", "none")


# ── build_camera ───────────────────────────────────────────────────────────────

def test_build_camera_mock_returns_mock_camera():
    cam = pf.build_camera("synthetic")
    assert isinstance(cam, MockCamera)
    cam.stop()


def test_build_camera_mock_webcam_returns_mock_camera():
    """build_camera with webcam source — verify type without opening real device."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        cam = pf.build_camera("webcam")
    assert isinstance(cam, MockCamera)
    cam.stop()


# ── move_robot ─────────────────────────────────────────────────────────────────

def _make_client():
    client = MagicMock()
    client.jog_joint = AsyncMock(return_value=None)
    return client


def test_move_robot_within_deadzone_sends_no_jog():
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=5, error_y=10))
    client.jog_joint.assert_not_called()


def test_move_robot_horizontal_deadzone_boundary():
    """error_x == H_DEADZONE_PX - 1 should not trigger a jog."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=pf.H_DEADZONE_PX - 1, error_y=0))
    client.jog_joint.assert_not_called()


def test_move_robot_person_right_uses_base_neg():
    """Positive error_x means person is to the right → rotate CW (BASE_NEG=6)."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=100, error_y=0))
    joint_used = client.jog_joint.call_args[0][0]
    assert joint_used == pf.BASE_NEG


def test_move_robot_person_left_uses_base_pos():
    """Negative error_x means person is to the left → rotate CCW (BASE_POS=0)."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=-100, error_y=0))
    joint_used = client.jog_joint.call_args[0][0]
    assert joint_used == pf.BASE_POS


def test_move_robot_person_below_uses_tilt_down():
    """Positive error_y means person is below centre → tilt down (TILT_DOWN=7)."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=0, error_y=100))
    joint_used = client.jog_joint.call_args[0][0]
    assert joint_used == pf.TILT_DOWN


def test_move_robot_person_above_uses_tilt_up():
    """Negative error_y means person is above centre → tilt up (TILT_UP=1)."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=0, error_y=-100))
    joint_used = client.jog_joint.call_args[0][0]
    assert joint_used == pf.TILT_UP


def test_move_robot_both_axes_calls_jog_twice():
    """Large errors on both axes → two separate jog_joint calls (gathered)."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=100, error_y=100))
    assert client.jog_joint.call_count == 2


def test_move_robot_speed_increases_with_error():
    """A larger pixel error should produce a higher jog speed."""
    client_small = _make_client()
    client_large = _make_client()
    asyncio.run(pf.move_robot(client_small, error_x=30,  error_y=0))
    asyncio.run(pf.move_robot(client_large, error_x=200, error_y=0))
    speed_small = client_small.jog_joint.call_args[0][1]
    speed_large = client_large.jog_joint.call_args[0][1]
    assert speed_large > speed_small


def test_move_robot_speed_does_not_exceed_max():
    """Speed should never exceed H_MAX_SPEED regardless of error magnitude."""
    client = _make_client()
    asyncio.run(pf.move_robot(client, error_x=99999, error_y=0))
    speed = client.jog_joint.call_args[0][1]
    assert speed <= pf.H_MAX_SPEED
