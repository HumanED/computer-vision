"""
Unit tests for mock_camera.py.

All tests run without any hardware — no RealSense, no webcam required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mock_camera import MockCamera, _DepthFrame, COLOR_W, COLOR_H


# ── _DepthFrame ────────────────────────────────────────────────────────────────

def test_depth_frame_returns_same_array():
    data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    frame = _DepthFrame(data)
    result = frame.get_data()
    np.testing.assert_array_equal(result, data)


def test_depth_frame_preserves_dtype():
    data = np.zeros((10, 10), dtype=np.uint16)
    assert _DepthFrame(data).get_data().dtype == np.uint16


# ── MockCamera — synthetic source ──────────────────────────────────────────────

@pytest.fixture
def synthetic():
    cam = MockCamera("synthetic")
    cam.start()
    yield cam
    cam.stop()


def test_synthetic_start_does_not_raise():
    cam = MockCamera("synthetic")
    cam.start()  # should not raise
    cam.stop()


def test_synthetic_read_color_shape(synthetic):
    bgr, _ = synthetic.read()
    assert bgr.shape == (COLOR_H, COLOR_W, 3)


def test_synthetic_read_color_dtype(synthetic):
    bgr, _ = synthetic.read()
    assert bgr.dtype == np.uint8


def test_synthetic_read_depth_shape(synthetic):
    _, depth = synthetic.read()
    assert depth.get_data().shape == (COLOR_H, COLOR_W)


def test_synthetic_read_depth_dtype(synthetic):
    _, depth = synthetic.read()
    assert depth.get_data().dtype == np.uint16


def test_synthetic_read_depth_range(synthetic):
    _, depth = synthetic.read()
    arr = depth.get_data()
    assert arr.min() >= 0
    assert arr.max() <= 65535


def test_frame_idx_increments(synthetic):
    for _ in range(5):
        synthetic.read()
    assert synthetic._frame_idx == 5


def test_synthetic_read_returns_non_none(synthetic):
    bgr, depth = synthetic.read()
    assert bgr is not None
    assert depth is not None


def test_depth_background_near_1500mm(synthetic):
    """Most of the frame is background at ~1500 mm."""
    _, depth = synthetic.read()
    arr = depth.get_data().astype(np.float32)
    # Exclude the override blobs by sampling a background region
    bg = arr[220:280, 220:280]  # central region, no shape overrides
    assert abs(float(np.median(bg)) - 1500) < 150


def test_depth_rect_blob_near_1000mm(synthetic):
    """The top-left rectangle is set to ~1000 mm in mock_camera.hpp."""
    _, depth = synthetic.read()
    arr = depth.get_data().astype(np.float32)
    patch = arr[80:180, 80:180]  # well inside [50:200, 50:200]
    assert abs(float(np.median(patch)) - 1000) < 150


def test_depth_rect_blob_near_2000mm(synthetic):
    """The right rectangle is set to ~2000 mm."""
    _, depth = synthetic.read()
    arr = depth.get_data().astype(np.float32)
    patch = arr[150:250, 350:450]  # well inside [100:300, 300:500]
    assert abs(float(np.median(patch)) - 2000) < 150


def test_stop_synthetic_does_not_raise(synthetic):
    synthetic.stop()   # second stop on a synthetic camera is harmless
    synthetic.stop()


def test_stop_clears_cap_to_none():
    """For non-synthetic sources, stop() should release and null the capture."""
    cam = MockCamera("synthetic")
    cam.start()
    cam._cap = MagicMock()
    cam.stop()
    assert cam._cap is None


# ── MockCamera — webcam / file error handling ──────────────────────────────────

def test_webcam_start_raises_when_device_unavailable():
    cam = MockCamera("webcam")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="webcam"):
            cam.start()


def test_file_start_raises_when_file_missing():
    cam = MockCamera("file:/nonexistent/clip.mp4")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="clip.mp4"):
            cam.start()


def test_read_returns_none_tuple_when_cap_exhausted():
    """If VideoCapture.read() fails even after rewind, read() → (None, None)."""
    cam = MockCamera("file:/some/video.mp4")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)  # always fails
    with patch("cv2.VideoCapture", return_value=mock_cap):
        cam.start()
    bgr, depth = cam.read()
    assert bgr is None
    assert depth is None
