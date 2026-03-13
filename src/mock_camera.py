import numpy as np
import cv2

COLOR_W, COLOR_H = 640, 480

# Matches the focal-length values from mock_camera.hpp
MOCK_FX  = 608.345
MOCK_FY  = 607.212
MOCK_PPX = 322.442
MOCK_PPY = 248.247


class _DepthFrame:
    """Minimal stand-in for rs2::depth_frame — only `.get_data()` is needed."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


class MockCamera:
    """
    Drop-in camera source for person_follower.py.

    Parameters
    ----------
    source : str
        "synthetic" | "webcam" | "file:<path>"
    """

    def __init__(self, source: str = "synthetic") -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._source == "webcam":
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError("[MOCK] Could not open webcam (index 0).")
        elif self._source.startswith("file:"):
            path = self._source[5:]
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"[MOCK] Could not open video file: {path!r}")
        # synthetic: nothing to open
        print(f"[MOCK] Camera started  source={self._source!r}  "
              f"resolution={COLOR_W}x{COLOR_H}")

    def read(self) -> tuple[np.ndarray | None, _DepthFrame | None]:
        """Return (bgr_frame, depth_frame) or (None, None) on failure."""
        bgr = self._read_color()
        if bgr is None:
            return None, None
        depth = _DepthFrame(self._synthetic_depth())
        return bgr, depth

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Frame generation ───────────────────────────────────────────────────────

    def _read_color(self) -> np.ndarray | None:
        if self._cap is not None:
            ret, bgr = self._cap.read()
            if not ret:
                # loop video files back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bgr = self._cap.read()
            if not ret:
                return None
            return cv2.resize(bgr, (COLOR_W, COLOR_H))
        return self._synthetic_color()

    def _synthetic_color(self) -> np.ndarray:
        """
        Animated geometric pattern — same shapes as mock_camera.hpp but with
        a slowly drifting background so you can see the loop is live.
        """
        frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)

        # Slowly animating gradient background
        t = self._frame_idx / 30.0
        xs = np.linspace(0.0, 1.0, COLOR_W, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, COLOR_H, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        frame[:, :, 0] = ((np.sin(xv * 4 + t) + 1.0) * 50).astype(np.uint8)
        frame[:, :, 1] = (yv * 70).astype(np.uint8)
        frame[:, :, 2] = ((np.cos(yv * 3 - t * 0.5) + 1.0) * 40).astype(np.uint8)

        # Static shapes (mirrors mock_camera.hpp geometry)
        cv2.rectangle(frame, (50,  50),  (200, 200), (200, 200, 200), -1)
        cv2.rectangle(frame, (300, 100), (500, 300), (120, 120, 120), -1)
        cv2.circle(frame,    (100, 350), 50,         (170, 170, 170), -1)
        cv2.circle(frame,    (400, 400), 30,         ( 90,  90,  90), -1)

        # Salt-and-pepper noise for more realistic feature response
        noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Overlay "MOCK" banner so it's obvious this is not real hardware
        cv2.putText(frame, "MOCK CAMERA", (COLOR_W // 2 - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        self._frame_idx += 1
        return frame

    def _synthetic_depth(self) -> np.ndarray:
        """
        Synthetic depth map in mm (uint16), matching mock_camera.hpp values.
        Background ~1500 mm, shapes at various depths, ±50 mm noise.
        """
        depth = np.full((COLOR_H, COLOR_W), 1500, dtype=np.int32)

        # Mirror the depth blobs from mock_camera.hpp
        depth[50:200,   50:200]  = 1000   # 1.0 m
        depth[100:300, 300:500]  = 2000   # 2.0 m
        cy, cx, r = 350, 100, 50
        yy, xx = np.ogrid[:COLOR_H, :COLOR_W]
        depth[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1500  # 1.5 m
        depth[(yy - 400) ** 2 + (xx - 400) ** 2 <= 30 ** 2] = 3000  # 3.0 m

        noise = np.random.randint(-50, 51, depth.shape, dtype=np.int32)
        depth = np.clip(depth + noise, 0, 65535).astype(np.uint16)
        return depth
