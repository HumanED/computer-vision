# PAROL6 Vision Scripts

Experimental Python scripts for vision-guided control of the PAROL6 robot arm using an Intel RealSense depth camera. The goal is to explore behaviours where the robot perceives and reacts to its environment — person following is one example, with more planned.

Hardware access is limited, so all scripts support a `--mock` camera mode and a `--sim` robot mode so development and testing can happen offline.

## Project layout

```
src/
  person_follower.py   Vision-guided person tracking
  go_home.py           Move the robot to its home position
mock_camera.py         Mock RealSense camera (synthetic / webcam / video file)
tests/                 Unit tests (no hardware required)
```

## Features

| Script | Description |
|---|---|
| `person_follower.py` | Detects a person with HOG + Haar and pans/tilts the robot to keep them centred |
| `go_home.py` | Sends the robot to its home position |

More behaviours will be added here as the project grows.

## Setup

**1. Install the parol6 package** (from the repository root):
```bash
pip install ..
```

**2. Install script dependencies:**
```bash
pip install -r requirements.txt
```

> `pyrealsense2` is only needed when running against a real RealSense camera.
> It can be skipped when using `--mock`.

**3. Start the PAROL6 server** (in a separate terminal):
```bash
parol6-server --log-level=INFO
```

## Running without hardware

Every script supports two flags for offline development:

| Flag | Effect |
|---|---|
| `--sim` | Commands are sent to the server but the robot does not physically move |
| `--mock [SOURCE]` | Replaces the RealSense with a mock camera (see below) |

Combine them to run with no hardware at all:
```bash
python src/person_follower.py --sim --mock
```

## Mock camera

`mock_camera.py` generates synthetic color and depth frames so scripts can be developed and tested without a RealSense attached. Three sources are available:

| Source | What you get |
|---|---|
| `synthetic` (default) | Animated geometric shapes + noise. No person detected — good for testing the display loop and robot comms. |
| `webcam` | Your webcam feed with a synthetic depth map. Detection works — test with yourself. |
| `file:PATH` | A video file with a synthetic depth map. Loops automatically. Good for repeatable tests. |

Intrinsics match the C++ mock in `tests/mocks/mock_camera.hpp`: fx=608.3, fy=607.2, cx=322.4, cy=248.2 (640×480).

## Usage examples

```bash
# Person follower — real hardware
python src/person_follower.py

# Person follower — fully offline
python src/person_follower.py --sim --mock

# Person follower — webcam image, robot in simulator
python src/person_follower.py --sim --mock webcam

# Person follower — replay a video clip
python src/person_follower.py --mock file:clip.mp4

# Send robot home
python src/go_home.py
```

Press **`q`** in any preview window to quit.

## Tests

```bash
python -m pytest tests/ -v
```

All tests run without hardware. The mock camera and pure-logic functions in
`person_follower.py` (speed calculation, depth sampling, robot direction) are
covered without needing a RealSense or a running PAROL6 server.

## Notes

- The Haar face cascade path in `person_follower.py` (`_HAAR_PATH`) is hardcoded to the conda `humaned` environment. Update it if your OpenCV lives elsewhere:
  ```bash
  python -c "import cv2, os; print(os.path.dirname(cv2.__file__))"
  ```
- Depth values from the mock camera are in millimetres (uint16), matching the RealSense z16 format.
