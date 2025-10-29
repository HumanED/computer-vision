# ORB-SLAM2-RGBD-D434i

This project builds a cleaned, RGB-D-only version of ORB-SLAM2 specifically configured for Intel RealSense D435i. It includes a C++ implementation with camera calibration settings, feature extraction, tracking, local mapping, and loop closure.
Intended as a semester-long project baseline for research and improvement.

## Pipeline
- Initialization
- Tracking
- Local Mapping
- Loop Closing
## Building

```
sudo bash setup_realsense.sh
```

## Library

## Repo Tree
```
ORB-SLAM2-RGBD-D435i/
├── cmake_modules/
│   └── ...         # Custom CMake module to locate Eigen3 headers on the system, if CMake cannot find it automatically
├── RGB-D/
│   └── ...         # Entry points
├── include/
│   └── ...         # ORB-SLAM2 core classes
├── src/
│   └── ...         # Implementations
├── Thirdparty/
│   ├── DBoW2/
│   └── g2o/
├── Vocabulary/
│   └── ...          # ORB feature vocabulary used for loop closure detection in ORB-SLAM2
├── CMakeLists.txt
├── Dependencies.md
├── LICENSE.txt
├── .gitignore
├── build.sh
├── License-gpl.txt
└── README.md
```

# Contributors
Hwayeon, Arjav, Ryan, Teymur
