#pragma once
#include "realsense_camera.hpp"
#include "orb_detector.hpp"
#include "map_point.hpp"
#include <vector>

class Initializer {
public:
    bool initialize(RealSenseCamera &camera);
    std::vector<MapPoint> backProjectKeypoints(
        const std::vector<cv::KeyPoint> &keypoints,
        const cv::Mat &depth,
        const rs2_intrinsics &intrinsics);
private:
};
