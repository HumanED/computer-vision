#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

struct MapPoint {
    cv::Point3f position;
    cv::KeyPoint keypoint;
};
