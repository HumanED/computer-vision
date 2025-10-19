#pragma once
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

class RealSenseCamera {
public:
    RealSenseCamera();
    ~RealSenseCamera();
    bool getFrame(cv::Mat &color, cv::Mat &depth);
    rs2_intrinsics getIntrinsics() const;

private:
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2_intrinsics intrinsics;
};
