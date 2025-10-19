#pragma once
#include <opencv2/opencv.hpp>

class ORBDetector {
public:
    ORBDetector(int nfeatures = 1000);
    void detectAndCompute(const cv::Mat &img,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors);

private:
    cv::Ptr<cv::ORB> orb;
};
