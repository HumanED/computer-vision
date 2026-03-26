#pragma once
#include "realsense_camera.hpp"
#include "orb_detector.hpp"
#include "map_point.hpp"
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <vector>

// Structure to hold initial map data
struct InitialMapData {
    cv::Mat color;
    cv::Mat depth;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<MapPoint> map_points;
    Eigen::Matrix4d initial_pose;
    int num_frames_used;  // Track how many frames were used for initialization
    
    InitialMapData() {
        initial_pose = Eigen::Matrix4d::Identity();
        num_frames_used = 1;
    }
};

class Initializer {
public:
    // Single-frame initialization (backward compatibility)
    bool initialize(RealSenseCamera &camera);
    bool initialize(RealSenseCamera &camera, InitialMapData &initial_data);
    
    // Multi-frame initialization (recommended for better reliability)
    bool initializeMultiFrame(RealSenseCamera &camera, InitialMapData &initial_data, int num_frames = 3);
    
    std::vector<MapPoint> backProjectKeypoints(
        const std::vector<cv::KeyPoint> &keypoints,
        const cv::Mat &depth,
        const rs2_intrinsics &intrinsics);
    
private:
    // Helper methods for multi-frame initialization
    bool validateFrameQuality(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &depth);
    bool validateMatchQuality(const std::vector<cv::DMatch> &matches, int min_matches = 50);
};
