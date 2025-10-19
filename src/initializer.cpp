#include "initializer.hpp"
#include <iostream>
#include <opencv2/calib3d.hpp>

bool Initializer::initialize(RealSenseCamera &camera) {
    std::cout << "[Initializer] Starting map initialization..." << std::endl;
    
    // Capture a single RGB-D frame
    cv::Mat color, depth;
    if (!camera.getFrame(color, depth)) {
        std::cerr << "[Initializer] Failed to capture frame" << std::endl;
        return false;
    }
    
    std::cout << "[Initializer] Captured RGB-D frame: " 
              << color.cols << "x" << color.rows << std::endl;
    
    // Save the captured image for debugging
    cv::imwrite("debug_captured_color.png", color);
    cv::imwrite("debug_captured_depth.png", depth);
    
    // Also save grayscale version for ORB analysis
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("debug_captured_gray.png", gray);
    
    // Save enhanced version
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    cv::imwrite("debug_captured_enhanced.png", enhanced);
    
    std::cout << "[Initializer] Saved debug images: debug_captured_color.png, debug_captured_depth.png, debug_captured_gray.png, debug_captured_enhanced.png" << std::endl;
    
    // Detect ORB features
    ORBDetector orb_detector(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_detector.detectAndCompute(color, keypoints, descriptors);
    
    if (keypoints.empty()) {
        std::cerr << "[Initializer] No keypoints detected" << std::endl;
        return false;
    }
    
    // Extract 3D coordinates using depth map
    rs2_intrinsics intrinsics = camera.getIntrinsics();
    std::vector<MapPoint> map_points = backProjectKeypoints(keypoints, depth, intrinsics);
    
    std::cout << "[Initializer] Created " << map_points.size() 
              << " 3D map points" << std::endl;
    
    // Set the first camera pose (identity matrix for the initial frame)
    cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    std::cout << "[Initializer] Set initial camera pose (identity matrix)" << std::endl;
    
    // Print some statistics
    int valid_3d_points = 0;
    for (const auto &mp : map_points) {
        if (mp.position.z > 0.1f && mp.position.z < 10.0f) { // Valid depth range
            valid_3d_points++;
        }
    }
    
    std::cout << "[Initializer] Valid 3D points: " << valid_3d_points 
              << " out of " << map_points.size() << std::endl;
    
    std::cout << "[Initializer] Map initialization completed successfully!" << std::endl;
    return true;
}

std::vector<MapPoint> Initializer::backProjectKeypoints(
    const std::vector<cv::KeyPoint> &keypoints,
    const cv::Mat &depth,
    const rs2_intrinsics &intrinsics) {
    
    std::vector<MapPoint> map_points;
    
    for (const auto &kp : keypoints) {
        int u = static_cast<int>(kp.pt.x);
        int v = static_cast<int>(kp.pt.y);
        
        // Check bounds
        if (u < 0 || u >= depth.cols || v < 0 || v >= depth.rows) {
            continue;
        }
        
        // Get depth value
        float depth_value = depth.at<uint16_t>(v, u) / 1000.0f; // Convert mm to meters
        
        // Skip invalid depth values
        if (depth_value <= 0.0f || depth_value > 10.0f) {
            continue;
        }
        
        // Back-project to 3D coordinates
        float x = (u - intrinsics.ppx) * depth_value / intrinsics.fx;
        float y = (v - intrinsics.ppy) * depth_value / intrinsics.fy;
        float z = depth_value;
        
        MapPoint mp;
        mp.position = cv::Point3f(x, y, z);
        mp.keypoint = kp;
        
        map_points.push_back(mp);
    }
    
    return map_points;
}
