#include "realsense_camera.hpp"
#include "orb_detector.hpp"
#include "initializer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

struct Frame {
    cv::Mat color;
    cv::Mat depth;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

// Convert pixel + depth to 3D point using camera intrinsics
cv::Point3f deproject(const cv::Point2f& pixel, float depth_value, const rs2_intrinsics& intrinsics) {
    float z = depth_value;
    float x = (pixel.x - intrinsics.ppx) * z / intrinsics.fx;
    float y = (pixel.y - intrinsics.ppy) * z / intrinsics.fy;
    return cv::Point3f(x, y, z);
}

// Estimate pose using PnP RANSAC
bool estimatePose(const std::vector<cv::Point3f>& points3d_prev,
                  const std::vector<cv::Point2f>& points2d_curr,
                  const rs2_intrinsics& intrinsics,
                  cv::Mat& rvec, cv::Mat& tvec) {
    
    if (points3d_prev.size() < 10) {
        return false;
    }

    // Build camera intrinsic matrix from RealSense intrinsics
    cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx,
                                            0, intrinsics.fy, intrinsics.ppy,
                                            0, 0, 1);
    
    // Use PnP RANSAC to estimate pose
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(
        points3d_prev,
        points2d_curr,
        K,
        cv::Mat(),  // no distortion
        rvec,
        tvec,
        false,
        100,        // iterations
        8.0,        // reprojection error threshold
        0.99,       // confidence
        inliers
    );

    if (success && inliers.size() >= 10) {
        std::cout << "  [PnP] Inliers: " << inliers.size() 
                  << "/" << points3d_prev.size() << std::endl;
        return true;
    }
    
    return false;
}

// Convert rotation vector and translation to transformation matrix
Eigen::Matrix4d toMatrix4d(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 0) = R.at<double>(0, 0);
    T(0, 1) = R.at<double>(0, 1);
    T(0, 2) = R.at<double>(0, 2);
    T(1, 0) = R.at<double>(1, 0);
    T(1, 1) = R.at<double>(1, 1);
    T(1, 2) = R.at<double>(1, 2);
    T(2, 0) = R.at<double>(2, 0);
    T(2, 1) = R.at<double>(2, 1);
    T(2, 2) = R.at<double>(2, 2);
    T(0, 3) = tvec.at<double>(0);
    T(1, 3) = tvec.at<double>(1);
    T(2, 3) = tvec.at<double>(2);
    
    return T;
}

void printPose(const Eigen::Matrix4d& pose, int frame_num) {
    Eigen::Vector3d translation = pose.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    
    // Convert rotation matrix to Euler angles (roll, pitch, yaw)
    Eigen::Vector3d euler = rotation.eulerAngles(0, 1, 2);
    
    std::cout << "\n=== Frame " << frame_num << " Camera Pose ===" << std::endl;
    std::cout << "Translation (x, y, z): " 
              << translation.x() << ", " 
              << translation.y() << ", " 
              << translation.z() << " meters" << std::endl;
    std::cout << "Rotation (roll, pitch, yaw): "
              << euler.x() * 180.0 / M_PI << ", "
              << euler.y() * 180.0 / M_PI << ", "
              << euler.z() * 180.0 / M_PI << " degrees" << std::endl;
    std::cout << "Transformation Matrix:\n" << pose << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== Integrated Visual Odometry System ===" << std::endl;
        
        // Initialize camera
        std::cout << "\n[Step 1] Initializing camera..." << std::endl;
        RealSenseCamera camera;
        rs2_intrinsics intrinsics = camera.getIntrinsics();

        // Initialize map using multi-frame approach (2-3 frames) for better reliability
        std::cout << "\n[Step 2] Initializing map with multi-frame approach..." << std::endl;
        Initializer initializer;
        InitialMapData initial_data;
        
        // Use 3 frames for initialization (configurable)
        const int NUM_INIT_FRAMES = 3;
        
        if (!initializer.initializeMultiFrame(camera, initial_data, NUM_INIT_FRAMES)) {
            std::cerr << "[ERROR] Multi-frame map initialization failed!" << std::endl;
            std::cerr << "         Falling back to single-frame initialization..." << std::endl;
            
            // Fallback to single-frame initialization
            if (!initializer.initialize(camera, initial_data)) {
                std::cerr << "[ERROR] Single-frame initialization also failed!" << std::endl;
                return EXIT_FAILURE;
            }
        }
        
        std::cout << "[Step 2] Map initialization successful!" << std::endl;
        std::cout << "  - Frames used: " << initial_data.num_frames_used << std::endl;
        std::cout << "  - Initial keypoints: " << initial_data.keypoints.size() << std::endl;
        std::cout << "  - Initial 3D map points: " << initial_data.map_points.size() << std::endl;

        // Create ORB feature detector for subsequent frames
        ORBDetector orb_detector(1000);
        
        // BFMatcher for feature matching
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        
        // Initialize tracking with the first frame from initializer
        Frame prev_frame, curr_frame;
        prev_frame.color = initial_data.color;
        prev_frame.depth = initial_data.depth;
        prev_frame.keypoints = initial_data.keypoints;
        prev_frame.descriptors = initial_data.descriptors;
        
        Eigen::Matrix4d cumulative_pose = initial_data.initial_pose;
        
        int frame_count = 1;  // Start from 1 since frame 0 was used for initialization

        std::cout << "\n[Step 3] Starting visual odometry tracking...\n" << std::endl;
        
        // Print initial pose
        printPose(cumulative_pose, 0);

        // Main tracking loop
        while (true) {
            // 1. Grab synchronized RGB and Depth frames
            if (!camera.getFrame(curr_frame.color, curr_frame.depth)) {
                std::cerr << "Failed to get frame, skipping..." << std::endl;
                continue;
            }
            
            // 2. Extract ORB features from RGB frame
            orb_detector.detectAndCompute(curr_frame.color, curr_frame.keypoints, curr_frame.descriptors);
            
            std::cout << "Frame " << frame_count << ": Detected " 
                      << curr_frame.keypoints.size() << " features" << std::endl;
            
            // 3. Track features from previous frame to current frame
            if (prev_frame.descriptors.empty() || curr_frame.descriptors.empty()) {
                std::cout << "  [Warning] No descriptors in one of the frames, skipping..." << std::endl;
                prev_frame = curr_frame;
                frame_count++;
                continue;
            }
            
            std::vector<cv::DMatch> matches;
            matcher.match(prev_frame.descriptors, curr_frame.descriptors, matches);
            
            // Filter matches by distance
            double min_dist = 100.0;
            for (const auto& match : matches) {
                if (match.distance < min_dist) {
                    min_dist = match.distance;
                }
            }
            
            std::vector<cv::DMatch> good_matches;
            for (const auto& match : matches) {
                if (match.distance <= std::max(2.0 * min_dist, 30.0)) {
                    good_matches.push_back(match);
                }
            }
            
            std::cout << "  [Matching] Good matches: " << good_matches.size() 
                      << "/" << matches.size() << std::endl;
            
            if (good_matches.size() < 10) {
                std::cout << "  [Warning] Not enough good matches, skipping pose estimation..." << std::endl;
                prev_frame = curr_frame;
                frame_count++;
                continue;
            }
            
            // 4. Estimate camera's new pose
            // Build 3D-2D correspondences
            std::vector<cv::Point3f> points3d_prev;
            std::vector<cv::Point2f> points2d_curr;
            
            for (const auto& match : good_matches) {
                cv::Point2f pt_prev = prev_frame.keypoints[match.queryIdx].pt;
                cv::Point2f pt_curr = curr_frame.keypoints[match.trainIdx].pt;
                
                // Get depth at previous frame keypoint
                int x = static_cast<int>(pt_prev.x);
                int y = static_cast<int>(pt_prev.y);
                
                if (x >= 0 && x < prev_frame.depth.cols && 
                    y >= 0 && y < prev_frame.depth.rows) {
                    
                    float depth_val = prev_frame.depth.at<uint16_t>(y, x) / 1000.0f; // Convert mm to meters
                    
                    if (depth_val > 0.1f && depth_val < 10.0f) {  // valid depth (0.1m - 10m)
                        cv::Point3f point3d = deproject(pt_prev, depth_val, intrinsics);
                        points3d_prev.push_back(point3d);
                        points2d_curr.push_back(pt_curr);
                    }
                }
            }
            
            std::cout << "  [3D-2D] Valid correspondences: " << points3d_prev.size() << std::endl;
            
            // Estimate pose using PnP
            cv::Mat rvec, tvec;
            if (estimatePose(points3d_prev, points2d_curr, intrinsics, rvec, tvec)) {
                // Convert to transformation matrix
                Eigen::Matrix4d relative_pose = toMatrix4d(rvec, tvec);
                
                // Update cumulative pose (T_new = T_old * T_relative^-1)
                cumulative_pose = cumulative_pose * relative_pose.inverse();
                
                // 5. Print estimated camera pose
                printPose(cumulative_pose, frame_count);
            } else {
                std::cout << "  [Warning] Pose estimation failed, keeping previous pose..." << std::endl;
            }
            
            // Update for next iteration
            prev_frame = curr_frame;
            frame_count++;
            
            // Optional: limit frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Optional: break after certain number of frames for testing
            // if (frame_count >= 50) break;
        }
        
        std::cout << "\nTracking finished." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
