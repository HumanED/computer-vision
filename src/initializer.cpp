#include "initializer.hpp"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <thread>
#include <chrono>

bool Initializer::initialize(RealSenseCamera &camera) {
    InitialMapData dummy_data;
    return initialize(camera, dummy_data);
}

bool Initializer::initialize(RealSenseCamera &camera, InitialMapData &initial_data) {
    std::cout << "[Initializer] Starting map initialization..." << std::endl;
    
    // Capture a single RGB-D frame
    cv::Mat color, depth;
    if (!camera.getFrame(color, depth)) {
        std::cerr << "[Initializer] Failed to capture frame" << std::endl;
        return false;
    }
    
    // Store frame data
    initial_data.color = color.clone();
    initial_data.depth = depth.clone();
    
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
    
    // Store keypoints and descriptors
    initial_data.keypoints = keypoints;
    initial_data.descriptors = descriptors.clone();
    
    // Extract 3D coordinates using depth map
    rs2_intrinsics intrinsics = camera.getIntrinsics();
    std::vector<MapPoint> map_points = backProjectKeypoints(keypoints, depth, intrinsics);
    
    // Store map points
    initial_data.map_points = map_points;
    
    std::cout << "[Initializer] Created " << map_points.size() 
              << " 3D map points" << std::endl;
    
    // Set the first camera pose (identity matrix for the initial frame)
    initial_data.initial_pose = Eigen::Matrix4d::Identity();
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

// Multi-frame initialization for more reliable tracking
bool Initializer::initializeMultiFrame(RealSenseCamera &camera, InitialMapData &initial_data, int num_frames) {
    std::cout << "[Initializer] Starting multi-frame initialization (" << num_frames << " frames)..." << std::endl;
    
    if (num_frames < 2) {
        std::cerr << "[Initializer] Warning: num_frames < 2, falling back to single-frame initialization" << std::endl;
        return initialize(camera, initial_data);
    }
    
    // Clamp num_frames to reasonable range
    if (num_frames > 5) {
        std::cout << "[Initializer] Clamping num_frames from " << num_frames << " to 5" << std::endl;
        num_frames = 5;
    }
    
    // Setup ORB detector and matcher
    ORBDetector orb_detector(1000);
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    rs2_intrinsics intrinsics = camera.getIntrinsics();
    
    // Storage for all frames
    struct FrameData {
        cv::Mat color;
        cv::Mat depth;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };
    
    std::vector<FrameData> frames;
    frames.reserve(num_frames);
    
    // Step 1: Capture multiple frames
    std::cout << "[Initializer] Capturing " << num_frames << " frames..." << std::endl;
    for (int i = 0; i < num_frames; ++i) {
        FrameData frame;
        
        if (!camera.getFrame(frame.color, frame.depth)) {
            std::cerr << "[Initializer] Failed to capture frame " << i << std::endl;
            return false;
        }
        
        // Detect features
        orb_detector.detectAndCompute(frame.color, frame.keypoints, frame.descriptors);
        
        std::cout << "  Frame " << i << ": " << frame.keypoints.size() << " keypoints" << std::endl;
        
        // Validate frame quality
        if (!validateFrameQuality(frame.keypoints, frame.depth)) {
            std::cerr << "[Initializer] Frame " << i << " quality check failed, retrying..." << std::endl;
            --i;  // Retry this frame
            continue;
        }
        
        frames.push_back(frame);
        
        // Small delay between frames to allow camera motion
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // Step 2: Match features across consecutive frames to validate motion
    std::cout << "[Initializer] Validating feature tracking across frames..." << std::endl;
    int total_good_matches = 0;
    
    for (size_t i = 0; i < frames.size() - 1; ++i) {
        std::vector<cv::DMatch> matches;
        matcher.match(frames[i].descriptors, frames[i+1].descriptors, matches);
        
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
        
        std::cout << "  Frame " << i << " -> Frame " << (i+1) << ": " 
                  << good_matches.size() << " good matches" << std::endl;
        
        if (!validateMatchQuality(good_matches)) {
            std::cerr << "[Initializer] Insufficient matches between frames " << i 
                      << " and " << (i+1) << std::endl;
            return false;
        }
        
        total_good_matches += good_matches.size();
    }
    
    std::cout << "[Initializer] Average matches per frame pair: " 
              << (total_good_matches / (frames.size() - 1)) << std::endl;
    
    // Step 3: Use the middle frame as reference (most stable)
    int reference_idx = frames.size() / 2;
    std::cout << "[Initializer] Using frame " << reference_idx << " as reference" << std::endl;
    
    FrameData& ref_frame = frames[reference_idx];
    
    // Step 4: Aggregate map points from all frames
    std::cout << "[Initializer] Building initial map from all frames..." << std::endl;
    std::vector<MapPoint> all_map_points;
    
    for (size_t i = 0; i < frames.size(); ++i) {
        std::vector<MapPoint> frame_points = backProjectKeypoints(
            frames[i].keypoints, 
            frames[i].depth, 
            intrinsics
        );
        
        all_map_points.insert(all_map_points.end(), frame_points.begin(), frame_points.end());
        std::cout << "  Frame " << i << ": " << frame_points.size() << " 3D points" << std::endl;
    }
    
    // Step 5: Populate initial_data with reference frame and all map points
    initial_data.color = ref_frame.color.clone();
    initial_data.depth = ref_frame.depth.clone();
    initial_data.keypoints = ref_frame.keypoints;
    initial_data.descriptors = ref_frame.descriptors.clone();
    initial_data.map_points = all_map_points;
    initial_data.initial_pose = Eigen::Matrix4d::Identity();
    initial_data.num_frames_used = num_frames;
    
    // Save debug images from reference frame
    cv::imwrite("debug_captured_color.png", ref_frame.color);
    cv::imwrite("debug_captured_depth.png", ref_frame.depth);
    
    cv::Mat gray;
    cv::cvtColor(ref_frame.color, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("debug_captured_gray.png", gray);
    
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    cv::imwrite("debug_captured_enhanced.png", enhanced);
    
    // Step 6: Print statistics
    int valid_3d_points = 0;
    for (const auto &mp : all_map_points) {
        if (mp.position.z > 0.1f && mp.position.z < 10.0f) {
            valid_3d_points++;
        }
    }
    
    std::cout << "[Initializer] Multi-frame initialization completed successfully!" << std::endl;
    std::cout << "  - Frames used: " << num_frames << std::endl;
    std::cout << "  - Reference frame keypoints: " << ref_frame.keypoints.size() << std::endl;
    std::cout << "  - Total 3D map points: " << all_map_points.size() << std::endl;
    std::cout << "  - Valid 3D points: " << valid_3d_points << std::endl;
    
    return true;
}

// Validate frame quality based on keypoint count and depth coverage
bool Initializer::validateFrameQuality(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &depth) {
    const int MIN_KEYPOINTS = 100;
    
    if (keypoints.size() < MIN_KEYPOINTS) {
        std::cerr << "  [Quality Check] Too few keypoints: " << keypoints.size() 
                  << " < " << MIN_KEYPOINTS << std::endl;
        return false;
    }
    
    // Check depth validity
    int valid_depth_count = 0;
    for (const auto &kp : keypoints) {
        int u = static_cast<int>(kp.pt.x);
        int v = static_cast<int>(kp.pt.y);
        
        if (u >= 0 && u < depth.cols && v >= 0 && v < depth.rows) {
            float depth_value = depth.at<uint16_t>(v, u) / 1000.0f;
            if (depth_value > 0.1f && depth_value < 10.0f) {
                valid_depth_count++;
            }
        }
    }
    
    float valid_ratio = static_cast<float>(valid_depth_count) / keypoints.size();
    const float MIN_VALID_RATIO = 0.3f;
    
    if (valid_ratio < MIN_VALID_RATIO) {
        std::cerr << "  [Quality Check] Too few valid depth values: " 
                  << valid_ratio * 100.0f << "% < " << MIN_VALID_RATIO * 100.0f << "%" << std::endl;
        return false;
    }
    
    return true;
}

// Validate match quality between frames
bool Initializer::validateMatchQuality(const std::vector<cv::DMatch> &matches, int min_matches) {
    if (matches.size() < static_cast<size_t>(min_matches)) {
        return false;
    }
    return true;
}
