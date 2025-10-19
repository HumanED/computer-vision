
#include "realsense_camera.hpp"
#include <iostream>

RealSenseCamera::RealSenseCamera() {
    // Configure RealSense pipeline
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    
    // Start the pipeline
    pipe.start(cfg);
    
    // Get intrinsics from the color stream
    auto stream = cfg.resolve(pipe);
    auto color_stream = stream.get_stream(RS2_STREAM_COLOR);
    intrinsics = color_stream.as<rs2::video_stream_profile>().get_intrinsics();
    
    std::cout << "[RealSense] Camera initialized successfully" << std::endl;
    std::cout << "[RealSense] Color intrinsics - fx: " << intrinsics.fx 
              << ", fy: " << intrinsics.fy 
              << ", cx: " << intrinsics.ppx 
              << ", cy: " << intrinsics.ppy << std::endl;
    
    // Wait a bit for camera to stabilize
    std::cout << "[RealSense] Waiting for camera to stabilize..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        rs2::frameset frames = pipe.wait_for_frames();
        std::cout << "[RealSense] Warming up frame " << i + 1 << "/10" << std::endl;
    }
    std::cout << "[RealSense] Camera stabilization complete" << std::endl;
}

RealSenseCamera::~RealSenseCamera() {
    pipe.stop();
    std::cout << "[RealSense] Camera stopped" << std::endl;
}

bool RealSenseCamera::getFrame(cv::Mat &color, cv::Mat &depth) {
    try {
        rs2::frameset frames = pipe.wait_for_frames();
        
        // Get color frame
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
            std::cerr << "[RealSense] Failed to get color frame" << std::endl;
            return false;
        }
        
        // Get depth frame
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        if (!depth_frame) {
            std::cerr << "[RealSense] Failed to get depth frame" << std::endl;
            return false;
        }
        
        // Convert to OpenCV Mat
        color = cv::Mat(cv::Size(color_frame.get_width(), color_frame.get_height()),
                       CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).clone();
        
        depth = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                       CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP).clone();
        
        // Debug: Check frame statistics
        cv::Scalar color_mean = cv::mean(color);
        double depth_mean = cv::mean(depth)[0];
        std::cout << "[RealSense] Frame stats - Color mean: " << color_mean[0] << ", " 
                  << color_mean[1] << ", " << color_mean[2] 
                  << " | Depth mean: " << depth_mean << std::endl;
        
        return true;
    } catch (const rs2::error &e) {
        std::cerr << "[RealSense ERROR] " << e.what() << std::endl;
        return false;
    }
}

rs2_intrinsics RealSenseCamera::getIntrinsics() const {
    return intrinsics;
}
