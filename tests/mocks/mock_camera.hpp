#pragma once
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

// Mock RealSense intrinsics for testing
struct MockIntrinsics {
    float fx = 608.345f;
    float fy = 607.212f;
    float ppx = 322.442f;
    float ppy = 248.247f;
    int width = 640;
    int height = 480;
};

// Mock camera class for testing without hardware
class MockRealSenseCamera {
public:
    MockRealSenseCamera() {
        // Generate synthetic test data
        generateTestImages();
    }
    
    bool getFrame(cv::Mat &color, cv::Mat &depth) {
        color = test_color.clone();
        depth = test_depth.clone();
        return true;
    }
    
    rs2_intrinsics getIntrinsics() const {
        rs2_intrinsics intrinsics;
        intrinsics.fx = mock_intrinsics.fx;
        intrinsics.fy = mock_intrinsics.fy;
        intrinsics.ppx = mock_intrinsics.ppx;
        intrinsics.ppy = mock_intrinsics.ppy;
        intrinsics.width = mock_intrinsics.width;
        intrinsics.height = mock_intrinsics.height;
        return intrinsics;
    }

private:
    cv::Mat test_color;
    cv::Mat test_depth;
    MockIntrinsics mock_intrinsics;
    
    void generateTestImages() {
        // Generate a synthetic color image with patterns for feature detection
        test_color = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Add some geometric patterns that will generate ORB features
        cv::rectangle(test_color, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
        cv::rectangle(test_color, cv::Point(300, 100), cv::Point(500, 300), cv::Scalar(128, 128, 128), -1);
        cv::circle(test_color, cv::Point(100, 350), 50, cv::Scalar(200, 200, 200), -1);
        cv::circle(test_color, cv::Point(400, 400), 30, cv::Scalar(100, 100, 100), -1);
        
        // Add some noise for more realistic features
        cv::Mat noise = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(25));
        test_color += noise;
        
        // Generate synthetic depth map
        test_depth = cv::Mat::zeros(480, 640, CV_16UC1);
        
        // Add depth values (in mm)
        cv::rectangle(test_depth, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(1000), -1);  // 1m
        cv::rectangle(test_depth, cv::Point(300, 100), cv::Point(500, 300), cv::Scalar(2000), -1);  // 2m
        cv::circle(test_depth, cv::Point(100, 350), 50, cv::Scalar(1500), -1);  // 1.5m
        cv::circle(test_depth, cv::Point(400, 400), 30, cv::Scalar(3000), -1);  // 3m
        
        // Add some noise to depth
        cv::Mat depth_noise = cv::Mat::zeros(480, 640, CV_16UC1);
        cv::randn(depth_noise, cv::Scalar::all(0), cv::Scalar::all(50));
        test_depth += depth_noise;
    }
};
