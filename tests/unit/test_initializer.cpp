#include "test_framework.hpp"
#include "../../include/initializer.hpp"
#include "mocks/mock_camera.hpp"
#include <opencv2/opencv.hpp>

// Test the back-projection function directly
void testBackProjectionBasic() {
    Initializer initializer;
    
    // Create test keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoints.push_back(cv::KeyPoint(320, 240, 10)); // Center of image
    keypoints.push_back(cv::KeyPoint(100, 100, 10)); // Corner region
    
    // Create test depth map
    cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
    depth.at<uint16_t>(240, 320) = 1000; // 1 meter at center
    depth.at<uint16_t>(100, 100) = 2000; // 2 meters at corner
    
    // Create test intrinsics
    rs2_intrinsics intrinsics;
    intrinsics.fx = 608.345f;
    intrinsics.fy = 607.212f;
    intrinsics.ppx = 322.442f;
    intrinsics.ppy = 248.247f;
    
    // Test back-projection
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    ASSERT_EQUAL(map_points.size(), 2);
    
    // Check first point (center)
    cv::Point3f center_point = map_points[0].position;
    ASSERT_LESS(abs(center_point.x), 0.1f); // Should be near 0 (center)
    ASSERT_LESS(abs(center_point.y), 0.1f); // Should be near 0 (center)
    ASSERT_EQUAL(center_point.z, 1.0f); // Should be 1 meter
    
    // Check second point (corner)
    cv::Point3f corner_point = map_points[1].position;
    ASSERT_LESS(corner_point.x, 0); // Should be negative (left of center)
    ASSERT_LESS(corner_point.y, 0); // Should be negative (above center)
    ASSERT_EQUAL(corner_point.z, 2.0f); // Should be 2 meters
}

void testBackProjectionInvalidDepth() {
    Initializer initializer;
    
    // Create test keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoints.push_back(cv::KeyPoint(320, 240, 10)); // Valid depth
    keypoints.push_back(cv::KeyPoint(100, 100, 10)); // Invalid depth (0)
    keypoints.push_back(cv::KeyPoint(200, 200, 10)); // Invalid depth (too far)
    
    // Create test depth map
    cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
    depth.at<uint16_t>(240, 320) = 1000;  // Valid: 1 meter
    depth.at<uint16_t>(100, 100) = 0;      // Invalid: 0 depth
    depth.at<uint16_t>(200, 200) = 15000; // Invalid: 15 meters (too far)
    
    rs2_intrinsics intrinsics;
    intrinsics.fx = 608.345f;
    intrinsics.fy = 607.212f;
    intrinsics.ppx = 322.442f;
    intrinsics.ppy = 248.247f;
    
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    // Should only return the valid point
    ASSERT_EQUAL(map_points.size(), 1);
    ASSERT_EQUAL(map_points[0].position.z, 1.0f);
}

void testBackProjectionBoundsChecking() {
    Initializer initializer;
    
    // Create keypoints outside image bounds
    std::vector<cv::KeyPoint> keypoints;
    keypoints.push_back(cv::KeyPoint(-10, 240, 10));  // Outside left
    keypoints.push_back(cv::KeyPoint(320, -10, 10));  // Outside top
    keypoints.push_back(cv::KeyPoint(700, 240, 10)); // Outside right
    keypoints.push_back(cv::KeyPoint(320, 500, 10)); // Outside bottom
    keypoints.push_back(cv::KeyPoint(320, 240, 10)); // Valid point
    
    cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
    depth.at<uint16_t>(240, 320) = 1000;
    
    rs2_intrinsics intrinsics;
    intrinsics.fx = 608.345f;
    intrinsics.fy = 607.212f;
    intrinsics.ppx = 322.442f;
    intrinsics.ppy = 248.247f;
    
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    // Should only return the valid point
    ASSERT_EQUAL(map_points.size(), 1);
}

// Test with mock camera data
void testInitializerWithMockData() {
    MockRealSenseCamera mock_camera;
    Initializer initializer;
    
    // Get mock data
    cv::Mat color, depth;
    mock_camera.getFrame(color, depth);
    rs2_intrinsics intrinsics = mock_camera.getIntrinsics();
    
    // Test ORB detection
    ORBDetector orb_detector(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_detector.detectAndCompute(color, keypoints, descriptors);
    
    ASSERT_GREATER(keypoints.size(), 0);
    
    // Test 3D reconstruction
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    ASSERT_GREATER(map_points.size(), 0);
}

int runInitializerTests() {
    std::cout << "=== Initializer Unit Tests ===" << std::endl;
    
    testBackProjectionBasic();
    testBackProjectionInvalidDepth();
    testBackProjectionBoundsChecking();
    testInitializerWithMockData();
    
    TestFramework::printSummary();
    return TestFramework::failedTests > 0 ? 1 : 0;
}

int main() {
    return runInitializerTests();
}
