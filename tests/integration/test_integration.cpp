#include "test_framework.hpp"
#include "../../include/initializer.hpp"
#include "../../include/orb_detector.hpp"
#include "mocks/mock_camera.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

// Test the complete pipeline with mock camera
void testCompletePipeline() {
    MockRealSenseCamera mock_camera;
    
    // Capture frame
    cv::Mat color, depth;
    bool frame_captured = mock_camera.getFrame(color, depth);
    ASSERT_TRUE(frame_captured);
    ASSERT_FALSE(color.empty());
    ASSERT_FALSE(depth.empty());
    
    // Detect ORB features
    ORBDetector orb_detector(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_detector.detectAndCompute(color, keypoints, descriptors);
    
    ASSERT_GREATER(keypoints.size(), 0);
    ASSERT_EQUAL(keypoints.size(), static_cast<size_t>(descriptors.rows));
    
    // Extract 3D coordinates
    rs2_intrinsics intrinsics = mock_camera.getIntrinsics();
    Initializer initializer;
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    ASSERT_GREATER(map_points.size(), 0);
    
    // Verify 3D points are reasonable
    int valid_points = 0;
    for (const auto &mp : map_points) {
        if (mp.position.z > 0.1f && mp.position.z < 10.0f) {
            valid_points++;
        }
    }
    ASSERT_GREATER(valid_points, 0);
}

// Test camera intrinsics
void testCameraIntrinsics() {
    MockRealSenseCamera mock_camera;
    rs2_intrinsics intrinsics = mock_camera.getIntrinsics();
    
    ASSERT_GREATER(intrinsics.fx, 0);
    ASSERT_GREATER(intrinsics.fy, 0);
    ASSERT_GREATER(intrinsics.ppx, 0);
    ASSERT_GREATER(intrinsics.ppy, 0);
    ASSERT_EQUAL(intrinsics.width, 640);
    ASSERT_EQUAL(intrinsics.height, 480);
}

// Test data consistency across pipeline
void testDataConsistency() {
    MockRealSenseCamera mock_camera;
    
    cv::Mat color, depth;
    mock_camera.getFrame(color, depth);
    
    // Check image dimensions match intrinsics
    rs2_intrinsics intrinsics = mock_camera.getIntrinsics();
    ASSERT_EQUAL(color.cols, intrinsics.width);
    ASSERT_EQUAL(color.rows, intrinsics.height);
    ASSERT_EQUAL(depth.cols, intrinsics.width);
    ASSERT_EQUAL(depth.rows, intrinsics.height);
    
    // Check depth values are reasonable
    cv::Scalar depth_mean = cv::mean(depth);
    ASSERT_GREATER(depth_mean[0], 0);
    ASSERT_LESS(depth_mean[0], 10000); // Less than 10 meters average
}

// Test error handling
void testErrorHandling() {
    // Test with empty images
    cv::Mat empty_color, empty_depth;
    ORBDetector orb_detector(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    orb_detector.detectAndCompute(empty_color, keypoints, descriptors);
    ASSERT_EQUAL(keypoints.size(), 0);
}

// Test performance characteristics
void testPerformanceCharacteristics() {
    MockRealSenseCamera mock_camera;
    ORBDetector orb_detector(1000);
    Initializer initializer;
    
    cv::Mat color, depth;
    mock_camera.getFrame(color, depth);
    
    // Measure feature detection time
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_detector.detectAndCompute(color, keypoints, descriptors);
    
    rs2_intrinsics intrinsics = mock_camera.getIntrinsics();
    std::vector<MapPoint> map_points = initializer.backProjectKeypoints(keypoints, depth, intrinsics);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time (< 1 second)
    ASSERT_LESS(duration.count(), 1000);
    
    ASSERT_GREATER(keypoints.size(), 0);
    ASSERT_GREATER(map_points.size(), 0);
}

int runIntegrationTests() {
    std::cout << "=== Integration Tests ===" << std::endl;
    
    testCompletePipeline();
    testCameraIntrinsics();
    testDataConsistency();
    testErrorHandling();
    testPerformanceCharacteristics();
    
    TestFramework::printSummary();
    return TestFramework::failedTests > 0 ? 1 : 0;
}

int main() {
    return runIntegrationTests();
}
