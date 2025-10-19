#include "test_framework.hpp"
#include "../../include/realsense_camera.hpp"
#include "../../include/initializer.hpp"
#include "../../include/orb_detector.hpp"
#include <opencv2/opencv.hpp>

// Hardware integration test - requires actual RealSense camera
void testRealCameraConnection() {
    std::cout << "[Hardware Test] Attempting to connect to RealSense camera..." << std::endl;
    
    try {
        RealSenseCamera camera;
        
        // Try to capture a frame
        cv::Mat color, depth;
        bool frame_captured = camera.getFrame(color, depth);
        
        if (frame_captured) {
            ASSERT_FALSE(color.empty());
            ASSERT_FALSE(depth.empty());
            ASSERT_EQUAL(color.cols, 640);
            ASSERT_EQUAL(color.rows, 480);
            ASSERT_EQUAL(depth.cols, 640);
            ASSERT_EQUAL(depth.rows, 480);
            
            std::cout << "[Hardware Test] [PASS] Camera connection successful!" << std::endl;
            std::cout << "[Hardware Test] Captured frame: " << color.cols << "x" << color.rows << std::endl;
        } else {
            std::cout << "[Hardware Test] [FAIL] Failed to capture frame" << std::endl;
            ASSERT_TRUE(false); // This will fail the test
        }
        
    } catch (const std::exception& e) {
        std::cout << "[Hardware Test] [FAIL] Camera connection failed: " << e.what() << std::endl;
        ASSERT_TRUE(false); // This will fail the test
    }
}

void testRealCameraPipeline() {
    std::cout << "[Hardware Test] Testing complete pipeline with real camera..." << std::endl;
    
    try {
        RealSenseCamera camera;
        Initializer initializer;
        
        // Test the complete initialization pipeline
        bool result = initializer.initialize(camera);
        
        if (result) {
            std::cout << "[Hardware Test] [PASS] Complete pipeline test successful!" << std::endl;
            ASSERT_TRUE(true);
        } else {
            std::cout << "[Hardware Test] [FAIL] Pipeline test failed" << std::endl;
            ASSERT_TRUE(false);
        }
        
    } catch (const std::exception& e) {
        std::cout << "[Hardware Test] [FAIL] Pipeline test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
}

void testCameraIntrinsicsReal() {
    std::cout << "[Hardware Test] Testing camera intrinsics..." << std::endl;
    
    try {
        RealSenseCamera camera;
        rs2_intrinsics intrinsics = camera.getIntrinsics();
        
        // Check that intrinsics are reasonable
        ASSERT_GREATER(intrinsics.fx, 0);
        ASSERT_GREATER(intrinsics.fy, 0);
        ASSERT_GREATER(intrinsics.ppx, 0);
        ASSERT_GREATER(intrinsics.ppy, 0);
        ASSERT_EQUAL(intrinsics.width, 640);
        ASSERT_EQUAL(intrinsics.height, 480);
        
        std::cout << "[Hardware Test] [PASS] Intrinsics test successful!" << std::endl;
        std::cout << "[Hardware Test] fx: " << intrinsics.fx << ", fy: " << intrinsics.fy << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[Hardware Test] [FAIL] Intrinsics test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
}

int main() {
    std::cout << "=== HARDWARE INTEGRATION TESTS ===" << std::endl;
    std::cout << "WARNING: These tests require a RealSense D435i camera to be connected!" << std::endl;
    std::cout << "If no camera is connected, these tests will FAIL (which is expected)." << std::endl;
    std::cout << "" << std::endl;
    
    testRealCameraConnection();
    testCameraIntrinsicsReal();
    testRealCameraPipeline();
    
    TestFramework::printSummary();
    return TestFramework::failedTests > 0 ? 1 : 0;
}
