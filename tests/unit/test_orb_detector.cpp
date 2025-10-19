#include "test_framework.hpp"
#include "../../include/orb_detector.hpp"
#include <opencv2/opencv.hpp>

// Test ORB detector with synthetic images
void testORBDetectorBasicFunctionality() {
    ORBDetector detector(1000);
    
    // Create a test image with geometric patterns
    cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(test_image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(test_image, cv::Point(300, 100), cv::Point(500, 300), cv::Scalar(128, 128, 128), -1);
    cv::circle(test_image, cv::Point(100, 350), 50, cv::Scalar(200, 200, 200), -1);
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    detector.detectAndCompute(test_image, keypoints, descriptors);
    
    ASSERT_GREATER(keypoints.size(), 0);
    ASSERT_EQUAL(keypoints.size(), static_cast<size_t>(descriptors.rows));
    ASSERT_EQUAL(descriptors.cols, 32); // ORB descriptor size
}

void testORBDetectorEmptyImage() {
    ORBDetector detector(1000);
    
    cv::Mat empty_image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    detector.detectAndCompute(empty_image, keypoints, descriptors);
    
    ASSERT_EQUAL(keypoints.size(), 0);
    ASSERT_TRUE(descriptors.empty());
}

void testORBDetectorGrayscaleConversion() {
    ORBDetector detector(1000);
    
    // Test with color image
    cv::Mat color_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(color_image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
    
    std::vector<cv::KeyPoint> keypoints_color;
    cv::Mat descriptors_color;
    detector.detectAndCompute(color_image, keypoints_color, descriptors_color);
    
    // Test with grayscale image
    cv::Mat gray_image;
    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::KeyPoint> keypoints_gray;
    cv::Mat descriptors_gray;
    detector.detectAndCompute(gray_image, keypoints_gray, descriptors_gray);
    
    // Should detect similar number of features
    ASSERT_GREATER(keypoints_color.size(), 0);
    ASSERT_GREATER(keypoints_gray.size(), 0);
}

void testORBDetectorDifferentFeatureCounts() {
    // Test with different feature counts
    std::vector<int> feature_counts = {100, 500, 1000, 2000};
    
    cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(test_image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(test_image, cv::Point(300, 100), cv::Point(500, 300), cv::Scalar(128, 128, 128), -1);
    
    for (int nfeatures : feature_counts) {
        ORBDetector detector(nfeatures);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        detector.detectAndCompute(test_image, keypoints, descriptors);
        
        // Should detect features (may be less than requested due to image content)
        ASSERT_GREATER(keypoints.size(), 0);
        ASSERT_LESS(keypoints.size(), static_cast<size_t>(nfeatures + 100)); // Allow some tolerance
    }
}

int runORBDetectorTests() {
    std::cout << "=== ORB Detector Unit Tests ===" << std::endl;
    
    testORBDetectorBasicFunctionality();
    testORBDetectorEmptyImage();
    testORBDetectorGrayscaleConversion();
    testORBDetectorDifferentFeatureCounts();
    
    TestFramework::printSummary();
    return TestFramework::failedTests > 0 ? 1 : 0;
}

int main() {
    return runORBDetectorTests();
}
