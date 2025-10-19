#include "orb_detector.hpp"
#include <iostream>

ORBDetector::ORBDetector(int nfeatures) {
    orb = cv::ORB::create(nfeatures);
    std::cout << "[ORB] Detector initialized with " << nfeatures << " features" << std::endl;
}

void ORBDetector::detectAndCompute(const cv::Mat &img,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &descriptors) {
    if (img.empty()) {
        std::cerr << "[ORB] Input image is empty" << std::endl;
        return;
    }
    
    std::cout << "[ORB] Input image: " << img.cols << "x" << img.rows 
              << ", channels: " << img.channels() << ", type: " << img.type() << std::endl;
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    std::cout << "[ORB] Grayscale image: " << gray.cols << "x" << gray.rows 
              << ", channels: " << gray.channels() << ", type: " << gray.type() << std::endl;
    
    // Check image statistics
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    std::cout << "[ORB] Image stats - Mean: " << mean[0] << ", StdDev: " << stddev[0] << std::endl;
    
    // Detect keypoints and compute descriptors
    orb->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
    
    std::cout << "[ORB] Detected " << keypoints.size() << " keypoints" << std::endl;
    
    // If no keypoints, try with different parameters and preprocessing
    if (keypoints.empty()) {
        std::cout << "[ORB] No keypoints detected, trying with relaxed parameters..." << std::endl;
        
        // Try histogram equalization to improve contrast
        cv::Mat enhanced;
        cv::equalizeHist(gray, enhanced);
        
        // Create ORB with more relaxed parameters
        auto relaxed_orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        relaxed_orb->detectAndCompute(enhanced, cv::Mat(), keypoints, descriptors);
        
        std::cout << "[ORB] With relaxed parameters: " << keypoints.size() << " keypoints" << std::endl;
        
        // If still no keypoints, try even more relaxed parameters
        if (keypoints.empty()) {
            std::cout << "[ORB] Still no keypoints, trying very relaxed parameters..." << std::endl;
            
            auto very_relaxed_orb = cv::ORB::create(200, 1.1f, 4, 15, 0, 2, cv::ORB::HARRIS_SCORE, 15, 10);
            very_relaxed_orb->detectAndCompute(enhanced, cv::Mat(), keypoints, descriptors);
            
            std::cout << "[ORB] With very relaxed parameters: " << keypoints.size() << " keypoints" << std::endl;
        }
    }
}
