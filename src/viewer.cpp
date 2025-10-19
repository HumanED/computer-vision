#include <opencv2/opencv.hpp>

// Simple viewer implementation for headless mode compatibility
// This is a placeholder that can be extended later for visualization

void displayFrame(const cv::Mat &color, const cv::Mat &depth) {
    // For now, just print frame info
    std::cout << "[Viewer] Color frame: " << color.cols << "x" << color.rows 
              << ", Depth frame: " << depth.cols << "x" << depth.rows << std::endl;
}
