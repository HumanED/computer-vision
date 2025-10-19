#include "realsense_camera.hpp"
#include "orb_detector.hpp"
#include "initializer.hpp"
#include <iostream>

int main() {
    std::cout << "[Main] Starting map initialization..." << std::endl;
    
    RealSenseCamera camera;
    Initializer initializer;

    if (initializer.initialize(camera)) {
        std::cout << "[Main] Map initialization completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "[Main] Map initialization failed!" << std::endl;
        return 1;
    }
}
