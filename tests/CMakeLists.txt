# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(realsense2 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${realsense2_INCLUDE_DIRS}
)

# Link to main project libraries
link_directories(${CMAKE_SOURCE_DIR}/build)

# Unit Tests
add_executable(test_orb_detector unit/test_orb_detector.cpp)
target_link_libraries(test_orb_detector 
    ${CMAKE_SOURCE_DIR}/build/liborb_detector.a
    ${OpenCV_LIBS}
)

add_executable(test_initializer unit/test_initializer.cpp)
target_link_libraries(test_initializer 
    ${CMAKE_SOURCE_DIR}/build/libinitializer.a
    ${CMAKE_SOURCE_DIR}/build/liborb_detector.a
    ${CMAKE_SOURCE_DIR}/build/librealsense_camera.a
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
)

# Integration Tests
add_executable(test_integration integration/test_integration.cpp)
target_link_libraries(test_integration
    ${CMAKE_SOURCE_DIR}/build/libinitializer.a
    ${CMAKE_SOURCE_DIR}/build/liborb_detector.a
    ${CMAKE_SOURCE_DIR}/build/librealsense_camera.a
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
)

# Hardware Integration Test (requires real camera)
add_executable(test_hardware integration/test_hardware.cpp)
target_link_libraries(test_hardware
    ${CMAKE_SOURCE_DIR}/build/libinitializer.a
    ${CMAKE_SOURCE_DIR}/build/liborb_detector.a
    ${CMAKE_SOURCE_DIR}/build/librealsense_camera.a
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_orb_detector
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_initializer
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_integration
    DEPENDS test_orb_detector test_initializer test_integration
    COMMENT "Running all tests"
)
