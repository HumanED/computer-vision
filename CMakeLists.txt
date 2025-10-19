cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

option(HEADLESS "Build for headless/offscreen mode" OFF)
if(HEADLESS)
    add_definitions(-DHEADLESS)
    message(STATUS "Headless mode enabled")
else()
    message(STATUS "GUI mode enabled")
endif()

# Dependencies
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(realsense2 REQUIRED)

if(NOT HEADLESS)
    find_package(Pangolin REQUIRED)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
)

if(NOT HEADLESS)
    include_directories(${Pangolin_INCLUDE_DIRS})
endif()

# Core libraries
add_library(realsense_camera src/realsense_camera.cpp)
target_link_libraries(realsense_camera ${OpenCV_LIBS} ${realsense2_LIBRARY})

add_library(orb_detector src/orb_detector.cpp)
target_link_libraries(orb_detector ${OpenCV_LIBS})

add_library(initializer src/initializer.cpp)
target_link_libraries(initializer realsense_camera orb_detector ${OpenCV_LIBS})

# Viewer library
add_library(viewer src/viewer.cpp)
target_link_libraries(viewer ${OpenCV_LIBS})

if(NOT HEADLESS)
    target_link_libraries(viewer ${Pangolin_LIBRARIES})
endif()

# Executable
add_executable(test_initial_setup src/test_initial_setup.cpp)
target_link_libraries(test_initial_setup
    viewer
    realsense_camera
    orb_detector
    initializer
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
)

# Map initialization executable
add_executable(initialize_map src/initialize_map.cpp)
target_link_libraries(initialize_map
    realsense_camera
    orb_detector
    initializer
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
)

if(NOT HEADLESS)
    target_link_libraries(test_initial_setup ${Pangolin_LIBRARIES})
endif()

# RPATH
set_target_properties(test_initial_setup PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "/usr/local/lib"
)

# Enable testing
enable_testing()
add_subdirectory(tests)
