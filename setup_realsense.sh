#!/usr/bin/env bash
set -e

if [ -f /etc/debian_version ]; then
    sudo apt update
    sudo apt install -y git cmake build-essential pkg-config libusb-1.0-0-dev \
        libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
elif [ -f /etc/fedora-release ]; then
    sudo dnf install -y git cmake make gcc-c++ libusb1-devel glfw-devel \
        mesa-libGL-devel mesa-libGLU-devel gtk3-devel pkgconf-pkg-config
else
    echo "Unsupported OS"
    exit 1
fi

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_WAYLAND=ON -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true
make -j$(nproc)
sudo make install
