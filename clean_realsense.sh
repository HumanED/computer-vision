#!/usr/bin/env bash
set -e

if [ -f /etc/debian_version ]; then
    sudo rm -rf /usr/local/lib/librealsense* /usr/local/include/librealsense* \
        /usr/local/bin/rs-* /usr/local/share/librealsense* \
        /etc/udev/rules.d/99-realsense-libusb.rules
    sudo ldconfig
elif [ -f /etc/fedora-release ]; then
    sudo rm -rf /usr/local/lib64/librealsense* /usr/local/include/librealsense* \
        /usr/local/bin/rs-* /usr/local/share/librealsense* \
        /etc/udev/rules.d/99-realsense-libusb.rules
    sudo ldconfig
else
    echo "Unsupported OS"
    exit 1
fi

if [ -d "$HOME/librealsense" ]; then
    rm -rf "$HOME/librealsense"
fi
rm -rf ~/Pangolin/build

echo "librealsense completely removed."
