#!/bin/bash

# Hardware test runner script for RGB-D Mapping System
# WARNING: This requires a RealSense D435i camera to be connected!

echo "=== RGB-D Mapping System HARDWARE Tests ==="
echo "WARNING: These tests require a RealSense D435i camera to be connected!"
echo "If no camera is connected, these tests will FAIL (which is expected)."
echo ""

# Build the main project first
echo "Building main project..."
cd "$(dirname "$0")"
./build.sh

if [ $? -ne 0 ]; then
    echo "[FAIL] Main project build failed!"
    exit 1
fi

# Run hardware tests
echo "Running hardware integration tests..."
echo ""

# Check if camera is connected
echo "Checking for RealSense camera..."
if lsusb | grep -q "Intel"; then
    echo "[PASS] Intel device detected in USB"
else
    echo "[WARN] No Intel device detected in USB - camera may not be connected"
fi

echo ""
echo "--- Hardware Integration Tests ---"
./build/tests/test_hardware
hardware_result=$?

# Summary
echo ""
echo "=== Hardware Test Results Summary ==="
if [ $hardware_result -eq 0 ]; then
    echo "[PASS] HARDWARE TESTS PASSED! Camera is working correctly."
else
    echo "[FAIL] HARDWARE TESTS FAILED!"
    echo "This is expected if:"
    echo "  - No RealSense camera is connected"
    echo "  - Camera is not properly configured"
    echo "  - Camera drivers are not installed"
    echo ""
    echo "To fix:"
    echo "  1. Connect RealSense D435i camera via USB"
    echo "  2. Install RealSense SDK: https://github.com/IntelRealSense/librealsense"
    echo "  3. Run: sudo apt install librealsense2-dev"
fi

exit $hardware_result
