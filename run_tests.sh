#!/bin/bash

# Test runner script for RGB-D Mapping System

echo "=== RGB-D Mapping System Test Suite ==="
echo "Building and running tests..."

# Build the main project first
echo "Building main project..."
cd "$(dirname "$0")"
./build.sh

if [ $? -ne 0 ]; then
    echo "[FAIL] Main project build failed!"
    exit 1
fi

# Run tests from the main build directory
echo "Running tests..."
echo ""

# Run individual test suites
echo "--- ORB Detector Tests ---"
./build/tests/test_orb_detector
orb_result=$?

echo ""
echo "--- Initializer Tests ---"
./build/tests/test_initializer
init_result=$?

echo ""
echo "--- Integration Tests ---"
./build/tests/test_integration
integration_result=$?

# Summary
echo ""
echo "=== Test Results Summary ==="
echo "ORB Detector Tests: $([ $orb_result -eq 0 ] && echo "[PASS]" || echo "[FAIL]")"
echo "Initializer Tests: $([ $init_result -eq 0 ] && echo "[PASS]" || echo "[FAIL]")"
echo "Integration Tests: $([ $integration_result -eq 0 ] && echo "[PASS]" || echo "[FAIL]")"

total_failures=$((orb_result + init_result + integration_result))

if [ $total_failures -eq 0 ]; then
    echo ""
    echo "[PASS] ALL TESTS PASSED! System is ready for use."
    exit 0
else
    echo ""
    echo "[WARN] Some tests failed. Please review the output above."
    exit 1
fi
