#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Simple test framework
class TestFramework {
public:
    static int totalTests;
    static int passedTests;
    static int failedTests;
    
    static void runTest(const std::string& testName, bool result) {
        totalTests++;
        if (result) {
            passedTests++;
            std::cout << "[PASS] " << testName << " - PASSED" << std::endl;
        } else {
            failedTests++;
            std::cout << "[FAIL] " << testName << " - FAILED" << std::endl;
        }
    }
    
    static void printSummary() {
        std::cout << "\n=== TEST SUMMARY ===" << std::endl;
        std::cout << "Total Tests: " << totalTests << std::endl;
        std::cout << "Passed: " << passedTests << std::endl;
        std::cout << "Failed: " << failedTests << std::endl;
        std::cout << "Success Rate: " << (passedTests * 100.0 / totalTests) << "%" << std::endl;
    }
    
    static void reset() {
        totalTests = 0;
        passedTests = 0;
        failedTests = 0;
    }
};

// Test macros for convenience
#define ASSERT_TRUE(condition) TestFramework::runTest(__FUNCTION__, (condition))
#define ASSERT_FALSE(condition) TestFramework::runTest(__FUNCTION__, !(condition))
#define ASSERT_EQUAL(expected, actual) TestFramework::runTest(__FUNCTION__, (expected) == (actual))
#define ASSERT_NOT_EQUAL(expected, actual) TestFramework::runTest(__FUNCTION__, (expected) != (actual))
#define ASSERT_GREATER(value, threshold) TestFramework::runTest(__FUNCTION__, (value) > (threshold))
#define ASSERT_LESS(value, threshold) TestFramework::runTest(__FUNCTION__, (value) < (threshold))

// Initialize static members
int TestFramework::totalTests = 0;
int TestFramework::passedTests = 0;
int TestFramework::failedTests = 0;
