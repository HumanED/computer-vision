#include "test_framework.hpp"
#include <iostream>
#include <vector>
#include <string>

// Forward declarations for test functions
int runORBDetectorTests();
int runInitializerTests();
int runIntegrationTests();

int main() {
    std::cout << "=== RGB-D Mapping System Test Suite ===" << std::endl;
    std::cout << "Testing all components of the SLAM initialization system\n" << std::endl;
    
    int total_failures = 0;
    
    // Run unit tests
    std::cout << "\n--- Running Unit Tests ---" << std::endl;
    TestFramework::reset();
    total_failures += runORBDetectorTests();
    
    TestFramework::reset();
    total_failures += runInitializerTests();
    
    // Run integration tests
    std::cout << "\n--- Running Integration Tests ---" << std::endl;
    TestFramework::reset();
    total_failures += runIntegrationTests();
    
    // Final summary
    std::cout << "\n=== FINAL TEST SUMMARY ===" << std::endl;
    if (total_failures == 0) {
        std::cout << "🎉 ALL TESTS PASSED! System is ready for use." << std::endl;
    } else {
        std::cout << "⚠️  " << total_failures << " test suites failed. Please review the output above." << std::endl;
    }
    
    return total_failures;
}
