                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build || exit

# Configure CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build using all available CPU cores
make -j$(sysctl -n hw.ncpu)

echo "Build finished. Executables should be in the build directory."
