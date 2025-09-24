#!/bin/bash

# HENN-Faiss Build Script (No Tests)
# This script builds Faiss with HENN implementation without running tests

set -e  # Exit on any error

echo "🚀 Starting HENN-Faiss build process..."
echo "========================================"

# Configuration
BUILD_DIR="./build"
SOURCE_DIR="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "HENN source directory '$SOURCE_DIR' not found!"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

print_status "Found HENN source directory: $SOURCE_DIR"

# Create and enter build directory
echo -e "\n📁 Setting up build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

print_status "Entered build directory: $(pwd)"

# Check for required tools
echo -e "\n🔍 Checking for required build tools..."
command -v cmake >/dev/null 2>&1 || { print_error "cmake is required but not installed. Aborting."; exit 1; }
command -v make >/dev/null 2>&1 || { print_error "make is required but not installed. Aborting."; exit 1; }
command -v g++ >/dev/null 2>&1 || { print_error "g++ is required but not installed. Aborting."; exit 1; }

print_status "All required build tools found"

# Configure with CMake
echo -e "\n⚙️  Configuring build with CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=ON
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX=../install
)

if cmake "${CMAKE_ARGS[@]}" ..; then
    print_status "CMake configuration completed successfully"
else
    print_error "CMake configuration failed!"
    exit 1
fi

# Build
echo -e "\n🔨 Building HENN-Faiss..."
echo "This may take several minutes depending on your system..."

# Get number of CPU cores for parallel build
NPROC=$(nproc 2>/dev/null || echo 4)
print_warning "Building with $NPROC parallel jobs"

if make -j"$NPROC" faiss; then
    print_status "HENN-Faiss library built successfully"
else
    print_error "Build failed!"
    exit 1
fi

# Build Python bindings
echo -e "\n🐍 Building Python bindings..."
if make -j"$NPROC" swigfaiss; then
    print_status "Python bindings built successfully"
else
    print_error "Python bindings build failed!"
    exit 1
fi

# Check if the library was built
if [ -f "faiss/libfaiss.so" ]; then
    print_status "Found built library: faiss/libfaiss.so"
else
    print_warning "Library file not found at expected location"
fi

# Check Python bindings
if [ -f "faiss/python/swigfaiss.py" ]; then
    print_status "Found Python bindings: faiss/python/swigfaiss.py"
else
    print_warning "Python bindings not found at expected location"
fi

echo -e "\n🎉 Build completed successfully!"
echo "========================================"
echo "To use HENN-Faiss in Python, add this to your sys.path:"
echo "sys.path.insert(0, '$(pwd)/faiss/python')"
echo ""
echo "Then import with:"
echo "import swigfaiss as faiss"
echo ""
echo "Test your HENN implementation with:"
echo "python ../test_henn_working.py"
echo ""
echo "Run the comprehensive benchmark with:"
echo "python ../comprehensive_benchmark.py"
