cmake_minimum_required(VERSION 3.18)  # 3.18+ ensures good CUDA support
project(Hyperblocks LANGUAGES CXX CUDA)

# Put the binary in the project root regardless of config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(OpenMP_FLAGS OpenMP::OpenMP_CXX)
endif()

# Enable position-independent code if needed (good practice)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# C++ settings (Can change these if needed, cluster only has like C++ 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: enable debug flags (remove for release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

# Source files
set(SOURCES
        ./Host.cu
        ./hyperblock_generation/MergerHyperBlock.cu
        ./interval_hyperblock/IntervalHyperBlock.cu
        ./simplifications/Simplifications.cu

        ./hyperblock/HyperBlock.cpp
        ./cuda_util/CudaUtil.cpp
        ./data_utilities/DataUtil.cpp
        ./knn/Knn.cpp
        ./screen_output/PrintingUtil.cpp
        ./ClassificationTesting/ClassificationTests.cpp
)

# Add executable target
add_executable(Hyperblocks ${SOURCES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(Hyperblocks PRIVATE ${OpenMP_FLAGS})
endif()