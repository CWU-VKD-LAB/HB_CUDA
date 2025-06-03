# Hyperblocks

**Hyperblocks** is a standalone C++/CUDA implementation of the DV2.0 Hyperblocks model, originally developed in Java at the CWU VKD Lab. It is designed to be high-performing, explainable, and cross-platform, with GPU acceleration and parallelism support.

##  What Are Hyperblocks?

Hyperblocks are an interpretable, rule-based machine learning model. Each Hyperblock defines axis-aligned bounds (min/max) for each attribute in the dataset, forming a hyper-rectangle in feature space.

This structure supports:
- Transparent decision-making
- Easy visualization (e.g., Parallel Coordinates)
- Rule simplification and fusion
- Compatibility with Subject-Matter Expert (SME) analysis

## Prerequisites

To build this project, you need:

- CMake 3.18 or higher
- CUDA Toolkit (tested with 12.6)
- A C++17-compatible compiler (GCC, Clang, MSVC)
- OpenMP (optional but recommended)

To run this project, you need:

- CUDA compatible GPU

##  Build Instructions

Clone the repository and run the following:

```bash
# Step 1: Create a build directory
mkdir -p build
cd build

# Step 2: Generate the build files. (may need -DCMAKE_BUILD_TYPE=Debug on Linux)
cmake ..  


# Step 3: Compile the project
cmake --build . --config Debug
```



##  Run Instructions

After building with CMake (inside the `build/` directory), you can run the program as follows:


Run the executable from the project root:
```bash
cd ..
Hyperblocks.exe
```


##  Command Line Compiling Instructions (Optional)

This section is for users who prefer to compile the program manually instead of using CMake.

### Windows (MSVC)

- **Compile**:
```bash
nvcc -Xcompiler /openmp -o a.exe ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./cuda_util/CudaUtil.cpp ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./ClassificationTesting/ClassificationTests.cpp -g -G -O3
```

- **Run**:
```bash
a.exe
```

### Linux

- **Compile**:
```bash
nvcc -Xcompiler -fopenmp -o a ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./cuda_util/CudaUtil.cpp ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./ClassificationTesting/ClassificationTests.cpp -g -G -O3
```

- **Run**:
```bash
./a
```