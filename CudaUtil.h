#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Allocate and copy memory from host to device (template function)
template <typename T>
CUdeviceptr allocateAndCopy(const std::vector<T>& hostData);

// Get the number of CUDA cores per SM for a given GPU architecture
int getCudaCoresPerSM(int major, int minor);

// Get the total number of CUDA cores on the GPU
int getNumberCudaCores(const cudaDeviceProp& prop);

#include "CudaUtil.tpp" // Include the template implementation file

#endif // CUDA_UTIL_H
