#ifndef CUDA_UTIL_TPP
#define CUDA_UTIL_TPP

#include "CudaUtil.h"

template <typename T>
CUdeviceptr allocateAndCopy(const std::vector<T>& hostData) {
    CUdeviceptr devicePointer;
    size_t dataSize = hostData.size() * sizeof(T);

    // Allocate memory on the device
    cuMemAlloc(&devicePointer, dataSize);

    // Copy data from host to device
    cuMemcpyHtoD(devicePointer, hostData.data(), dataSize);

    return devicePointer;
}

#endif // CUDA_UTIL_TPP
