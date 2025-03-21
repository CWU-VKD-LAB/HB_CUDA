#include "CudaUtil.h"

// Get the number of CUDA cores per SM for a given GPU architecture
int getCudaCoresPerSM(int major, int minor) {
    switch (major) {
        case 2: return 32;
        case 3: return 192;
        case 5: return 128;
        case 6: return (minor == 1 ? 128 : 64);
        case 7: return 64;
        case 8: return (minor == 0 ? 64 : 128);
        case 9: return 128;
        default: return 2; // Fallback for unknown architectures
    }
}

// Get the total number of CUDA cores on the GPU
int getNumberCudaCores(const cudaDeviceProp& prop) {
    int coresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
    return prop.multiProcessorCount * coresPerSM;
}

int getNumberCudaSMs(const cudaDeviceProp& prop) {
    return prop.multiProcessorCount;
}

int getNumberCudaThreadsPerSM(const cudaDeviceProp& prop) {
    return getCudaCoresPerSM(prop.major, prop.minor);
}