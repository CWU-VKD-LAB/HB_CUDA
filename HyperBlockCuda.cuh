#ifndef HyperBlockCuda_CUH
#define HyperBlockCuda_CUH

#include <cuda_runtime.h>


// === KERNELS: ===
__global__ void mergerHyperBlocks(
    const int seedIndex, int *readSeedQueue, const int numBlocks,
    const int numAttributes, const int numPoints, const float *opposingPoints,
    float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable);

__global__ void rearrangeSeedQueue(
    int *readSeedQueue, int *writeSeedQueue, int *deleteFlags,int *mergable, const int numBlocks);

__global__ void resetMergableFlags(int *mergableFlags, const int numBlocks);


__global__ void assignPointsToBlocks(
    const float *dataPointsArray, const int numAttributes, const int numPoints,
    const float *blockMins, const float *blockMaxes, const int *blockEdges,
    const int numBlocks, int *dataPointBlocks);


// === WRAPPERS: USE THESE TO CALL THE KERNELS FROM A CPP FILE!  ===
void mergerHyperBlocksWrapper(
    const int seedIndex, int *readSeedQueue, const int numBlocks,
    const int numAttributes, const int numPoints, const float *opposingPoints,
    float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable, int gridSize, int blockSize, int sharedMemSize);

void rearrangeSeedQueueWrapper(int *readSeedQueue, int *writeSeedQueue, int *deleteFlags,int *mergable, const int numBlocks, int gridSize, int blockSize);

void resetMergableFlagsWrapper(int *mergableFlags, const int numBlocks, int gridSize, int blockSize);

void assignPointsToBlocksWrapper(
    const float *dataPointsArray, const int numAttributes, const int numPoints,
    const float *blockMins, const float *blockMaxes, const int *blockEdges,
    const int numBlocks, int *dataPointBlocks);

#endif
