#include "MergerHyperBlock.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------

// Define a macro to compare float4's for equality.
#define COMPARE_FLOAT4(a, b) ( (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z) && ((a).w == (b).w)) ? 0 : 1 )
// Define simple min/max macros for scalar floats.
#define min_f(a, b) ((a) > (b) ? (b) : (a))
#define max_f(a, b) ((a) > (b) ? (a) : (b))

// Hybrid kernel that uses block-level cooperation but vectorizes with float4's.
__global__ void mergerHyperBlocks(
    const int seedIndex, 
    int *readSeedQueue, 
    const int numBlocks, 
    const int numAttributes,   // padded to be a multiple of 4
    const int numPoints, 
    const float *opposingPoints,
    float *hyperBlockMins, 
    float *hyperBlockMaxes, 
    int *deleteFlags, 
    int *mergable
) 
{
    // Get our block and thread indices.
    const int blockIndex = blockIdx.x;
    const int localID = threadIdx.x;

    // Retrieve our seed block.
    const int seedBlock = readSeedQueue[seedIndex];

    // One thread per block sets initial flags for the seed block.
    if (localID == 0) {
        atomicMin(&deleteFlags[seedBlock], -9);
        atomicMin(&mergable[seedBlock], -1);
    }

    // Shared flag to mark if the candidate merge remains possible.
    __shared__ int blockMergable;
    // Shared flags for early-out: did we actually change bounds vs. seed or candidate?
    __shared__ int usedSeedBlock;
    __shared__ int usedCandidateBlock;

    // Calculate number of float4 elements we need.
    int numAttributesAsFours = numAttributes / 4;

    // Declare shared memory to hold the combined bounds.
    // We allocate space for 2 * numAttributes floats.
    extern __shared__ float sharedMem[];
    // Treat first half as an array of float4's for the mins...
    float4 *localCombinedMins = (float4*) sharedMem;
    // ...and the second half for the maxes.
    float4 *localCombinedMaxes = (float4*) (sharedMem + numAttributes);

    __syncthreads();

    // Each CUDA block will iterate over candidate Hyperblocks (with a stride of gridDim.x).
    for (int candidate = blockIndex; candidate < numBlocks; candidate += gridDim.x) {
        // Skip candidate if it is the seed or already flagged.
        if (candidate == seedBlock || deleteFlags[candidate] < 0)
            continue;

        // Reset our per-candidate flags.
        if (localID == 0) {
            blockMergable = 1;      // assume merge is allowed initially
            usedSeedBlock = 0;      // will be set if seed's values are not solely used
            usedCandidateBlock = 0; // will be set if candidate's values are not solely used
        }
        __syncthreads();

        // For each chunk (float4) of attributes, load the seed and candidate bounds,
        // compute the combined bounds, and store them in shared memory.
        for (int i = localID; i < numAttributesAsFours; i += blockDim.x) {
            // Load seed block's bounds.
            float4 seedMin = *((float4*)&hyperBlockMins[seedBlock * numAttributes] + i);
            float4 seedMax = *((float4*)&hyperBlockMaxes[seedBlock * numAttributes] + i);
            // Load candidate block's bounds.
            float4 candMin = *((float4*)&hyperBlockMins[candidate * numAttributes] + i);
            float4 candMax = *((float4*)&hyperBlockMaxes[candidate * numAttributes] + i);

            // Compute the combined bounds (min and max) using our simple scalar functions.
            float4 combinedMin;
            combinedMin.x = min_f(seedMin.x, candMin.x);
            combinedMin.y = min_f(seedMin.y, candMin.y);
            combinedMin.z = min_f(seedMin.z, candMin.z);
            combinedMin.w = min_f(seedMin.w, candMin.w);

            float4 combinedMax;
            combinedMax.x = max_f(seedMax.x, candMax.x);
            combinedMax.y = max_f(seedMax.y, candMax.y);
            combinedMax.z = max_f(seedMax.z, candMax.z);
            combinedMax.w = max_f(seedMax.w, candMax.w);

            // Write the combined bounds to shared memory.
            localCombinedMins[i] = combinedMin;
            localCombinedMaxes[i] = combinedMax;

            // Early-out check: if the combined bound differs from the seed block's bounds,
            // mark that candidate’s values were used.
            if ((combinedMin.x != seedMin.x) || (combinedMin.y != seedMin.y) ||
                (combinedMin.z != seedMin.z) || (combinedMin.w != seedMin.w) ||
                (combinedMax.x != seedMax.x) || (combinedMax.y != seedMax.y) ||
                (combinedMax.z != seedMax.z) || (combinedMax.w != seedMax.w)) {
                atomicOr(&usedSeedBlock, 1);
            }
            // Similarly, check against the candidate block's original bounds.
            if ((combinedMin.x != candMin.x) || (combinedMin.y != candMin.y) ||
                (combinedMin.z != candMin.z) || (combinedMin.w != candMin.w) ||
                (combinedMax.x != candMax.x) || (combinedMax.y != candMax.y) ||
                (combinedMax.z != candMax.z) || (combinedMax.w != candMax.w)) {
                atomicOr(&usedCandidateBlock, 1);
            }
        }
        __syncthreads();

        // Determine whether we really need to scan the dataset.
        // If one block's bounds were taken entirely as-is, then we can skip checking opposing points.
        int needDatasetCheck = (usedSeedBlock && usedCandidateBlock);

        // Now, check opposing points to see if any fall completely inside the new combined bounds.
        // The opposing points are split among threads.
        for (int pointIndex = localID; pointIndex < numPoints && blockMergable && needDatasetCheck; pointIndex += blockDim.x) {
            
            bool pointOutside = false; // assume point is out-of-bounds until proven inside
            
            // Loop over each float4 segment of the point's attributes.
            for (int i = 0; i < numAttributesAsFours; i++) {
            
                // Load four attributes for this point.
                float4 pointVal = *((float4*)&opposingPoints[pointIndex * numAttributes] + i);
                float4 combMin = localCombinedMins[i];
                float4 combMax = localCombinedMaxes[i];

                // If any attribute is outside the combined bounds, mark the point as valid.
                if (pointVal.x < combMin.x || pointVal.x > combMax.x ||
                    pointVal.y < combMin.y || pointVal.y > combMax.y ||
                    pointVal.z < combMin.z || pointVal.z > combMax.z ||
                    pointVal.w < combMin.w || pointVal.w > combMax.w) {
                    pointOutside = true;
                    break;
                }
            }
            // If the entire point lies within the combined bounds, then merging fails.
            if (!pointOutside) {
                blockMergable = 0;
                break;
            }
        }
        __syncthreads();

        // If the candidate block passes the check, update its global bounds.
        if (blockMergable) {
            for (int i = localID; i < numAttributesAsFours; i += blockDim.x) {
                *((float4*)&hyperBlockMins[candidate * numAttributes] + i) = localCombinedMins[i];
                *((float4*)&hyperBlockMaxes[candidate * numAttributes] + i) = localCombinedMaxes[i];
            }
            if (localID == 0) {
                // Mark the seed block as merged (or “deleted”).
                atomicMax(&deleteFlags[seedBlock], -1);
                // Indicate that the candidate block did merge.
                mergable[candidate] = 1;
            }
        } else {
            if (localID == 0) {
                // Merging failed for this candidate.
                mergable[candidate] = 0;
            }
        }
        __syncthreads();
    }
}


__global__ void rearrangeSeedQueue(const int deadSeedNum, int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;
    // now we are just going to loop through the seed queue and compute each blocks new position in the queue.
    for (int i = threadID; i < numBlocks; i += globalThreadCount) {
        
        if (i <= deadSeedNum){
            writeSeedQueue[i] = readSeedQueue[i];
            continue;
        }

        const int seed = readSeedQueue[i];

        if (mergable[seed] == 0) {
            int newIndex = deadSeedNum + 1;
            // Count only live, non-merged blocks in positions after deadSeedNum.
            for (int j = deadSeedNum + 1; j < i; j++) {
                int other = readSeedQueue[j];
                if (mergable[other] == 0)
                    newIndex++;
            }
            writeSeedQueue[newIndex] = seed;
        }
        // if we DID MERGE.
        else {
            int count = 0;
            // Count live merged blocks among indices after deadSeedNum.
            for (int j = deadSeedNum + 1; j < i; j++) {
                int other = readSeedQueue[j];
                if (mergable[other] == 1)
                    count++;
            }
            writeSeedQueue[numBlocks - 1 - count] = seed;
        }
    }
}

__global__ void resetMergableFlags(int *mergableFlags, const int numBlocks){
    // make all the flags 0.
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < numBlocks; i += gridDim.x * blockDim.x){
        mergableFlags[i] = 0;
    }
}

// ------------------------------------------------------------------------------------------------
// REMOVING USELESS HYPERBLOCKS KERNEL FUNCTION.
// RETAINS THE LOGIC FOR A DISJUNCTIVE BLOCK.
// LAUNCH 4 KERNELS!!! ASSIGN -> SUM -> FIND BETTER -> SUM. Once we have the count back, just delete all the HB's with 0 points remaining.
// ------------------------------------------------------------------------------------------------

// ASSIGN POINTS TO BLOCKS KERNEL FUNCTION.
// once every point has been assigned to a block, then we can start doing our removing of useless blocks.
__global__ void assignPointsToBlocks(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    int *dataPointBlock;       // pointer to the slot in the numPoints long array that tells us which block this point goes into.
    int currentBlock = 0;          // the current block we are on.
    int nextBlock = 1;             // the next block. useful because blocks are varying sizes.

    if (threadID < numPoints){
        const float *thisThreadPoint = &dataPointsArray[threadID * numAttributes];

        // set out pointer to where we assign this point to a block.
        dataPointBlock = &dataPointBlocks[threadID];
        *dataPointBlock = -1;

        // now we iterate through all the blocks. checking which block this point falls into first.
        while (currentBlock < numBlocks){

            // set up our start of mins and maxes.
            // we need these pointers because we could have a disjunctive block, where there are multiple of different attributes. hence the complex indexing logic later on.
            const float *startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            const float *startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
            const float *endOfBlock = &blockMins[blockEdges[nextBlock]];
            // now, we iterate through all the blocks, and the first one our point falls into, we set that block as the value of dataPointBlock, if not we put -1 and we have a coverage issue

            bool inThisBlock = true;

            // the x we are at, x0, x1, ...
            int particularAttribute = 0;

            // check through all the attributes for this block.
            while(startOfBlockMins < endOfBlock){

                // get the amount of x1's that we have in this particular block
                int countOfThisAttribute = (int)*startOfBlockMins;

                // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
                startOfBlockMins++;
                startOfBlockMaxes++;

                // now loop that many times, checking if the point is in bounds of any of those intervals
                // we don't actually use i here, because we don't want to check the next attribute of our point on accident. since we may have 2 x2's and such.
                bool inBounds = false;
                for(int i = 0; i < countOfThisAttribute; i++){

                    const double min = *startOfBlockMins;
                    startOfBlockMins++;

                    const double max = *startOfBlockMaxes;
                    startOfBlockMaxes++;

                    const double pointValue = thisThreadPoint[particularAttribute];

                    // this loop is for the disjunctive blocks. if there is just one x, it doesn't matter. when we have 4 x2's to consider, once we are in one of them, we are done.
                    if(pointValue >= min && pointValue <= max){
                        inBounds = true;
                        break;
                    }
                }
                if (!inBounds){
                    inThisBlock = false;
                    break;
                }
                particularAttribute++;
            }
            // if in this block, we can set dataPointBlock and we're done
            if (inThisBlock){
                *dataPointBlock = currentBlock;
                break;
            }
            // increment the currentBlock and the next block.
            currentBlock++;
            nextBlock++;
        }
    }
}

// NOW OUR FUNCTION WHICH SUMS UP THE AMOUNT OF POINTS PER BLOCK
__global__ void sumPointsPerBlock(int *dataPointBlocks, const int numPoints, int *numPointsInBlocks){
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // increment the count of points in the block of whatever block our point is in
    // each thread does one single point
    if (threadID < numPoints){
        atomicAdd(&numPointsInBlocks[dataPointBlocks[threadID]], 1);
    }
}
__global__ void findBetterBlocks(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks, int *numPointsInBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < numPoints){

        int currentBlock = dataPointBlocks[threadID];
        int nextBlock = currentBlock + 1;
        int *dataPointBlock = &dataPointBlocks[threadID];
        // largest block size is our current block we are assigned to for this point
        int largestBlockSize = numPointsInBlocks[currentBlock];

        // we have a coverage issue if this happens.
        if (currentBlock == -1){
            return;
        }

        const float *thisThreadPoint = &dataPointsArray[threadID * numAttributes];

        // now we iterate through and finally assign our point to the most populous block we find that we fit into.
        while (currentBlock < numBlocks){

            // now, we iterate through all the blocks after the one we chose, and if we find a bigger one we fit into, we go into that one.
            const float *startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            const float *startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
            const float *endOfBlock = &blockMins[blockEdges[nextBlock]];
            bool inThisBlock = true;

            // the x we are at, x0, x1, ...
            int particularAttribute = 0;

            // check through all the attributes for this block.
            while(startOfBlockMins < endOfBlock){

                // get the amount of x1's that we have in this particular block
                int countOfThisAttribute = (int)*startOfBlockMins;

                // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
                startOfBlockMins++;
                startOfBlockMaxes++;

                // now loop that many times, checking if the point is in bounds of any of those intervals
                bool inBounds = false;
                for(int att = 0; att < countOfThisAttribute; att++){

                    const double min = *startOfBlockMins;
                    startOfBlockMins++;

                    const double max = *startOfBlockMaxes;
                    startOfBlockMaxes++;

                    const double pointValue = thisThreadPoint[particularAttribute];

                    if (pointValue >= min && pointValue <= max && numPointsInBlocks[currentBlock] > largestBlockSize){
                        inBounds = true;
                        break;
                    }
                }
                if (!inBounds){
                    inThisBlock = false;
                    break;
                }
                particularAttribute++;
            }
            if (inThisBlock){
                *dataPointBlock = currentBlock;
                largestBlockSize = numPointsInBlocks[currentBlock];
            }
            currentBlock++;
            nextBlock++;
        }
    }
}

// -------------------------------------------------------------------
// Removing useless attributes functions
/*
 *
 *
 *
 */
__global__ void removeUselessAttributes(float* mins, float* maxes, const int* intervalCounts, const int minMaxLen, const int* blockEdges, const int numBlocks, const int* blockClasses, char* attrRemoveFlags, const int fieldLen, const float* dataset, const int numPoints, const int* classBorder, const int numClasses, const int *attributeOrder) {

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadID >= numBlocks) return;

    // Get block-specific data
    float* blockMins = &mins[blockEdges[threadID]];
    float* blockMaxes = &maxes[blockEdges[threadID]];
    const int* blockIntervalCounts = &intervalCounts[threadID * fieldLen];
    int classNum = blockClasses[threadID];

    // Class boundaries
    const int startClass = classBorder[classNum] * fieldLen;
    const int endClass = classBorder[classNum+1] * fieldLen;

    // Try removing each attribute
    for(int removedIndex = 0; removedIndex < fieldLen; removedIndex++) {
        // we are using the attribute order with different orderings per class. therefore, we must offset into our own class.
        int removed = attributeOrder[fieldLen * classNum + removedIndex];

        // Calculate offset to the start of this attribute's intervals
        int checkOffset = 0;
        for(int i = 0; i < removed; i++) {
            checkOffset += blockIntervalCounts[i];
        }

        // Skip if this attribute is already marked as removed (checking first interval)
        if(blockMins[checkOffset] == 0.0 && blockMaxes[checkOffset] == 1.0) {
            continue;
        }

        bool someOneInBounds = false;
        // Check all points from other classes
        for(int j = 0; j < numPoints * fieldLen; j += fieldLen) {
            if(j < endClass && j >= startClass) continue;

            bool pointInside = true;
            int totalOffset = 0;

            // Check each attribute
            for(int attr = 0; attr < fieldLen; attr++) {
                if(attr == removed) {
                    totalOffset += blockIntervalCounts[attr];
                    continue;
                }

                const float attrValue = dataset[j + attr];
                bool inAnInterval = false;

                for(int intv = 0; intv < blockIntervalCounts[attr]; intv++) {
                    float min = blockMins[totalOffset + intv];
                    float max = blockMaxes[totalOffset + intv];

                    if(attrValue <= max && attrValue >= min) {
                        inAnInterval = true;
                        break;
                    }
                }

                if(!inAnInterval) {
                    pointInside = false;
                    break;
                }

                totalOffset += blockIntervalCounts[attr];
            }

            if(pointInside) {
                someOneInBounds = true;
                break;
            }
        }

        // If no points from other classes fall in, we can remove this attribute
        if(!someOneInBounds) {
            int removeOffset = 0;
            for(int i = 0; i < removed; i++) {
                removeOffset += blockIntervalCounts[i];
            }

            // Reset intervals for removed attribute to [0,1]
            for(int i = 0; i < blockIntervalCounts[removed]; i++) {
                blockMins[removeOffset + i] = 0.0;
                blockMaxes[removeOffset + i] = 1.0;
            }

            // Mark attribute as removed
            attrRemoveFlags[fieldLen * threadID + removed] = 1;
        }
    }
}

void mergerHyperBlocksWrapper(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float *opposingPoints,float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable, int gridSize, int blockSize, int sharedMemSize){
	mergerHyperBlocks<<<gridSize, blockSize, sharedMemSize>>>(
            seedIndex,
            readSeedQueue,
            numBlocks,
            numAttributes,
            numPoints,
            opposingPoints,
	    	hyperBlockMins,
			hyperBlockMaxes,
			deleteFlags,
			mergable
		);
}

void rearrangeSeedQueueWrapper(const int deadSeedCount, int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks, int gridSize, int blockSize){
    rearrangeSeedQueue<<<gridSize, blockSize>>>(deadSeedCount, readSeedQueue, writeSeedQueue, deleteFlags, mergable, numBlocks);
}
void resetMergableFlagsWrapper(int *mergableFlags, const int numBlocks, int gridSize, int blockSize){
	resetMergableFlags<<<gridSize, blockSize>>>(mergableFlags, numBlocks);
}

/**
* --------------- wrapper functions to run the removing useless blocks functionality.
*/
void assignPointsToBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks, int gridSize, int blockSize){
    assignPointsToBlocks<<<gridSize, blockSize>>>(dataPointsArray, numAttributes, numPoints, blockMins, blockMaxes, blockEdges, numBlocks, dataPointBlocks);
}

void sumPointsPerBlockWrapper(int *dataPointBlocks, const int numPoints, int *numPointsInBlocks, int gridSize, int blockSize){
    sumPointsPerBlock<<<gridSize, blockSize>>>(dataPointBlocks, numPoints, numPointsInBlocks);
}

void findBetterBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks, int *numPointsInBlocks, int gridSize, int blockSize){
    findBetterBlocks<<<gridSize, blockSize>>>(dataPointsArray, numAttributes, numPoints, blockMins, blockMaxes, blockEdges, numBlocks, dataPointBlocks, numPointsInBlocks);
}

/*
 * Wrapper functions for the removing useless attributes.
 * */

void removeUselessAttributesWrapper(float* mins, float* maxes, const int* intervalCounts, const int minMaxLen, const int* blockEdges, const int numBlocks, const int* blockClasses, char* attrRemoveFlags, const int fieldLen, const float* dataset, const int numPoints, const int* classBorder, const int numClasses, const int *attributeOrder, int gridSize, int blockSize){
   removeUselessAttributes<<<gridSize, blockSize>>>(mins, maxes, intervalCounts, minMaxLen, blockEdges, numBlocks, blockClasses, attrRemoveFlags, fieldLen, dataset, numPoints, classBorder, numClasses, attributeOrder);
}