#include "HyperBlockCuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------
__global__ void mergerHyperBlocks(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float* __restrict__ points, float *hyperBlockMins, float *hyperBlockMaxes, int* deleteFlags, int* mergable, float* combinedMins, float* combinedMaxes){
    
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // get our seed block.
    const int seedBlock = readSeedQueue[seedIndex];

  	// Put the seed block attributes in shared memory
    extern __shared__ float seedBlockAttributes[];
    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];

    // put seed block into shared mem
    for (int index = threadIdx.x; index < numAttributes; index += blockDim.x){
        seedBlockMins[index] = hyperBlockMins[seedBlock * numAttributes + index];
        seedBlockMaxes[index] = hyperBlockMaxes[seedBlock * numAttributes + index];
    }

	// sync block so shared mem is right.
    __syncthreads();

    // if our threadID corresponds to a block, which is not seedblock or already dead, we do our merging business.
    if (threadID < numBlocks && threadID != seedBlock && deleteFlags[threadID] != -1 ){

        // float 4 pointer to look through the dataset
        float4 *points4Pointer = (float4*)points;

        // our local float4's which will actually be holding these things. 
        float4 fourMinFloats;
        float4 fourMaxFloats;
        float4 fourAttributes;
    
        // get our float4 pointers to point at the combined mins and maxes for this block.
        float4 *thisBlockCombinedMaxes4 = (float4*)&combinedMaxes[threadID * numAttributes];
        float4 *thisBlockCombinedMins4 = (float4*)&combinedMins[threadID * numAttributes];

        // float4 pointer for our points. allows us to grab 4 floats at a time out of global memory
        float4 *hyperBlockMins4 = (float4*) &hyperBlockMins[threadID * numAttributes];
        float4 *hyperBlockMaxes4 = (float4*) &hyperBlockMaxes[threadID * numAttributes];

        // first we build our combined list.
        for (int i = 0; i < numAttributes; i += 4){
            fourMinFloats = hyperBlockMins4[i / 4];
            fourMaxFloats = hyperBlockMaxes4[i / 4];

            // update our four mins and maxes compared to seed block. that is, take the min of the two and the max of the two HBs bounds.
            fourMinFloats.x = fminf(seedBlockMins[i], fourMinFloats.x);
            fourMinFloats.y = fminf(seedBlockMins[i + 1], fourMinFloats.y);
            fourMinFloats.z = fminf(seedBlockMins[i + 2], fourMinFloats.z);
            fourMinFloats.w = fminf(seedBlockMins[i + 3], fourMinFloats.w);
            fourMaxFloats.x = fmaxf(seedBlockMaxes[i], fourMaxFloats.x);
            fourMaxFloats.y = fmaxf(seedBlockMaxes[i + 1], fourMaxFloats.y);
            fourMaxFloats.z = fmaxf(seedBlockMaxes[i + 2], fourMaxFloats.z);
            fourMaxFloats.w = fmaxf(seedBlockMaxes[i + 3], fourMaxFloats.w);

            // throw those four mins and maxes into our combined mins and maxes
            thisBlockCombinedMins4[i / 4] = fourMinFloats;
            thisBlockCombinedMaxes4[i / 4] = fourMaxFloats;
        }

        
        char allPassed = 1;                         // sentinel value which tells us if we passed all points
        unsigned char outMask;                      // one particular comparison value of if our wrong class point is in bounds of four attributes 
        unsigned char someAttributeOutside = 0;     // sentinel value for each particular point at an iteration.

        // Check all wrong class data points for falling into new bounds
        for (int point = 0; point < numPoints; point++) {
            
            // get our correct point for our points4 pointer
            points4Pointer = (float4*)&points[point * numAttributes];
            someAttributeOutside = 0;
            // loop through one point and check until we find an attribute that is out of bounds
            // i steps by one because we are just using float 4's therefore we just run it numAttributes / 4 times. or until we have found an attribute outside.
            for (int i = 0; i < numAttributes / 4 && !someAttributeOutside; i++) {
                outMask = 0; 
                // Load four attributes and their min/max bounds
                fourAttributes = points4Pointer[i];
                fourMinFloats = thisBlockCombinedMins4[i];
                fourMaxFloats = thisBlockCombinedMaxes4[i];

                // Bitmask to track out-of-bounds attributes
                // any of these true is going to result in someAttributeOutside not equaling 0. 
                outMask |= (fourAttributes.x < fourMinFloats.x) << 0;
                outMask |= (fourAttributes.x > fourMaxFloats.x) << 1;
                outMask |= (fourAttributes.y < fourMinFloats.y) << 2;
                outMask |= (fourAttributes.y > fourMaxFloats.y) << 3;
                outMask |= (fourAttributes.z < fourMinFloats.z) << 4;
                outMask |= (fourAttributes.z > fourMaxFloats.z) << 5;
                outMask |= (fourAttributes.w < fourMinFloats.w) << 6;
                outMask |= (fourAttributes.w > fourMaxFloats.w) << 7;

                // if any of those don't give us 0, then we have an attribute that is out of bounds.
                someAttributeOutside |= outMask;
            }

            // If all attributes are within bounds, merging is not possible
            // so if this is 0, we have passed. if not, we have failed.
            if (!someAttributeOutside) {
                allPassed = 0;
                break;
            }
        }

        // if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
        if (allPassed){
            // copy the combined mins and maxes into the original array
            for (int i = 0; i < numAttributes / 4; i++){
                // copy the mins and maxes over
                hyperBlockMins4[i] = thisBlockCombinedMins4[i];
                hyperBlockMaxes4[i] = thisBlockCombinedMaxes4[i];
            }
            // set the flag to -1 for seedBlock, meaning he is garbage.
            deleteFlags[seedBlock] = -1;
            mergable[threadID] = 1;
        }
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

        thisThreadPoint = &dataPointsArray[threadID * numAttributes];

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

void mergerHyperBlocksWrapper(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float *opposingPoints,float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable, int gridSize, int blockSize, int sharedMemSize, float* combinedMins, float* combinedMaxes){
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
			mergable,
      		combinedMins,
      		combinedMaxes
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

