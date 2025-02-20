#include "HyperBlockCuda.cuh"
#include <cuda_runtime.h>

// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------
#define min(a, b) (a > b)? b : a
#define max(a, b) (a > b)? a : b
__global__ void mergerHyperBlocks(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float *opposingPoints, float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable){

    // get our block index. which block are we on?
    const int blockIndex = blockIdx.x;

    // get our local ID. which thread of the block are we?
    const int localID = threadIdx.x;

    // get our seed block.
    const int seedBlock = readSeedQueue[seedIndex];

    // every block tries to do this, so that we can make sure and not start executing until the flag is set.
    // it only updates once obviously, but it makes all the other blocks wait until someone has set that value.
    if (localID == 0){
        atomicMin(&deleteFlags[seedBlock], -9);
        atomicMin(&mergable[seedBlock], 0);     // shouldn't probably matter for the seedblock but we do it for the love of the game.
    }
    // all the threads of a block are going to deal with their flag to determine our early out condition.
    __shared__ int blockMergable;

    // our shared memory will store the bounds of a hyperblock. each cuda block will run through one block at a time. with an offset of gridDim.x * numAttributes
    extern __shared__ float hyperBlockAttributes[];
    // copy into our attributes the combined mins and maxes of seed block and our current block.
    float *localBlockMins = &hyperBlockAttributes[0];
    float *localBlockMaxes = &hyperBlockAttributes[numAttributes];

    __syncthreads();
    // iterate through all the blocks, with a stride of numBlocks of CUDA that we have.
    for(int i = blockIndex; i < numBlocks; i+= gridDim.x){

        // if the block has already been a seed block, we aren't going to do our merging business with it.
        // every thread skips this so it's ok. the sync isn't a problem. 
        if (deleteFlags[i] < 0 || i == seedBlock){
            continue;
        }

        // set our flag to 1, since we are passing until someone fails.
        if (localID == 0){
            blockMergable = 1;
        }

        // copy the mins and maxes of the seed block merged with our current block into our shared memory
        for(int att = localID; att < numAttributes; att += blockDim.x){
            localBlockMins[att] = min(hyperBlockMins[seedBlock * numAttributes + att], hyperBlockMins[i * numAttributes + att]);
            localBlockMaxes[att] = max(hyperBlockMaxes[seedBlock * numAttributes + att], hyperBlockMaxes[i * numAttributes + att]);
        }

        // sync so we don't start early.
        __syncthreads();

        // now we need to check if the current block is mergable with the seed block.
        // to do this we simply check all the datapoints. before a thread starts a datapoint, we are going to check the shared flag and make sure it's worth our time.
        // if any threads finds unmergable, we set the flag, and wait for everyone else.
        for(int pointIndex = localID; pointIndex < numPoints && blockMergable; pointIndex += blockDim.x){
            char someAttributeOutside = 0;
            for(int att = 0; att < numAttributes; att++){
                if(opposingPoints[pointIndex * numAttributes + att] > localBlockMaxes[att] || opposingPoints[pointIndex * numAttributes + att] < localBlockMins[att]){
                    someAttributeOutside = 1;
                    break;
                }
            }
            // if every single attribute was inside, we have failed. since these are all opposing points.
            if(!someAttributeOutside){
                blockMergable = 0;
                break;
            }
        }
        // wait for everyone else to finish that block.
        __syncthreads();

        // if it was mergable, we copy the mins and maxes into the original array.
        if (blockMergable){
            for(int att = localID; att < numAttributes; att += blockDim.x){
                hyperBlockMins[i * numAttributes + att] = localBlockMins[att];
                hyperBlockMaxes[i * numAttributes + att] = localBlockMaxes[att];
            }

            // now we update the delete flag for the seed block to show that it is trash.
            // -1 means it got merged, so we don't need to copy it back. -9 is for if it never merged, so it ends up living on.
            if (localID == 0){
                atomicMax(&deleteFlags[seedBlock], -1);
            }
        }
        // if we're the first thread, we need to write the delete flags properly.
        if (localID == 0){
            mergable[i] = blockMergable;
        }
        // must sync here so that we don't accidentally pick a seed block while we are updating the delete flags queue potentially.
        __syncthreads();
    } // end of checking one single block.
}

__global__ void rearrangeSeedQueue(int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;
    // now we are just going to loop through the seed queue and compute each blocks new position in the queue.
    for(int i = threadID; i < numBlocks; i += globalThreadCount){
        // if the block is dead, we just copy it over. it is dead if we have already used it as a seed block.
        if (deleteFlags[readSeedQueue[i]] < 0){
            writeSeedQueue[i] = readSeedQueue[i];
            continue;
        }
        // if we didn't merge, we are just going to iterate through and our new index is just the amount of numbers <= 0 to our LEFT.
        if (mergable[readSeedQueue[i]] == 0){
            int newIndex = 0;
            for(int j = 0; j < i; j++){
                if (mergable[readSeedQueue[j]] == 0){
                    newIndex++;
                }
            }
            writeSeedQueue[newIndex] = readSeedQueue[i];
        }
        else{
            int count = 0;
            // if we did merge our new index is the amount of 1's (flags that we merged) to our LEFT, SUBTRACTED FROM N - 1.
            // this is because if you were at the front and merged we want you to go to the back.
            for(int j = 0; j < i; j++){
                if (mergable[readSeedQueue[j]] == 1){
                    count++;
                }
            }
            writeSeedQueue[numBlocks - 1 - count] = readSeedQueue[i];
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
__global__ void assignPointsToBlocks(const float *dataPointsArray, const int numAttributes, const int numPoints, float *blockMins, float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    const float *thisThreadPoint = &dataPointsArray[threadID * numAttributes];

    int *dataPointBlock;       // pointer to the slot in the numPoints long array that tells us which block this point goes into.
    float *startOfBlockMins;    // pointer to the start of the mins of the current block.
    float *startOfBlockMaxes;  // pointer to the start of the maxes of the current block.
    float *endOfBlock;         // pointer to the end of the current block.
    int currentBlock;          // the current block we are on.
    int nextBlock;             // the next block. useful because blocks are varying sizes.

    for(int i = threadID; i < numPoints; i += globalThreadCount){

        currentBlock = 0;
        nextBlock = 1;

        // set out pointer to where we assign this point to a block.
        dataPointBlock = &dataPointBlocks[i];
        *dataPointBlock = -1;

        thisThreadPoint = &dataPointsArray[i * numAttributes];

        // now we iterate through all the blocks. checking which block this point falls into first.
        while (currentBlock < numBlocks){

            // set up our start of mins and maxes.
            startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
            endOfBlock = &blockMins[blockEdges[nextBlock]];
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
/*

Ryans algo. not fixing since we not using yet.


__global__ void sumPointsPerBlock(const int *dataPointBlocks, const int numPoints, int *numPointsInBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    for(int i = threadID; i < numPoints; i += globalThreadCount){
        atomicInc(&numPointsInBlocks[dataPointBlocks[i]]);
    }
}
__global__ void findBetterBlocks(int *dataPointsBlocks, const int numPoints, const int numBlocks, const int numAttributes, const float *points, const int *blockEdges, float *hyperBlockMins, float *hyperBlockMaxes, int *numPointsInBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    float *startOfBlockMins;
    float *startOfBlockMaxes;
    float *endOfBlock;
    int *dataPointBlock;
    for(int i = threadID; i < numPoints; i += globalThreadCount){

        int currentBlock = dataPointsBlocks[i];
        int nextBlock = currentBlock + 1;
        dataPointBlock = &dataPointsBlocks[i];

        // we have a coverage issue if this happens.
        if (currentBlock == -1){
            continue;
        }

        float *thisThreadPoint = &points[i * numAttributes];

        // largest block size is our current block we are assigned to for this point
        int largestBlockSize = numPointsInBlocks[currentBlock];

        // now we iterate through and finally assign our point to the most populous block we find that we fit into.
        while (currentBlock < numBlocks){

            // now, we iterate through all the blocks after the one we chose, and if we find a bigger one we fit into, we go into that one.
            startOfBlockMins = &hyperBlockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &hyperBlockMaxes[blockEdges[currentBlock]];
            endOfBlock = &blockMins[blockEdges[nextBlock]];

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
*/

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
    return;
}

void rearrangeSeedQueueWrapper(int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks, int gridSize, int blockSize){
    rearrangeSeedQueue<<<gridSize, blockSize>>>(readSeedQueue, writeSeedQueue, deleteFlags, mergable, numBlocks);
	return;
}
void resetMergableFlagsWrapper(int *mergableFlags, const int numBlocks, int gridSize, int blockSize){
	resetMergableFlags<<<gridSize, blockSize>>>(mergableFlags, numBlocks);
  	return;
}

/**
* IMPLEMENT ME NOT YET DONE!!!!!!!!!!
*/
void assignPointsToBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints,const float *blockMins, const float *blockMaxes, const int *blockEdges,const int numBlocks, int *dataPointBlocks, int gridSize, int blockSize){
	return;
}