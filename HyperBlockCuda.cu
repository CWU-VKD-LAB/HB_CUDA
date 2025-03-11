#include "HyperBlockCuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------

#define COMPARE_FLOAT4(a, b) ( (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z) && ((a).w == (b).w)) ? 0 : 1 )
__global__ void mergerHyperBlocks(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float* __restrict__ points, float *hyperBlockMins, float *hyperBlockMaxes, int* deleteFlags, int* mergable, float* combinedMins, float* combinedMaxes){
    
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // get our seed block.
    const int seedBlock = readSeedQueue[seedIndex];

    // so that we don't always have to do this division a bunch of times
    const int numAttributesAsFours = numAttributes / 4;

  	// Put the seed block attributes in shared memory using float 4's
    // a float 4 is literally just a struct of four floats. it is faster for memory reading and writing.
    extern __shared__ float seedBlockAttributes[];
    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];

    // seed block as a float 4 
    float4 *seedBlockMins4 = (float4 *)seedBlockMins;
    float4 *seedBlockMaxes4 = (float4 *)seedBlockMaxes;

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

        // our local float4's which will actually be holding seed mins and maxes
        float4 fourSeedMins;
        float4 fourSeedMaxes;
        
        // two for our block, and the two above are for the seed block
        float4 fourOurMins;
        float4 fourOurMaxes;

        // we take the min and max of each attribute and put them in here to write to combined mins and maxes.
        // the reason this is like this, is so that if we take all of the bounds from either block we already know that it will be valid and don't need to check the entire dataset.
        float4 fourCombinedMins;
        float4 fourCombinedMaxes;
            
        // get our float4 pointers to point at the combined mins and maxes for this block.
        float4 *thisBlockCombinedMaxes4 = (float4*)&combinedMaxes[threadID * numAttributes];
        float4 *thisBlockCombinedMins4 = (float4*)&combinedMins[threadID * numAttributes];

        // float4 pointer for our points. allows us to grab 4 floats at a time out of global memory
        float4 *hyperBlockMins4 = (float4*) &hyperBlockMins[threadID * numAttributes];
        float4 *hyperBlockMaxes4 = (float4*) &hyperBlockMaxes[threadID * numAttributes];

        // first we build our combined list.

        // these give us an important early out case. if we are taking an exact copy of either block, meaning that one 
        // block is totally eating the other block, don't need to check that block anymore. obviously it is going to be valid since it is a copy of a valid block.
        char usedSeedBlock = 0;
        char usedOurBlock = 0; 

        for (int i = 0; i < numAttributesAsFours; i++){
            
            // get our seed mins and maxes
            fourSeedMins = seedBlockMins4[i];
            fourSeedMaxes = seedBlockMaxes4[i];

            // get our own hyperblock mins and maxes
            fourOurMins = hyperBlockMins4[i];
            fourOurMaxes = hyperBlockMaxes4[i];

            // update our four mins and maxes compared to seed block. that is, take the min of the two and the max of the two HBs bounds.
            fourCombinedMins.w = fminf(fourSeedMins.w, fourOurMins.w);
            fourCombinedMins.x = fminf(fourSeedMins.x, fourOurMins.x);
            fourCombinedMins.y = fminf(fourSeedMins.y, fourOurMins.y);
            fourCombinedMins.z = fminf(fourSeedMins.z, fourOurMins.z);
            
            fourCombinedMaxes.w = fmaxf(fourSeedMaxes.w, fourOurMaxes.w);
            fourCombinedMaxes.x = fmaxf(fourSeedMaxes.x, fourOurMaxes.x);
            fourCombinedMaxes.y = fmaxf(fourSeedMaxes.y, fourOurMaxes.y);
            fourCombinedMaxes.z = fmaxf(fourSeedMaxes.z, fourOurMaxes.z);
            
            // now if they are not exactly the same one way or the other we can set the flag which tells us to check the dataset.
            
            // Check for discrepancies.
            // If the combined min for any component doesn't equal the seed's corresponding value,
            // then we know the seed value was not used for that bound. This means of course that we used OURS.
            // this is helpful because if EITHER FLAG IS FALSE AFTER THIS WE DON'T CHECK THE DATASET.
            // THIS IS BECAUSE BOTH SEEDBLOCK AND OUR HYPERBLOCK ARE VALID TO BEGIN WITH, SO TAKING AN EXACT COPY OF COURSE IS STILL VALID! SAVES TIME!
            usedOurBlock |= COMPARE_FLOAT4(fourCombinedMins, fourOurMins);
            usedOurBlock |= COMPARE_FLOAT4(fourCombinedMaxes, fourOurMaxes);

            usedSeedBlock |= COMPARE_FLOAT4(fourCombinedMins, fourSeedMins);
            usedSeedBlock |= COMPARE_FLOAT4(fourCombinedMaxes, fourSeedMaxes);

            thisBlockCombinedMins4[i] = fourCombinedMins;
            thisBlockCombinedMaxes4[i] = fourCombinedMaxes;
        }

        
        char allPassed = 1; // sentinel value which tells us if we passed all points

        // if we did use both blocks, so we made a new set of bounds, we now have to check the dataset.
        if (usedOurBlock && usedSeedBlock){
            
            // Check all opposing class data points for falling into new bounds
            unsigned char outMask;                      // one particular comparison value of if our wrong class point is in bounds of four attributes 
            unsigned char someAttributeOutside = 0;     // sentinel value for each particular point at an iteration.
            float4 fourAttributes;                      // for the checking attributes. 

            for (int point = 0; point < numPoints; point++) {
                
                // get our correct point for our points4 pointer
                points4Pointer = (float4*)&points[point * numAttributes];
                someAttributeOutside = 0;
                // loop through one point and check until we find an attribute that is out of bounds
                // i steps by one because we are just using float 4's therefore we just run it numAttributes / 4 times. or until we have found an attribute outside.
                for (int i = 0; i < numAttributesAsFours && !someAttributeOutside; i++) {
                    
                    outMask = 0; 
                    // Load four attributes and their min/max bounds
                    fourAttributes = points4Pointer[i];
                    fourOurMins = thisBlockCombinedMins4[i];
                    fourOurMaxes = thisBlockCombinedMaxes4[i];

                    // Bitmask to track out-of-bounds attributes
                    // any of these true is going to result in someAttributeOutside not equaling 0. 
                    outMask |= (fourAttributes.x < fourOurMins.x) << 0;
                    outMask |= (fourAttributes.x > fourOurMaxes.x) << 1;
                    outMask |= (fourAttributes.y < fourOurMins.y) << 2;
                    outMask |= (fourAttributes.y > fourOurMaxes.y) << 3;
                    outMask |= (fourAttributes.z < fourOurMins.z) << 4;
                    outMask |= (fourAttributes.z > fourOurMaxes.z) << 5;
                    outMask |= (fourAttributes.w < fourOurMins.w) << 6;
                    outMask |= (fourAttributes.w > fourOurMaxes.w) << 7;

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
        }

        // if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
        if (allPassed){
            // copy the combined mins and maxes into the original array
            for (int i = 0; i < numAttributesAsFours; i++){
                // copy the mins and maxes over
                hyperBlockMins4[i] = thisBlockCombinedMins4[i];
                hyperBlockMaxes4[i] = thisBlockCombinedMaxes4[i];
            }
            // set the flag to -1 for seedBlock, meaning he is garbage.
            deleteFlags[seedBlock] = -1;

            // set our own flag that we DID merge.
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

/*
 * Wrapper functions for the removing useless attributes.
 * */

void removeUselessAttributesWrapper(float* mins, float* maxes, const int* intervalCounts, const int minMaxLen, const int* blockEdges, const int numBlocks, const int* blockClasses, char* attrRemoveFlags, const int fieldLen, const float* dataset, const int numPoints, const int* classBorder, const int numClasses, const int *attributeOrder, int gridSize, int blockSize){
   removeUselessAttributes<<<gridSize, blockSize>>>(mins, maxes, intervalCounts, minMaxLen, blockEdges, numBlocks, blockClasses, attrRemoveFlags, fieldLen, dataset, numPoints, classBorder, numClasses, attributeOrder);
}