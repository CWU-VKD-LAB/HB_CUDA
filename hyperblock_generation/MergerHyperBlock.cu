#include "MergerHyperBlock.cuh"
// Define a macro to compare float4's for equality.
#define COMPARE_FLOAT4(a, b) ( (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z) && ((a).w == (b).w)) ? 0 : 1 )
// Define simple min/max macros for scalar floats.
#define min_f(a, b) ((a) > (b) ? (b) : (a))
#define max_f(a, b) ((a) > (b) ? (a) : (b))

// Hybrid kernel that uses block-level cooperation but vectorizes with float4's.
// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS LIKE WE USE IN DV2.0.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------

// the monster. this is one of the heaviest parts of the whole program. it is very heavily optimized, so hard to read.
// a block of threads work together to check HBs, striding by number of total cuda blocks. basically it works like this.
// we take a seed block out of the queue. each HB besides that block, and all ones we have already used, try to eat the seed block.
// to eat it, you just take the min of the bounds of the seed block and the block we are merging, and the max.
// so we take whoever's x1 min is lower, and whichever x1 max is higher, and so on or all attributes.
// then we check THE ENTIRE DATASET to determine if other class points are inside our new bounds. if so, we don't update the bounds of our HB.
// if there were no points inside, we can then do the merge, and we mark the seedblock as merged to, so it can die, and then we mark ours as merged.
// if even one point merges to seed block, we know that we can delete it, since that block will be entirely inside of another one.
__global__ void mergerHyperBlocks(
    const int seedIndex, 
    int *readSeedQueue, 
    const int numBlocks, 
    const int numAttributes,   // padded to be a multiple of 4 in calling function (merger_cuda() in interval_hyperblock.cu).
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

    // we use these stupid looking float4's all over. These are literally just a struct with 4 floats.
    // they are more efficient because we are able to read more memory at a time.
    // Treat first half as an array of float4's for the mins...
    float4 *localCombinedMins = (float4*) sharedMem;
    // ...and the second half for the maxes.
    float4 *localCombinedMaxes = (float4*) (sharedMem + numAttributes);

    // Reinterpret opposingPoints as a SoA float4 array for coalesced loads.
    const float4* opp4 = reinterpret_cast<const float4*>(opposingPoints);

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

        // (a) build combined bounds into shared memory
        for (int i = localID; i < numAttributesAsFours; i += blockDim.x) {
            // Load seed block's bounds.
            float4 seedMin = *((float4*)&hyperBlockMins[seedBlock * numAttributes] + i);
            float4 seedMax = *((float4*)&hyperBlockMaxes[seedBlock * numAttributes] + i);
            // Load candidate block's bounds.
            float4 candMin = *((float4*)&hyperBlockMins[candidate * numAttributes] + i);
            float4 candMax = *((float4*)&hyperBlockMaxes[candidate * numAttributes] + i);

            // Compute the combined bounds (min and max).
            float4 combinedMin = {
                min_f(seedMin.x, candMin.x),
                min_f(seedMin.y, candMin.y),
                min_f(seedMin.z, candMin.z),
                min_f(seedMin.w, candMin.w)
            };
            float4 combinedMax = {
                max_f(seedMax.x, candMax.x),
                max_f(seedMax.y, candMax.y),
                max_f(seedMax.z, candMax.z),
                max_f(seedMax.w, candMax.w)
            };

            // Store in shared memory.
            localCombinedMins[i] = combinedMin;
            localCombinedMaxes[i] = combinedMax;

            // Did we actually use any seed values? just a quick check, if we basically just used one block or the other, we already know it's valid. checking if we updated the bounds is all we are doing.
            if (COMPARE_FLOAT4(combinedMin, seedMin) || COMPARE_FLOAT4(combinedMax, seedMax)) {
                atomicOr(&usedSeedBlock, 1);
            }
            // Did we actually use any candidate values?
            if (COMPARE_FLOAT4(combinedMin, candMin) || COMPARE_FLOAT4(combinedMax, candMax)) {
                atomicOr(&usedCandidateBlock, 1);
            }
        }
        __syncthreads();

        // Determine whether we really need to scan the dataset.
        int needDatasetCheck = (usedSeedBlock && usedCandidateBlock);

        // check the entire dataset now. we have transposed our point matrix, so that threads are working on adjacent elements.
        // this is a small change which gives big speedup. having threads 0 and 1 reading elements 0 and 1, instead of 0 and (numAttributes + 1) is a huge speed gain.
        for (int pointIndex = localID; pointIndex < numPoints && blockMergable && needDatasetCheck; pointIndex += blockDim.x) {
            bool pointOutside = false;
            for (int i = 0; i < numAttributesAsFours; i++) {
                // coalesced SoA load:
                float4 pointVal = opp4[ i * numPoints + pointIndex ];
                float4 combMin  = localCombinedMins[i];
                float4 combMax  = localCombinedMaxes[i];
                if (pointVal.x < combMin.x || pointVal.x > combMax.x ||
                    pointVal.y < combMin.y || pointVal.y > combMax.y ||
                    pointVal.z < combMin.z || pointVal.z > combMax.z ||
                    pointVal.w < combMin.w || pointVal.w > combMax.w) {
                    pointOutside = true;
                    break;
                }
            }
            if (!pointOutside) {
                blockMergable = 0;
                break;
            }
        }
        __syncthreads();

        // (c) write back result for this candidate
        if (blockMergable) {
            // if the block was mergeable, we are just going to save it's new combined bounds
            for (int i = localID; i < numAttributesAsFours; i += blockDim.x) {
                *((float4*)&hyperBlockMins[candidate * numAttributes] + i) = localCombinedMins[i];
                *((float4*)&hyperBlockMaxes[candidate * numAttributes] + i) = localCombinedMaxes[i];
            }

            // now thread 0 can mark the seedblock as deleteable, and we are merged.
            if (localID == 0) {
                atomicMax(&deleteFlags[seedBlock], -1);
                mergable[candidate] = 1;
            }
        } else if (localID == 0) {
            mergable[candidate] = 0;
        }
        __syncthreads();
    }
}


// rearrange seed queue is important. we run this right after the merging has happened. so we run the merge for one particular seed block. then we rearrange.
// the HBs which merged go to the back, and the ones which didn't slide to the front. notice how if block deadSeedNum + 1 merged, it would actually end up at the back. that's
// intentional, because it does much better.
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

// reset the mergeable flags for the next round.
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
// all this function does is take each point, and assign it to the first HB it falls into.
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

// almost the same as the assignPointsToBlocks, but this time, instead of searching for the first HB it fits inside, we take the BIGGEST HB it fits insde
// we use the count for this part, an HB with a bigger count, we assign there.
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
 * takes in all the data, and our goofy flattened HBs array.
 * the core logic is this, it determine is we skipped an attribute, by allowing it's bound to be 0-1.0, or the entire dataset, if we would let any wrong points in.
 * if we would not, we can safely remove that attribute.
 * takes tons of pointers, because we have to track all the data, the mins and maxes, how many intervals in each HB (to accomodate disjunctions), where an HB starts, what class, and then finally the attributes we can remove and so on.
 * THIS IS AN IMPORTANT FUNCTION TO OPTIMIZE. IN THE FUTURE, MAYBE TAKE THIS, AND MAKE AN INITIAL SIMPLER VERSION WHICH IS ABLE TO OPERATE ON A REGULAR HB ONLY.
 * FOR EXAMPLE: RIGHT NOW THIS IS DISJUNCTION FRIENDLY, BUT THAT MEANS WE DO ALL THIS GARBAGE WITH COUNTING INTERVALS AND BLOCK START AND SO ON.
 * IF YOU JUST MADE A VERSION WHICH ONLY WORKS ON REGULAR HBS, IT WOULD BE WAY MORE EFFICIENT. YOU MIGHT BE ABLE TO SIMPLIFY FULL_MNIST EASILY. (especially on 10 lab computers like we did.)
 * it could be hyper optimized in the future as we did with the merging. but this version runs "good enough" for anything but mnist set.
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

// same as above version, but this one assumes that the HBs are only one rule per attribute. so we can use a lot more efficient methods.
// makes changes to block bounds directly in the kernel, no need for the dumb flags.
__global__ void removeUselessAttributesNoDisjunctions(
    float *mins,
    float *maxes,
    const int numBlocks,
    const int FIELD_LENGTH,
    const int *blockClasses,
    const float *__restrict__ dataset,
    const int numPoints,
    const int *classBorder,
    const int *attributeOrder)
{
    int blockID  = blockIdx.x;
    int threadID = threadIdx.x;

    // set up our pointers to our shared memory
    extern __shared__ float hyperBlockBounds[];
    float *blockMins  = hyperBlockBounds;
    float *blockMaxes = hyperBlockBounds + FIELD_LENGTH;

    // shared scalars
    __shared__ int classNum, classStart, classEnd;
    __shared__ int attributeToRemove;               // current candidate
    volatile __shared__ int attrRemovableWarp;      // 1 = still removable (volatile so every load hits shared mem)

    //-------------------------------------------------------------------
    // Handle the blocks assigned to this thread-block
    //-------------------------------------------------------------------
    for (int b = blockID; b < numBlocks; b += gridDim.x) {

        /* Load bounds for this HB into shared memory ------------------ */
        for (int i = threadID; i < FIELD_LENGTH; i += blockDim.x) {
            blockMins [i] = mins [b * FIELD_LENGTH + i];
            blockMaxes[i] = maxes[b * FIELD_LENGTH + i];
        }

        // initialize our flags and whatnot
        if (threadID == 0) {
            classNum   = blockClasses[b];
            classStart = classBorder[classNum];
            classEnd   = classBorder[classNum + 1];
        }
        __syncthreads();

        /* Test every attribute for removability ----------------------- */
        for (int attrIdx = 0; attrIdx < FIELD_LENGTH; ++attrIdx) {

            // thread 0 resets stuff.
            if (threadID == 0) {
                attributeToRemove = attributeOrder[classNum * FIELD_LENGTH + attrIdx];
                attrRemovableWarp = 1;               // reset block-wide flag
            }
            __syncthreads();

            //---------------------------- point loop (warp-early-exit) --
            int lane = threadID & 31; // warp lane ID

            for (int p = threadID; p < numPoints; p += blockDim.x) {

                // Skip own-class rows
                if (p >= classStart && p < classEnd) continue;

                // bail out early if another thread already vetoed removal
                if (attrRemovableWarp == 0) break;

                bool inBounds = true;
                for (int a = 0; a < FIELD_LENGTH; ++a) {
                    // skip the attribute we are “removing”
                    if (a == attributeToRemove) continue;

                    float v = dataset[a * numPoints + p];
                    if (v < blockMins[a] || v > blockMaxes[a]) {
                        inBounds = false;
                        break;
                    }
                }

                // Found a foreign-class point inside means that our attribute NOT removable
                if (inBounds) {
                    attrRemovableWarp = 0;
                    break; // other lanes in this warp will see it soon
                }
            }

            // warp-level ballot: if any lane in the warp has attrRemovableWarp == 0
            // thread 0 of the warp writes 0; otherwise leaves it as-is.
            unsigned int vote = __ballot_sync(0xFFFFFFFF, attrRemovableWarp);
            if ((lane == 0) && (vote != 0xFFFFFFFF)) attrRemovableWarp = 0;

            __syncthreads();

            //-------------------------- apply the change ---------------
            if (threadID == 0 && attrRemovableWarp) {
                blockMins [attributeToRemove] = 0.0f;
                blockMaxes[attributeToRemove] = 1.0f;
            }
            __syncthreads();
        }

        // now we just copy our bounds back into global memory from shared. 
        for (int i = threadID; i < FIELD_LENGTH; i += blockDim.x) {
            mins [b * FIELD_LENGTH + i] = blockMins [i];
            maxes[b * FIELD_LENGTH + i] = blockMaxes[i];
        }
        __syncthreads();
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