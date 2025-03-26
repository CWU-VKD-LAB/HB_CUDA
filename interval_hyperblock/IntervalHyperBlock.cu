//
// Created by asnyd on 3/20/2025.
//

#include "IntervalHyperBlock.h"
#include <vector>
#include <future>
#include <algorithm>
#include <iostream>
#include <ostream>
#include "Interval.h"
#include "DataAttr.h"
#include "../hyperblock_generation/MergerHyperBlock.cuh"

using namespace std;

#define USED true

#define EPSILON 0.00001
// comparing float helper
static inline bool closeEnough(float a, float b) {
    return abs(a - b) < EPSILON;
}

// helper function. checks if there are any of the exact same value that our current start of an interval has, of the wrong class behind it
static inline bool checkBackwards(vector<DataATTR> &dataByAttribute, int currentStart) {
    //------------------------------------------------------------------
    // 2) BACKWARD CHECK for mismatch among same-value items
    //------------------------------------------------------------------
    int startClass = dataByAttribute[currentStart].classNum;
    float startVal = dataByAttribute[currentStart].value;

    int backIdx = currentStart - 1;
    while (backIdx >= 0) {
        // If value changes, the same-value “block” behind has ended, so stop.
        if (!closeEnough(dataByAttribute[backIdx].value, startVal)) {
            return true;
        }
        // If we find a different class in that same-value region, mismatch.
        if (dataByAttribute[backIdx].classNum != startClass) {
            return false;
        }
        backIdx--;
    }
    // if we go all the way to the end with no issues, return true
    return true;
}

Interval IntervalHyperBlock::longestInterval(std::vector<DataATTR> &dataByAttribute, int attribute)
{
    Interval bestInterval(-1, -1, -1, attribute, -1);
    int n = static_cast<int>(dataByAttribute.size());
    int currentStart = 0;

    while (currentStart < n) {
        // Skip "used" items to find a valid start:
        while (currentStart < n && dataByAttribute[currentStart].used) {
            currentStart++;
        }

        if (currentStart >= n) {
            break; // all used or out of range
        }

        //------------------------------------------------------------------
        // BACKWARD CHECK for mismatch among same-value items
        //------------------------------------------------------------------
        int startClass = dataByAttribute[currentStart].classNum;

        // if our back check failed, that means there is a matching value behind us, from the wrong class.
        // this means we have to just move on as an interval of ONE no matter what.
        if (!checkBackwards(dataByAttribute, currentStart)) {
            if (1 > bestInterval.size) {
                bestInterval.size         = 1;
                bestInterval.start        = currentStart;
                bestInterval.end          = currentStart;
                bestInterval.dominantClass = startClass;
            }
            // move one and carry on
            currentStart++;
            continue;
        }

        //------------------------------------------------------------------
        // 3) FORWARD EXTENSION to find the longest run
        //------------------------------------------------------------------
        int currentEnd = currentStart; // we'll move this as far as we can
        while (currentEnd < n) {
            if (dataByAttribute[currentEnd].classNum != startClass) {
                break;
            }
            currentEnd++;
        }

        int length = currentEnd - currentStart;
        if (length > bestInterval.size) {
            bestInterval.size = length;
            bestInterval.start = currentStart;
            bestInterval.end = currentEnd - 1;
            bestInterval.dominantClass = startClass;
        }

        //------------------------------------------------------------------
        // Move on to the next interval start
        //------------------------------------------------------------------
        currentStart = currentEnd;
    }

    // now, one final check. we are going to simply trim off values at the end, if they are matching with exact same value as a wrong class point
    int finalEnd = bestInterval.end;

    // if we won't go out of bounds by checking the next one
    if (finalEnd < n - 1) {
        // get our next value
        float neighborVal = dataByAttribute[bestInterval.end + 1].value;

        // while there is a match between the next value and where we are trying to end, we have to trim off our end guy.
        // this prevents us from making an interval which includes a value which would be shared between our class and another class.
        while (finalEnd > bestInterval.start && closeEnough(neighborVal, dataByAttribute[finalEnd].value)) {
            finalEnd--;
        }
        bestInterval.end = finalEnd;
    }
    return bestInterval;
}

// REMAINING DATA PASSED AS A COPY INTENTIONALLY! WE TAKE TRAINING DATA AS A COPY, SO WE CAN REMOVE WITHOUT RUINING ORIGINAL DATA
void IntervalHyperBlock::intervalHyper(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> remainingData, vector<HyperBlock> &hyperBlocks) {

    // sort the input dataAttr's in each column by the value
    for (int i = 0; i < remainingData.size(); i++) {
        sort(remainingData[i].begin(), remainingData[i].end(),
             [](const DataATTR &a, const DataATTR &b) {
                 return a.value < b.value;
        });
    }

    while (true) {

        // Launch our intervals asynchronously; get the intervals back and pick the biggest.
        vector<future<Interval>> intervals;
        Interval best(-1, -1, -1, -1, -1);

        // Search each attribute
        for (int i = 0; i < remainingData.size(); i++) {
            intervals.emplace_back(async(launch::async, longestInterval, ref(remainingData[i]), i));
        }

        // Wait for results then find largest interval
        for (auto &future1 : intervals) {
            Interval intr = future1.get();
            if (intr.size >= 1 && intr.size > best.size) {
                best = intr;  // copy entire interval
            }
        }

        // Build a list of removed points based on the best interval.
        // (We assume that best.start and best.end are valid indices in remainingData[attr].)
        vector<pair<int, int>> usedIDs;

        // if we had a valid interval, we have to do all this business
        // if there was not we are obviously just done.
        if (best.size >= 1) {
            for (int i = best.start; i <= best.end; i++) {
                DataATTR d = remainingData[best.attribute][i];
                if (!d.used)
                    usedIDs.push_back({d.classNum, d.classIndex});
            }

            // Build the block of points from the real data.
            vector<vector<float>> pointsInThisBlock;
            pointsInThisBlock.reserve(best.size);  // reserve capacity to avoid extra copies
            for (int i = best.start; i <= best.end; i++) {

                // From the best attribute column, grab the identification for the point.
                DataATTR thisPoint = remainingData[best.attribute][i];

                if (thisPoint.used) {
                    continue;
                }

                int classNum = thisPoint.classNum;
                int classIndex = thisPoint.classIndex;

                // Get the actual point from the real data and add it.
                pointsInThisBlock.push_back(realData[classNum][classIndex]);
            }

            // Compute bounds for each attribute.
            vector<vector<float>> maxes(remainingData.size(), vector<float>(1, -numeric_limits<float>::infinity()));
            vector<vector<float>> mins(remainingData.size(), vector<float>(1, numeric_limits<float>::infinity()));

            for (int point = 0; point < pointsInThisBlock.size(); point++) {
                for (int att = 0; att < remainingData.size(); att++) {
                    maxes[att][0] = max(pointsInThisBlock[point][att], maxes[att][0]);
                    mins[att][0] = min(pointsInThisBlock[point][att], mins[att][0]);
                }
            }

            // make a block and throw it into the hyperblocks vector
            HyperBlock h(maxes, mins, best.dominantClass);
            hyperBlocks.push_back(h);

            // --- REMOVAL PHASE ---
            // Remove the points that were just used from each column in remainingData.
            // --- REMOVAL PHASE ---
            for (int att = 0; att < remainingData.size(); att++) {
                for (int dataAtt = 0; dataAtt < remainingData[att].size(); dataAtt++) {

                    // Make sure to read from the "att" column, *not* best.attribute
                    DataATTR d = remainingData[att][dataAtt];

                    // Check if this DataATTR matches any of the removed points.
                    for (auto &removed : usedIDs) {
                        if (d.classNum == removed.first && d.classIndex == removed.second) {
                            // Mark it used in the att-th column
                            remainingData[att][dataAtt].used = USED;
                            break;
                        }
                    }
                }
            }
        }
        else
            break;
    }
}

/**
 * Seperates data into seperate vecs by attribute
 */
vector<vector<DataATTR>> IntervalHyperBlock::separateByAttribute(vector<vector<vector<float>>>& data, int FIELD_LENGTH){
    vector<vector<DataATTR>> attributes;

    // Go through the attribute columns
    for(int k = 0; k < FIELD_LENGTH; k++){
        vector<DataATTR> tmpField;

        // Go through the classes
        for(int i = 0; i < data.size(); i++){
            // Go through the points
            for(int j = 0; j < data[i].size(); j++){
                tmpField.push_back(DataATTR(data[i][j][k], i, j, false));
            }
        }

        // Sort data by value then add
        sort(tmpField.begin(), tmpField.end(), [](const DataATTR& a, const DataATTR& b) {
            return a.value < b.value;
        });
        attributes.push_back(tmpField);
    }

    return attributes;
}

/***
 * This will sort the array based on the "best" columns values
 *
 * The columns themselves aren't moving, we are moving the points
 * based on the one columns values;
 */
void IntervalHyperBlock::sortByColumn(vector<vector<float>>& classData, int colIndex) {
    sort(classData.begin(), classData.end(), [colIndex](const vector<float>& a, const vector<float>& b) {
        return a[colIndex] < b[colIndex];
    });
}

void IntervalHyperBlock::generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyperBlocks, vector<int> &bestAttributes, int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS){

    // Get data to create hyperblocks
    vector<vector<DataATTR>> dataByAttribute = separateByAttribute(data, FIELD_LENGTH);

    // make our interval based blocks
    intervalHyper(data, dataByAttribute, hyperBlocks);
    
    try{
        merger_cuda(data, hyperBlocks, COMMAND_LINE_ARGS_CLASS);
        cout << "BlOCK GENERATION FINISHED! WE FOUND: " << hyperBlocks.size() << " BLOCKS" << endl;
    } catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
        cout << e.what() << endl;
    }
}

// Source
void IntervalHyperBlock::merger_cuda(const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS) {

    int NUM_CLASSES = allData.size();
    int FIELD_LENGTH = allData[0][0].size();

    cout << "Num classes " << NUM_CLASSES << endl;

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : allData) {
        numPoints += classData.size();
    }

    // Count blocks per class
    vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyperBlocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    vector<vector<HyperBlock>> inputBlocks(NUM_CLASSES);
    vector<vector<HyperBlock>> resultingBlocks(NUM_CLASSES);

    int PADDED_LENGTH = ((FIELD_LENGTH + 3) / 4) * 4;
    // Find best occupancy
    int sharedMemSize = 2 * PADDED_LENGTH * sizeof(float);
    int minGridSize, blockSize;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mergerHyperBlocks, sharedMemSize, 0);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaOccupancyMaxPotentialBlockSize: %s\n", cudaGetErrorString(err));
        exit(-1);
    }


    int temp = 0;
    int goToClass = NUM_CLASSES;
    if (COMMAND_LINE_ARGS_CLASS != -1){
         temp = COMMAND_LINE_ARGS_CLASS;
         goToClass = COMMAND_LINE_ARGS_CLASS + 1;
    }

    for (HyperBlock& hyperBlock : hyperBlocks) {
        // store this block in the slot which corresponds to it's class.
        inputBlocks[hyperBlock.classNum].push_back(hyperBlock);
    }

    for (int classN = temp; classN < goToClass; classN++) {

        // set our device based on class. this way even single threaded we use multiple GPUs
        // MORE MULTI GPU BUSINESS
        //cudaSetDevice(classN % deviceCount);

        int totalDataSetSizeFlat = numPoints * PADDED_LENGTH;

        // Compute grid size to cover all HBs. we already know our ideal block size from before.
        int gridSize = ((numBlocksOfEachClass[classN]) + blockSize - 1) / blockSize;

        #ifdef DEBUG
        cout << "Grid size: " << gridSize << endl;
        cout << "Block size: " << blockSize << endl;
        cout << "Shared memory size: " << sharedMemSize << endl;
        #endif

        int currentClassBlockLengthFlattened = inputBlocks[classN].size() * PADDED_LENGTH;

        // Allocate host memory
        vector<float> hyperBlockMinsC(currentClassBlockLengthFlattened);
        vector<float> hyperBlockMaxesC(currentClassBlockLengthFlattened);
        vector<int> deleteFlagsC(currentClassBlockLengthFlattened / PADDED_LENGTH);

        int nSize = allData[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

        // Fill hyperblock array
        for (int i = 0; i < inputBlocks[classN].size(); i++) {
            HyperBlock h = inputBlocks[classN][i];
            for (int j = 0; j < FIELD_LENGTH; j++) {
                hyperBlockMinsC[i * PADDED_LENGTH + j] = h.minimums[j][0];
                hyperBlockMaxesC[i * PADDED_LENGTH + j] = h.maximums[j][0];
            }
            for (int j = FIELD_LENGTH; j < PADDED_LENGTH; j++) {
                hyperBlockMinsC[i * PADDED_LENGTH + j] = -numeric_limits<float>::infinity();
                hyperBlockMaxesC[i * PADDED_LENGTH + j] = numeric_limits<float>::infinity();
            }
        }

        // prepare other class points
        int otherClassIndex = 0;
        for (int currentClass = 0; currentClass < allData.size(); currentClass++) {
            if (currentClass == classN) continue;

            for (const auto& point : allData[currentClass]) {
                for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                    pointsC[otherClassIndex++] = point[attr];
                }
                for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                    pointsC[otherClassIndex++] = -numeric_limits<float>::infinity();
                }
            }
        }

        // Allocate device memory
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, currentClassBlockLengthFlattened * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, currentClassBlockLengthFlattened * sizeof(float));
        cudaMalloc(&d_deleteFlags, (currentClassBlockLengthFlattened / PADDED_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (currentClassBlockLengthFlattened / PADDED_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = inputBlocks[classN].size();
        vector<int> seedQueue(numBlocks);
        for(int i = 0; i < numBlocks; i++){
            seedQueue[i] = i;
        }

        cudaMalloc(&d_mergable, numBlocks * sizeof(int));
        cudaMemset(d_mergable, 0, numBlocks * sizeof(int));
        cudaMalloc(&d_seedQueue, numBlocks * sizeof(int));
        cudaMalloc(&d_writeSeedQueue, numBlocks * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_hyperBlockMins, hyperBlockMinsC.data(), currentClassBlockLengthFlattened * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyperBlockMaxes, hyperBlockMaxesC.data(), currentClassBlockLengthFlattened * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsC.data(), pointsC.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seedQueue, seedQueue.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

        cout << "Launched a kernel for class: " << classN << endl;

        // funky wap to swap the readQueue and writeQueue
        int* queues[2] = {d_seedQueue, d_writeSeedQueue};
        for(int i = 0; i < numBlocks; i++){
            // swap between the two queues
            int* readQueue = queues[i & 1];
            int* writeQueue = queues[(i + 1) & 1];
            mergerHyperBlocksWrapper(
                i, 			// seednum
                readQueue,  // seedQueue
                numBlocks,  // number seed blocks
                PADDED_LENGTH,	// num attributes
                pointsC.size() / PADDED_LENGTH,	// num op class points
                d_points,						// op class points
                d_hyperBlockMins,				// mins
                d_hyperBlockMaxes,				// maxes
                d_deleteFlags,
                d_mergable,						// mergable flags
                gridSize,
                blockSize,
                sharedMemSize
            );
            cudaDeviceSynchronize();

            // Reorder the seedblock order
            rearrangeSeedQueueWrapper(i, readQueue, writeQueue, d_deleteFlags, d_mergable, numBlocks, gridSize, blockSize);
            cudaDeviceSynchronize();

            // Reset mergable flags
            resetMergableFlagsWrapper(d_mergable, numBlocks, gridSize, blockSize);
            cudaDeviceSynchronize();
        }

        // Copy results back
        cudaMemcpy(hyperBlockMinsC.data(), d_hyperBlockMins, currentClassBlockLengthFlattened * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hyperBlockMaxesC.data(), d_hyperBlockMaxes, currentClassBlockLengthFlattened * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(deleteFlagsC.data(), d_deleteFlags, deleteFlagsC.size() * sizeof(int), cudaMemcpyDeviceToHost);

        // Process results
        for (int i = 0; i < hyperBlockMinsC.size(); i += PADDED_LENGTH) {

            if (deleteFlagsC[i / PADDED_LENGTH] == -1) continue;  // -1 is a seed block which was merged to. so it doesn't need to be copied back.

            vector<vector<float>> blockMins(FIELD_LENGTH);
            vector<vector<float>> blockMaxes(FIELD_LENGTH);
            for (int j = 0; j < FIELD_LENGTH; j++) {
                blockMins[j].push_back(hyperBlockMinsC[i + j]);
                blockMaxes[j].push_back(hyperBlockMaxesC[i + j]);
            }
            HyperBlock hb(blockMaxes, blockMins, classN);
            resultingBlocks[classN].emplace_back(hb);
        }

        // Free device memory
        cudaFree(d_hyperBlockMins);
        cudaFree(d_hyperBlockMaxes);
        cudaFree(d_deleteFlags);
        cudaFree(d_points);
        cudaFree(d_mergable);
        cudaFree(d_seedQueue);
        cudaFree(d_writeSeedQueue);
    }

    hyperBlocks.clear();
    for(const vector<HyperBlock>& classBlocks : resultingBlocks) {
      hyperBlocks.insert(hyperBlocks.end(), classBlocks.begin(), classBlocks.end());
    }

}