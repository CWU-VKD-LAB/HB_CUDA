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

/**
     * Finds largest interval across all dimensions of a set of data.
     * @param dataByAttribute all data split by attribute
     * @param accThreshold accuracy threshold for interval
     * @param existingHB existing hyperblocks to check for overlap
     * @return largest interval
     */
std::vector<DataATTR> IntervalHyperBlock::intervalHyper(std::vector<std::vector<DataATTR>>& dataByAttribute, float accThreshold, std::vector<HyperBlock>& existingHB){

    std::vector<std::future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -1);

    // Search each attribute
    for (int i = 0; i < dataByAttribute.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(std::launch::async, IntervalHyperBlock::longestInterval, ref(dataByAttribute[i]), accThreshold, ref(existingHB), i));
    }

    // Wait for results then find largest interval
    for(auto& future1 : intervals){
        Interval intr = future1.get();
        if(intr.size > 1 && intr.size > best.size){
            best.size = intr.size;
            best.start = intr.start;
            best.end = intr.end;
            best.attribute = intr.attribute;

            attr = intr.attribute;
        }
    }

    // Construct ArrayList of data
    std::vector<DataATTR> longest;
    if(best.size != -1){
        for(int i = best.start; i <= best.end; i++){
            longest.push_back(dataByAttribute[attr][i]);
        }
    }

    return longest;
}




/**
 * Seperates data into seperate vecs by attribute
 */
std::vector<std::vector<DataATTR>> IntervalHyperBlock::separateByAttribute(std::vector<std::vector<std::vector<float>>>& data, int FIELD_LENGTH){
    std::vector<std::vector<DataATTR>> attributes;

    // Go through the attribute columns
    for(int k = 0; k < FIELD_LENGTH; k++){
        std::vector<DataATTR> tmpField;

        // Go through the classes
        for(int i = 0; i < data.size(); i++){
            // Go through the points
            for(int j = 0; j < data[i].size(); j++){
                tmpField.push_back(DataATTR(data[i][j][k], i, j));
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
void IntervalHyperBlock::sortByColumn(std::vector<std::vector<float>>& classData, int colIndex) {
    sort(classData.begin(), classData.end(), [colIndex](const std::vector<float>& a, const std::vector<float>& b) {
        return a[colIndex] < b[colIndex];
    });
}


/***
 * Finds the longest interval in a sorted list of data by attribute.
 * @param dataByAttribute sorted data by attribute
 * @param accThreshold accuracy threshold for interval
 * @param existingHB existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval IntervalHyperBlock::longestInterval(std::vector<DataATTR>& dataByAttribute, float accThreshold, std::vector<HyperBlock>& existingHB, int attr){
    //cout << "Started longest interval \n" << endl;

    Interval intr(1, 0, 0, attr);
    Interval max_intr(-1, -1, -1, attr);

    int n = dataByAttribute.size();
    float misclassified = 0;

    for(int i = 1; i < n; i++){
        // If current class matches with next
        if(dataByAttribute[intr.start].classNum == dataByAttribute[i].classNum){
            intr.size++;
        }
        else if( (misclassified+1) / intr.size > accThreshold){
            // ^ i think this is a poor way to check. but not changing rn for the translation from java
            misclassified++;
            intr.size++;
        }
        else{
            // Remove value from interval if accuracy is below threshold.
            if(dataByAttribute[i-1].value == dataByAttribute[i].value){
                // remove then skip overlapped values
                IntervalHyperBlock::removeValueFromInterval(dataByAttribute, intr, dataByAttribute[i].value);
                i = IntervalHyperBlock::skipValueInInterval(dataByAttribute, i, dataByAttribute[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && IntervalHyperBlock::checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
                max_intr.start = intr.start;
                max_intr.end = intr.end;
                max_intr.size = intr.size;
                max_intr.attribute = attr;
            }

            // Reset curr interval
            intr.size = 1;
            intr.start = i;
            misclassified = 0;
        }
        intr.end = i;
    }

    // final check update longest interval if it doesn't overlap
    if(intr.size > max_intr.size && IntervalHyperBlock::checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    //cout << "Finished longest interval \n" << endl;

    return max_intr;
}


/*
*  Check if interval range overlaps with any existing hyperblocks
*  to not overlap the interval maximum must be below all existing hyperblock minimums
*  or the interval minimum must be above all existing hyperblock maximums
*/
bool IntervalHyperBlock::checkIntervalOverlap(std::vector<DataATTR>& dataByAttribute, Interval& intr, int attr, std::vector<HyperBlock>& existingHB){
    // interval range of vals
    float intv_min = dataByAttribute[intr.start].value;
    float intv_max = dataByAttribute[intr.end].value;

    for(const HyperBlock& hb : existingHB){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    // If unique return true
    return true;
}



int IntervalHyperBlock::skipValueInInterval(std::vector<DataATTR>& dataByAttribute, int i, float value){
    while(dataByAttribute[i].value == value){
        if(i < dataByAttribute.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    return i;
}


void IntervalHyperBlock::removeValueFromInterval(std::vector<DataATTR>& dataByAttribute, Interval& intr, float value){
    while(dataByAttribute[intr.end].value == value){
        if(intr.end > intr.start){
            intr.size--;
            intr.end--;
        }
        else{
            intr.size = -1;
            break;
        }
    }
}

void IntervalHyperBlock::generateHBs(std::vector<std::vector<std::vector<float>>>& data, std::vector<HyperBlock>& hyperBlocks, std::vector<int> &bestAttributes, int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS){
    // Hyperblocks generated with this algorithm
    std::vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    std::vector<std::vector<DataATTR>> dataByAttribute = separateByAttribute(data, FIELD_LENGTH);
    std::vector<std::vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    std::vector<std::vector<std::vector<float>>> datum;
    std::vector<std::vector<std::vector<float>>> seed_data;
    std::vector<std::vector<int>> skips;
	// "Initialized datum, seed_data, skips\n" << endl;

    // Initially generate blocks
    while(dataByAttribute[0].size() > 0){

        std::vector<DataATTR> intv = intervalHyper(dataByAttribute, 100, gen_hb);
  		all_intv.push_back(intv);

    // if hyperblock is unique then add
    if(intv.size() > 1){
        std::vector<std::vector<std::vector<float>>> hb_data;
        std::vector<std::vector<float>> intv_data;

        // Add the points from real data that are in the intervals
        for(DataATTR& dataAttr : intv){
            intv_data.push_back(data[dataAttr.classNum][dataAttr.classIndex]);
        }

        // add data and hyperblock
        hb_data.push_back(intv_data);

        HyperBlock hb(hb_data, intv[0].classNum);

        gen_hb.push_back(hb);
    }else{
        break;
    }
}

    // Add all hbs from gen_hb to hyperBlocks
    hyperBlocks.insert(hyperBlocks.end(), gen_hb.begin(), gen_hb.end());

    // All data: go through each class and add points from data
    for(const std::vector<std::vector<float>>& classData : data){
        datum.push_back(classData);
        seed_data.push_back(std::vector<std::vector<float>>());
        skips.push_back(std::vector<int>());
    }

    // find which data to skip
    for(const std::vector<DataATTR>& dataAttrs : all_intv){
        for(const DataATTR& dataAttr : dataAttrs){
            skips[dataAttr.classNum].push_back(dataAttr.classIndex);
        }
    }
    // Sort the skips
    for(std::vector<int>& skip : skips){
        sort(skip.begin(), skip.end());
    }

    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(skips[i].size() > 0){
                if(j != skips[i][0]){
                    seed_data[i].push_back(data[i][j]);
                }
                else{
                    // remove first element from skips[i]
                    skips[i].erase(skips[i].begin());
                }
            }
            else{
                seed_data[i].push_back(data[i][j]);
            }
        }
    }

    // Sort data by most important attribute
    for(int i = 0; i < datum.size(); i++){
        sortByColumn(datum[i], bestAttributes[i]);
        sortByColumn(seed_data[i], bestAttributes[i]);
    }

    try{
        merger_cuda(seed_data, datum, hyperBlocks, COMMAND_LINE_ARGS_CLASS);
    }catch (std::exception e){
        std::cout << "Error in generateHBs: merger_cuda" << std::endl;
    }
}


// Source
void IntervalHyperBlock::merger_cuda(const std::vector<std::vector<std::vector<float>>>& dataWithSkips, const std::vector<std::vector<std::vector<float>>>& allData, std::vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS) {

  	int NUM_CLASSES = dataWithSkips.size();
    int FIELD_LENGTH = allData[0][0].size();

    std::cout << "Num classes " << NUM_CLASSES << std::endl;

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : allData) {
        numPoints += classData.size();
    }

    // Count blocks per class
    std::vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyperBlocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    std::vector<std::vector<HyperBlock>> resultingBlocks(NUM_CLASSES);

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

    for (int classN = temp; classN < goToClass; classN++) {

        // set our device based on class. this way even single threaded we use multiple GPUs
        // MORE MULTI GPU BUSINESS
        //cudaSetDevice(classN % deviceCount);

        int totalDataSetSizeFlat = numPoints * PADDED_LENGTH;
        int sizeWithoutHBpoints = ((dataWithSkips[classN].size() + numBlocksOfEachClass[classN]) * PADDED_LENGTH);
        if (dataWithSkips[classN].empty()) {
            sizeWithoutHBpoints = numBlocksOfEachClass[classN] * PADDED_LENGTH;
        }

        // Compute grid size to cover all elements. we already know our ideal block size from before.
        int gridSize = ((sizeWithoutHBpoints / PADDED_LENGTH) + blockSize - 1) / blockSize;

        #ifdef DEBUG
        std::cout << "Grid size: " << gridSize << std::endl;
        std::cout << "Block size: " << blockSize << std::endl;
        std::cout << "Shared memory size: " << sharedMemSize << std::endl;
        #endif

        // Allocate host memory
        std::vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        std::vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        std::vector<int> deleteFlagsC(sizeWithoutHBpoints / PADDED_LENGTH);

        int nSize = allData[classN].size();
        std::vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

        // Fill hyperblock arrays
        int currentClassIndex = 0;
        for (int currentClass = 0; currentClass < dataWithSkips.size(); currentClass++) {
            for (const auto& point : dataWithSkips[currentClass]) {
                if (currentClass == classN) {
                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        //if (removed[attr]) continue;
                        hyperBlockMinsC[currentClassIndex] = point[attr];
                        hyperBlockMaxesC[currentClassIndex] = point[attr];
                        currentClassIndex++;
                    }
                    for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                        hyperBlockMinsC[currentClassIndex] = -std::numeric_limits<float>::infinity();
                        hyperBlockMaxesC[currentClassIndex] = std::numeric_limits<float>::infinity();
                        currentClassIndex++;
                    }
                }
            }
        }

        // Process other class points
        int otherClassIndex = 0;
        for (int currentClass = 0; currentClass < allData.size(); currentClass++) {
            if (currentClass == classN) continue;

            for (const auto& point : allData[currentClass]) {
                for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                    pointsC[otherClassIndex++] = point[attr];
                }
                for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                    pointsC[otherClassIndex++] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        // Add the existing blocks from intervalHyper
        for (auto it = hyperBlocks.begin(); it != hyperBlocks.end(); ++it) {
            if (it->classNum == classN) {
                for (int i = 0; i < it->minimums.size(); i++) {
                    //if (removed[i]) continue;
                    hyperBlockMinsC[currentClassIndex] = it->minimums[i][0];
                    hyperBlockMaxesC[currentClassIndex] = it->maximums[i][0];
                    currentClassIndex++;
                }
                for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                    hyperBlockMinsC[currentClassIndex] = -std::numeric_limits<float>::infinity();
                    hyperBlockMaxesC[currentClassIndex] = std::numeric_limits<float>::infinity();
                    currentClassIndex++;
                }
            }
        }

        // Allocate device memory
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / PADDED_LENGTH;
        std::vector<int> seedQueue(numBlocks);
        for(int i = 0; i < numBlocks; i++){
            seedQueue[i] = i;
        }

        cudaMalloc(&d_mergable, numBlocks * sizeof(int));
        cudaMemset(d_mergable, 0, numBlocks * sizeof(int));
        cudaMalloc(&d_seedQueue, numBlocks * sizeof(int));
        cudaMalloc(&d_writeSeedQueue, numBlocks * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_hyperBlockMins, hyperBlockMinsC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyperBlockMaxes, hyperBlockMaxesC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsC.data(), pointsC.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seedQueue, seedQueue.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "Launched a kernel for class: " << classN << std::endl;

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
        cudaMemcpy(hyperBlockMinsC.data(), d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hyperBlockMaxesC.data(), d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(deleteFlagsC.data(), d_deleteFlags, deleteFlagsC.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // Process results
        for (int i = 0; i < hyperBlockMinsC.size(); i += PADDED_LENGTH) {

            if (deleteFlagsC[i / PADDED_LENGTH] == -1) continue;  // -1 is a seed block which was merged to. so it doesn't need to be copied back.

            std::vector<std::vector<float>> blockMins(FIELD_LENGTH);
            std::vector<std::vector<float>> blockMaxes(FIELD_LENGTH);
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
    for(const std::vector<HyperBlock>& classBlocks : resultingBlocks) {
      hyperBlocks.insert(hyperBlocks.end(), classBlocks.begin(), classBlocks.end());
    }
}

