//
// Created by asnyd on 3/20/2025.
//

#include "IntervalHyperBlock.h"
#include <vector>
#include <future>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <map>
#include <ostream>
#include <unordered_set>
#include <utility>

#include "Interval.h"
#include "DataAttr.h"
#include "../hyperblock_generation/MergerHyperBlock.cuh"

using namespace std;

#define EPSILON 0.000001
// comparing float helper
static bool closeEnough(float a, float b) {
    return abs(a - b) < EPSILON;
}

// helper function. checks if there are any of the exact same value that our current start of an interval has, of the wrong class behind it
static bool checkBackwards(vector<DataATTR> &dataByAttribute, int currentStart) {
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
#define USED true
#define STOP 2
// the reason for these is that it needs to be different values. the threads send intervals, and we change the state.
// once the supervisor is ready, he changes it to the other one and away they go again.
#define FLIP 1
#define FLOP 0

// supervisor is going to update this after everyone has found their own best intervals. Supervisor goes through and finds best.
// then the threads are all going to go through and mark all points which are a part of that interval in their own columns.
mutex mtx;
condition_variable supervisorReady; // used to signal all the workers when they can work
condition_variable workersReady;    // used to signal boss man that we need more work

// worker function. we spawn a bunch of threads, who come here and find longest intervals in each attribute. rather than returning, they simply wait here until the supervisor
// has determined which is longest. then the threads mark all guys belonging to the longest interval, and we find the next longest interval.
// the worker finds best interval he has, then put it into threadBestInterval. this is an array of intervals for the supervisor to run through. the supervisor just populates this array with the interval which is best
// and then the workers mark all the points which are in that interval, in their own columns.
void IntervalHyperBlock::intervalHyperWorker(vector<vector<DataATTR>> &attributeColumns, Interval &threadBestInterval, int threadID, int threadCount, atomic<int> &readyThreadsCount, char *currentPhase, unordered_set<pair<int, int>, PairHash, PairEq> &usedPoints, vector<char> &doneColumns) {

    // we run this loop of finding, wait, marking, wait until the supervisor sends us the STOP signal, meaning that there were no good intervals anywhere.
    while (true){

        // initialize with this so if we find nobody bigger than 1 we just return this.
        threadBestInterval.start = -1;
        threadBestInterval.end = -1;
        threadBestInterval.attribute = threadID;
        threadBestInterval.size = -1;
        threadBestInterval.dominantClass = -1;

        // threadID corresponds to the first column we check. then we stride by number of threads to the next column.
        int n = (int)attributeColumns[threadID].size();
        Interval emptyInterval(-1,-1,-1,-1,-1);

        // run through all columns, with a stride of number of threads.
        for (int column = threadID; column < attributeColumns.size(); column += threadCount) {

            if (doneColumns[column]) {
                continue;
            }

            Interval columnBestInterval = emptyInterval;
            int currentStart = 0;
            while (currentStart < n) {
                // find our first start of the column
                while (currentStart < n && attributeColumns[column][currentStart].used == USED) {
                    currentStart++;
                }

                if (currentStart >= n)
                    break;

                // checking backwards to make sure we don't have the same value, class mismatch issue.
                // if we do have that issue, we are just going to try the next one.
                if (!checkBackwards(attributeColumns[column], currentStart)) {
                    currentStart++;
                    continue;
                }

                // class which we are trying to make an interval out of
                int startClass = attributeColumns[column][currentStart].classNum;

                // we are going to go on forward until we find a class mismatch. we are looking for 100% accurate intervals
                int currentEnd = currentStart;
                while (currentEnd < n) {
                    if (attributeColumns[column][currentEnd].classNum != startClass) {
                        break;
                    }
                    currentEnd++;
                }

                // once we are done, we simply check if this is our largest interval and update it if so.
                int length = currentEnd - currentStart;
                if (length > columnBestInterval.size && length > 1) {
                    columnBestInterval.size = length;
                    columnBestInterval.start = currentStart;
                    columnBestInterval.end = currentEnd - 1;
                    columnBestInterval.dominantClass = startClass;
                    columnBestInterval.attribute = column;
                }
                currentStart = currentEnd;
            } // end of one current start loop

            // if this column's is better than our current, update the current best
            if (columnBestInterval.size > threadBestInterval.size && columnBestInterval.size > 1) {
                threadBestInterval = columnBestInterval;
            }

            if (columnBestInterval.size < 2) {
                doneColumns[column] = 1;
            }

        } // end of one column

        // final check. if our biggest interval has a goofy edge case.
        // the edge case is when the last value in our interval is actually identical to a value of another class, this means we must remove the values which are matching from our class.
        int finalEnd = threadBestInterval.end;
        // if we won't go out of bounds by checking the next one
        if (finalEnd < n - 1 && finalEnd >= 0) {
            // get our next value
            float neighborVal = attributeColumns[threadBestInterval.attribute][threadBestInterval.end + 1].value;

            // while there is a match between the next value and where we are trying to end, we have to trim off our end guy.
            // this prevents us from making an interval which includes a value which would be shared between our class and another class.
            while (finalEnd > threadBestInterval.start && closeEnough(neighborVal, attributeColumns[threadBestInterval.attribute][finalEnd].value)) {
                finalEnd--;
            }
            threadBestInterval.end = finalEnd;
            threadBestInterval.size = finalEnd - threadBestInterval.start;
        }

        // ===========================================================
        // NOW WE WAIT FOR ALL THREADS TO FINISH AND GET HERE.
        // ===========================================================

        ++readyThreadsCount;
        // let the supervisor know someone else is done. once our counter gets to numWorkers, he is awoken
        // save the current phase so that we know when it has changed.
        char lastState = *currentPhase;
        workersReady.notify_all();

        unique_lock<mutex> findingLock(mtx);
        // Wait until the supervisor sets the signal to the opposite so that we can mark and then search again
        supervisorReady.wait(findingLock, [&] () { return *currentPhase != lastState; });

        // our ending case. when we find that there were no good intervals left in the supervisor thread, he is going to set the state to STOP.
        if (*currentPhase == STOP) {
            return;
        }

        // if our phase isn't stop, then we mark, and continue on searching again.
        // find all the used points in our columns and mark them.
        for (int column = threadID; column < attributeColumns.size(); column += threadCount) {

            // go through each of the threads columns
            for (auto &dataAtt : attributeColumns[column]) {
                // the list of blacklisted points. check if each point is in the list.
                // using the stupid unordered map is going to be way faster than keeping a list an iterating the list a bunch of times.
                pair <int, int> point = {dataAtt.classNum, dataAtt.classIndex};
                if (usedPoints.find(point) != usedPoints.end()) {
                    dataAtt.used = USED;
                }
            }
        }
        // continue back around to searching
    }
}

// EXACTLY THE SAME AS THE INTERVAL HYPER ALGORITHM, BUT IT USES A MANAGER WORKER SETUP INSTEAD OF LAUNCHING THREADS AND KILLING AND LAUNCHING AGAIN
// takes in the training data which is broken up so that each value of each point is broken up into DataATTR's. finds longest interval of an attribute which is all one class.
// then makes HBs out of all those points we found which belong to an interval.
void IntervalHyperBlock::intervalHyperSupervisor(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> &dataByAttribute, vector<HyperBlock> &hyperBlocks) {

    // sort the columns of data attributes
    for (auto &i : dataByAttribute) {
        sort(i.begin(), i.end(), [] (const DataATTR &a, const DataATTR &b) {
                 return a.value < b.value;
        });
    }

    // get our number of workers and set up our vector of intervals for them to populate.
    int numWorkers = min(thread::hardware_concurrency(), (int)dataByAttribute.size());
    Interval initializer{-1, -1, -1, -1, -1};
    vector<Interval> bestIntervals(numWorkers, initializer);

    // setting up our variables which the workers care about
    atomic<int> readyThreads(0);
    char currentPhase = 0;

    // Now declare the unordered_set using the stupid structs from above.
    unordered_set<pair<int,int>, PairHash, PairEq> usedPoints;

    // use this so that we have an early return condition. once we've found that a column doesn't have any more good intervals, we can just skip it.
    vector<char> doneColumns(dataByAttribute.size(), 0);

    // now we are ready to launch all our threads.
    vector<future<void>> futures(numWorkers);

    // launch all our workers using a bunch of nasty parameters.
    for (int i = 0; i < numWorkers; i++) {
        futures.emplace_back(
            async(launch::async, intervalHyperWorker,
                  ref(dataByAttribute),          // pass dataByAttribute by reference
                  ref(bestIntervals[i]),         // pass each Interval by reference
                  i,                            // threadID
                  numWorkers,                   // threadCount
                  ref(readyThreads),             // pass atomic<int> by reference
                  &currentPhase,                    // pass address of currentPhase (char*)
                  ref(usedPoints),              // pass usedPoints by reference
                  ref(doneColumns)
                )
        );
    }

    // now that the workers are going, we simply have a while true.
    // all we do is find the largest interval, set that in everyone's slots of the vector, and then set it to CONTINUE or STOP after we signal.
    while (true) {

        // each time around, we wait for them to finish searching
        unique_lock<mutex> searchingLock(mtx);
        // Wait until all workers have incremented readyThreads.
        workersReady.wait(searchingLock, [&]() { return readyThreads.load() == numWorkers;});

        // At this point, all workers are ready to mark.
        currentPhase = (currentPhase == FLIP) ? FLOP : FLIP;   // Set the phase to opposite so that when we start the threads they go around and then wait for us to be ready for next interval
        readyThreads.store(0);   // Reset the counter for the next round.

        // now we find best interval, and matriculate that through all the best intervals vector, so that everybody can do their marking
        Interval bestInterval(initializer);
        for (auto interval : bestIntervals) {
            // if this one is better just copy it
            if (interval.size > bestInterval.size && interval.size > 1) {
                bestInterval = interval;
            }
        }

        // fill that interval through all of bestIntervals
        for (int i = 0; i < numWorkers; i++) {
            bestIntervals[i] = bestInterval;
        }

        // clear the used points, and set it back up with all the points from the best interval
        usedPoints.clear();

        // get all our points which were in that best interval, and put them into the usedPoints set.
        // then we make an HB and put it into the Hyperblocks vector
        // if size wasn't greater than 1 we are done and stopping the workers in the else case.
        if (bestInterval.size > 1) {

            // mark all the points in the interval as used by putting them in the used set. then all the threads will do the marking
            for (int i = bestInterval.start; i <= bestInterval.end; i++) {
                DataATTR d = dataByAttribute[bestInterval.attribute][i];
                if (!d.used) {
                    usedPoints.insert({d.classNum, d.classIndex});
                }
            }

            // Signal all workers that they can continue to the marking now that we've set up the list of used points.
            supervisorReady.notify_all();

            // make our list of points which are in this best interval
            vector<vector<float>> pointsInThisBlock;
            pointsInThisBlock.reserve(bestInterval.size);  // reserve capacity to avoid extra copies
            for (int i = bestInterval.start; i <= bestInterval.end; i++) {

                // From the best attribute column, grab the identification for the point.
                DataATTR thisPoint = dataByAttribute[bestInterval.attribute][i];

                // don't use the same point multiple times.
                if (thisPoint.used) {
                    continue;
                }

                int classNum = thisPoint.classNum;
                int classIndex = thisPoint.classIndex;

                // Get the actual point from the real data and add it.
                pointsInThisBlock.push_back(realData[classNum][classIndex]);
            }

            // Compute bounds for each attribute.
            vector<vector<float>> maxes(dataByAttribute.size(), vector<float>(1, -numeric_limits<float>::infinity()));
            vector<vector<float>> mins(dataByAttribute.size(), vector<float>(1, numeric_limits<float>::infinity()));
            for (auto & point : pointsInThisBlock) {
                for (int att = 0; att < dataByAttribute.size(); att++) {
                    maxes[att][0] = max(point[att], maxes[att][0]);
                    mins[att][0] = min(point[att], mins[att][0]);
                }
            }

            // make a block and throw it into the hyperblocks vector
            HyperBlock h(maxes, mins, bestInterval.dominantClass);
            hyperBlocks.push_back(h);
            // continue back around to the waiting for everyone to be ready so that we can do next interval
        }
        // if we didn't have a best interval of more than 1 point. we just break
        else {
            currentPhase = STOP;
            supervisorReady.notify_all();
            break;
        }
    }

    // at the end of that loop, we have a bunch of points which we have not put into blocks. We now make all those guys into their own one point blocks.
    // this is easy, you just find the guys who aren't used yet, and make them their own block to live in.
    // we only have to use one column, since all the data points are in each column.
    vector<pair<int, int>> notUsedPoints;
    for (auto &dataAtt : dataByAttribute[0]) {
        if (dataAtt.used != USED) {
            notUsedPoints.push_back({dataAtt.classNum, dataAtt.classIndex});
        }
    }

    // loop through each class and all their points, and find the guys who are not used, and make blocks out of them.
    for (auto &point : notUsedPoints) {
        int classNum = point.first;
        int classIndex = point.second;

        // copy this point into it's own HB.
        vector<float> thisPoint = realData[classNum][classIndex];

        vector<vector<float>> mins(dataByAttribute.size());
        vector<vector<float>> maxes(dataByAttribute.size());
        // copy the point in as both min and max
        for (int att = 0; att < dataByAttribute.size(); att++) {
            mins[att].push_back(thisPoint[att]);
            maxes[att].push_back(thisPoint[att]);
        }
        // make a block and throw it into the hyperblocks vector
        HyperBlock h(maxes, mins, classNum);
        hyperBlocks.push_back(h);
    }
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
        // this means we have to just move on as an interval of ONE no matter what. we aren't using intervals of one, so just continue
        if (!checkBackwards(dataByAttribute, currentStart)) {
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
        if (length > bestInterval.size && length > 1) {
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

// takes in a vector of DataATTR's per attribute. which are simply our data, chopped up by attribute. Finds the longest interval of all one class across all attributes iteratively.
// populates the list of hyperblocks, and we then send those blocks to the merger_cuda to get smashed together as much as possible.
void IntervalHyperBlock::intervalHyper(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> &remainingData, vector<HyperBlock> &hyperBlocks) {
    // sort the input dataAttr's in each column by the value
    for (auto & i : remainingData) {
        sort(i.begin(), i.end(),
             [](const DataATTR &a, const DataATTR &b) {
                 return a.value < b.value;
        });
    }

    vector<bool> doneFlags(remainingData.size(), false);

    while (true) {

        // Launch our intervals asynchronously; get the intervals back and pick the biggest.
        vector<future<Interval>> intervals;
        Interval best(-1, -1, -1, -1, -1);

        // Search each attribute
        for (int i = 0; i < remainingData.size(); i++) {
            // if we are done with the column, just skip.
            if (doneFlags[i] == true)
                continue;

            intervals.emplace_back(async(launch::async, longestInterval, ref(remainingData[i]), i));
        }

        // Wait for results then find largest interval
        for (int i = 0; i < intervals.size(); i++) {

            // if we already did this column completely skip
            if (doneFlags[i] == true)
                continue;

            // else get our result from this column
            Interval intr = intervals[i].get();
            // if it returned -1, that means the attribute is done and we don't need to keep checking it. helpful for high attribute datasets.
            if (intr.size == -1) {
                doneFlags[i] = true;
            }
            else {
                // we had an interval, check if it's better than current best.
                if (intr.size >= 1 && intr.size > best.size) {
                    best = intr;  // copy entire interval
                }
            }
        }

        // Build a list of removed points based on the best interval.
        // (We assume that best.start and best.end are valid indices in remainingData[attr].)
        vector<pair<int, int>> usedIDs;

        // if we had a valid interval, we have to do all this business
        // if there was not we are obviously just done.

        if (best.size > 1) {
            // DONE BY SUPERVISOR AND USED BY EVERYONE
            for (int i = best.start; i <= best.end; i++) {
                DataATTR d = remainingData[best.attribute][i];
                if (!d.used)
                    usedIDs.push_back({d.classNum, d.classIndex});
            }

            // Build the block of points from the real data.
            // DONE BY SUPERVISOR
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

            for (auto & point : pointsInThisBlock) {
                for (int att = 0; att < remainingData.size(); att++) {
                    maxes[att][0] = max(point[att], maxes[att][0]);
                    mins[att][0] = min(point[att], mins[att][0]);
                }
            }

            // make a block and throw it into the hyperblocks vector
            HyperBlock h(maxes, mins, best.dominantClass);
            hyperBlocks.push_back(h);

            // --- REMOVAL PHASE ---
            // Remove the points that were just used from each column in remainingData.
            // --- REMOVAL PHASE ---


            // DONE BY EVERYONE
            for (auto & att : remainingData) {
                for (auto & dataAtt : att) {

                    // Check if this DataATTR matches any of the removed points.
                    for (auto &removed : usedIDs) {
                        if (dataAtt.classNum == removed.first && dataAtt.classIndex == removed.second) {
                            // Mark it used in the att-th column
                            dataAtt.used = USED;
                            break;
                        }
                    }
                }
            }
        }

        // once all the attributes returned us on intervals or just intervals of one, we break.
        else
            break;
    }

    // at the end of that loop, we have a bunch of points which we have not put into blocks. We now make all those guys into their own one point blocks.
    // this is easy, you just find the guys who aren't used yet, and make them their own block to live in.
    // we only have to use one column, since all the data points are in each column.
    vector<pair<int, int>> notUsedPoints;
    for (auto &dataAtt : remainingData[0]) {
        if (dataAtt.used != USED) {
            notUsedPoints.push_back({dataAtt.classNum, dataAtt.classIndex});
        }
    }

    // loop through each class and all their points, and find the guys who are not used, and make blocks out of them.
    for (auto &point : notUsedPoints) {
        int classNum = point.first;
        int classIndex = point.second;

        // copy this point into it's own HB.
        vector<float> thisPoint = realData[classNum][classIndex];

        vector<vector<float>> mins(remainingData.size());
        vector<vector<float>> maxes(remainingData.size());
        // copy the point in as both min and max
        for (int att = 0; att < remainingData.size(); att++) {
            mins[att].push_back(thisPoint[att]);
            maxes[att].push_back(thisPoint[att]);
        }
        // make a block and throw it into the hyperblocks vector
        HyperBlock h(maxes, mins, classNum);
        hyperBlocks.push_back(h);
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

    cout << "STARTING INTERVAL HYPER" << endl;
    // make our interval based blocks

    // the two functions use identical logic, except that one uses a supervisor thread and workers, instead of
    // constantly launching and killing threads each iteration. Supervisor version works better on any machine except cwu cluster.
    // intervalHyper(data, dataByAttribute, hyperBlocks);
    intervalHyperSupervisor(data, dataByAttribute, hyperBlocks);

    cout << "STARTING MERGING" << endl;
    try{
        merger_cuda(data, hyperBlocks, COMMAND_LINE_ARGS_CLASS);
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