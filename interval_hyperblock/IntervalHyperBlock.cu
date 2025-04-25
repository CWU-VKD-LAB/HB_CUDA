//
// Created by asnyd on 3/20/2025.
//

#include "IntervalHyperBlock.h"
#include <algorithm>
using namespace std;


#define EPSILON 1e-6
// comparing float helper

static bool closeEnough(float a, float b) {
    return abs(a - b) < EPSILON;
}

// helper function. checks if there are any of the exact same value that our current start of an interval has, of the wrong class behind it
// basically tells us whether or not an index is a valid place we can start an interval from
static bool checkValidStart(vector<DataATTR> &dataByAttribute, int currentStart) {
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

// helper function. returns how many indexes we can move forward while still maintaining the integrity of our pure interval, and
// also returns the count of points which we are using for the first time in this interval. Meaning the count of previously unused points.
static pair<int, int> checkForwards(vector<DataATTR> &dataByAttribute, int currentStart, int targetClass) {
    int n = dataByAttribute.size();
    int end = currentStart;

    int notUsedYetPoints = 0;
    while (end + 1 < n && dataByAttribute[end + 1].classNum == targetClass) {

        // we need to return how many points we are using for the first time. This a better indicator of size for an interval.
        // it could be 50 points long interval, but 49 are used, that is not going to speed up the removal process much.
        if (dataByAttribute[end + 1].used != true)
            notUsedYetPoints++;
        end++;
    }

    // 2) If the next item (end+1) is still in range and:
    //    - has the SAME-ISH as dataByAttribute[end],
    //    - but is a DIFFERENT class,
    //    then we must trim off the last items with that value
    //    because that attribute value is "shared" by a different class.
    if (end + 1 < n) {
        float nextVal = dataByAttribute[end + 1].value;
        bool differentClass = (dataByAttribute[end + 1].classNum != targetClass);
        bool sameValue = closeEnough(dataByAttribute[end].value, nextVal);
        // If we have a conflict (same value, different class),
        // remove *all* items that share this value from the tail
        if (differentClass && sameValue) {
            float conflictVal = nextVal;
            // Trim backward while the end item has the conflictVal
            while (end >= currentStart && closeEnough(dataByAttribute[end].value, conflictVal)) {

                // if the point is unused, but we aren't using it, shrink the counter
                if (dataByAttribute[end].used != true)
                    notUsedYetPoints--;

                end--;

            }
        }
    }

    if (end < currentStart) {
        // means the conflict happened right away,
        // so the valid interval is effectively just "start" itself or empty.
        return {currentStart, 0};
    }

    return {end, notUsedYetPoints};
}

static pair <int, int> checkBackwards(vector<DataATTR> &dataByAttribute, int currentStart, int targetClass) {

    if (!checkValidStart(dataByAttribute, currentStart)) {
        return {currentStart, 0};
    }

    // extend down while we legally can
    int beg = currentStart;
    int freshPoints = 0;
    while (beg - 1 >= 0 && checkValidStart(dataByAttribute, beg - 1)) {
        if (!dataByAttribute[beg-1].used)
            ++freshPoints;
        --beg;
    }

    // no trimming step here – if we ever hit a conflict, we’d have caught it in step 0
    return { beg, freshPoints };
}

// generate a HB from each seed case, we simply make a block of the pure attribute around each attribute of each case.
// so we would generate a block from each case, and then we end up merging them all.
// Assumes ‑std=c++17 and OpenMP enabled (‑fopenmp / /openmp)
void IntervalHyperBlock::pureBlockIntervalHyper(vector<vector<DataATTR>> &dataByAttribute, vector<vector<vector<float>>> &trainingData,vector<HyperBlock> &hyperBlocks,int COMMAND_LINE_ARGS_CLASS) {
    const int FIELD_LENGTH = dataByAttribute.size();
    bool doOneClass = (COMMAND_LINE_ARGS_CLASS != -1);

    for (int classification = 0; classification < trainingData.size(); classification++) {

        if (doOneClass && classification != COMMAND_LINE_ARGS_CLASS)
            continue;

        // go through each point
        #pragma omp parallel for schedule(static)
        for (int point = 0; point < trainingData[classification].size(); ++point) {
            /* 2 · containers to remember where the seed sits in every column   */
            vector<int> attrPos (FIELD_LENGTH, -1);   // index inside column
            vector<float> lower (FIELD_LENGTH);       // placeholder for bounds
            vector<float> upper (FIELD_LENGTH);

            /* 3 · binary‑search each attribute column ------------------------*/
            for (int d = 0; d < FIELD_LENGTH; ++d) {

                const float seedVal = trainingData[classification][point][d];
                auto &column = dataByAttribute[d];

                /* lower_bound on value (columns already sorted by value) */
                auto it = lower_bound(column.begin(), column.end(), seedVal, [](const DataATTR &a, float v){ return a.value < v; });

                /* Walk forward over duplicates until we match classNum & classIndex */
                while (it != column.end() && it->value != seedVal && !(it->classIndex == point && it->classNum == classification)) {
                    ++it;
                }

                // track where we are in each column
                attrPos[d] = static_cast<int>(distance(column.begin(), it));

                // now our checking up and down
                int upperIndex = checkForwards(column, attrPos[d], classification).first;

                int lowerIndex = checkBackwards(column, attrPos[d], classification).first;

                // set up our bounds with the values of furthest we can expand in each attribute
                lower[d] = column[lowerIndex].value;
                upper[d] = column[upperIndex].value;
            }

            // make a block out of the bounds we have just found
            vector<vector<float>> maxes(dataByAttribute.size(), vector<float>(1, -numeric_limits<float>::infinity()));
            vector<vector<float>> mins(dataByAttribute.size(), vector<float>(1, numeric_limits<float>::infinity()));
            for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {
                mins[attribute][0] = lower[attribute];
                maxes[attribute][0] = upper[attribute];
            }

            HyperBlock block(maxes, mins, classification);

            #pragma omp critical
            hyperBlocks.emplace_back(move(block));
        }
    }

    // now we go through and for each block we just check if the block is all the way inside a previous block already, if so we are going to remove it.
    // --- after you finish generating the blocks -------------------------------
    const int nBlocks = hyperBlocks.size();
    vector<char> keep(nBlocks, 1);      // we keep if true.

    // mark redundant blocks in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nBlocks; ++i) {
        const HyperBlock &small = hyperBlocks[i];
        for (int j = 0; j <  i; ++j) {

            const HyperBlock &big = hyperBlocks[j];
            bool inside = true;
            for (int d = 0; d < FIELD_LENGTH; ++d) {

                if (big.maximums[d][0] < small.maximums[d][0] || big.minimums[d][0] > small.minimums[d][0]) {
                    inside = false;
                    break;
                }
            }
            if (inside) {
                keep[i] = 0;
                break;
            } // mark for deletion and stop early
        }
    }

    int w = 0;
    for (int r = 0; r < nBlocks; ++r)
        if (keep[r]) {
            if (w != r) // ← guard against self-move
                hyperBlocks[w] = std::move(hyperBlocks[r]);
            ++w;
        }
    hyperBlocks.erase(hyperBlocks.begin() + w, hyperBlocks.end());

}

#define STOP 2
// the reason for these is that it needs to be different values. the threads send intervals, and we change the state.
// once the supervisor is ready, he changes it to the other one and away they go again. Without this, the workers don't stop and
// wait for the supervisor to be ready
#define FLIP 1
#define FLOP 0

mutex mtx;
condition_variable supervisorReady; // used to signal all the workers when they can work
condition_variable workersReady;    // used to signal boss man that we need more work
// worker function. we spawn a bunch of threads, who come here and find longest intervals in each attribute. rather than returning, they simply wait here until the supervisor
// has determined which is longest. then the threads mark all guys belonging to the longest interval, and we find the next longest interval.
// the worker finds best interval he has, then put it into threadBestInterval. this is an array of intervals for the supervisor to run through. the supervisor just populates this array with the interval which is best
// and then the workers mark all the points which are in that interval, in their own columns.
void IntervalHyperBlock::intervalHyperWorker(vector<vector<DataATTR>> &attributeColumns, Interval &threadBestInterval, int threadID, int threadCount, atomic<int> &readyThreadsCount, char *currentPhase, unordered_set<pair<int, int>, PairHash, PairEq> &usedPoints, vector<char> &doneColumns, int COMMAND_LINE_ARGS_CLASS) {

    // if the class is -1 we are doing them all. If not, we can treat all wrong class points as countercases, and don't build intervals from them
    bool doingOneClass = (COMMAND_LINE_ARGS_CLASS != -1) ? true : false;

    // we run this loop of finding, wait, marking, wait until the supervisor sends us the STOP signal, meaning that there were no good intervals anywhere.
    while (true){
        Interval emptyInterval(-1,-1,-1,-1,-1);
        // set with empty, and if we DO find a good one we obviously replace
        threadBestInterval = emptyInterval;

        // run through all columns, with a stride of number of threads.
        for (int column = threadID; column < attributeColumns.size(); column += threadCount) {
            int n = (int)attributeColumns[column].size();

            // if we have already found there are no good intervals to make with this column, we can skip.
            if (doneColumns[column]) {
                continue;
            }
          
            Interval columnBestInterval = emptyInterval;
            int currentStart = 0;
            while (currentStart < n) {
                // find our first start of the column
                // if we have used the point, or, we are doing one class, and this is the wrong class point, we skip it and don't consider it as a start.
                while (currentStart < n && (attributeColumns[column][currentStart].used == true || (doingOneClass && attributeColumns[column][currentStart].classNum != COMMAND_LINE_ARGS_CLASS))) {
                    currentStart++;
                }

                if (currentStart >= n)
                    break;

                // checking backwards to make sure we don't have the same value, class mismatch issue.
                // if we do have that issue, we are just going to try the next one.
                if (!checkValidStart(attributeColumns[column], currentStart)) {
                    currentStart++;
                    continue;
                }

                // class which we are trying to make an interval out of
                int startClass = attributeColumns[column][currentStart].classNum;

                // we are going to go on forward until we find a class mismatch. we are looking for 100% accurate intervals
                // the first thing is the furthest end we can include in the interval purely, and the second is the amount of points
                // which we are using for the first time in the interval. it seems to work slightly better than simply using the size as top - bottom of interval
                pair<int, int> result = checkForwards(attributeColumns[column], currentStart, startClass);
                int currentEnd = result.first;
                int uniquePoints = result.second;

                // once we are done, we simply check if this is our largest interval and update it if so.
                if (uniquePoints > columnBestInterval.size && uniquePoints > 1) {
                    columnBestInterval.size = uniquePoints;
                    columnBestInterval.start = currentStart;
                    columnBestInterval.end = currentEnd;
                    columnBestInterval.dominantClass = startClass;
                    columnBestInterval.attribute = column;
                }
                currentStart = currentEnd + 1;
            } // end of one current start loop

            // if this column's is better than our current, update the current best
            if (columnBestInterval.size > threadBestInterval.size && columnBestInterval.size > 1) {
                threadBestInterval = columnBestInterval;
            }

            // this is the reason we need a column best and worker best. that allows us to know whether there are intervals left to find in an attribute, even if it's not the best.
            if (columnBestInterval.size < 2) {
                doneColumns[column] = 1;
            }
        } // end of one column

        // ===========================================================
        // NOW WE WAIT FOR ALL THREADS TO FINISH THEIR COLUMNS AND GET HERE.
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
                    dataAtt.used = true;
                }
            }

            // if this is the column which we got the best global interval from:
            // we are going to remove all but the start of the interval, since we don't need it.
            // this makes it a little bit faster to continue to run through intervals constantly.
            if (column == threadBestInterval.attribute) {
                // inclusive bound, so that we remove the second element, and then it's an exclusive bound, so we remove all the way to the end of the interval.
                attributeColumns[column].erase(
                    attributeColumns[column].begin() + (threadBestInterval.start + 1),
                    attributeColumns[column].begin() + (threadBestInterval.end + 1)
                );
            }
        }
        // continue back around to searching
    }
}

// EXACTLY THE SAME AS THE INTERVAL HYPER ALGORITHM, BUT IT USES A MANAGER WORKER SETUP INSTEAD OF LAUNCHING THREADS AND KILLING AND LAUNCHING AGAIN
// takes in the training data which is broken up so that each value of each point is broken up into DataATTR's. finds longest interval of an attribute which is all one class.
// then makes HBs out of all those points we found which belong to an interval.
void IntervalHyperBlock::intervalHyperSupervisor(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> &dataByAttribute, vector<HyperBlock> &hyperBlocks, int COMMAND_LINE_ARGS_CLASS) {

    // sort the columns of data attributes
    for (auto &i : dataByAttribute) {
        sort(i.begin(), i.end(), [] (const DataATTR &a, const DataATTR &b) {
                 return a.value < b.value;
        });
    }

    // get our number of workers and set up our vector of intervals for them to populate.
    int numWorkers = fmin(thread::hardware_concurrency(), (int)dataByAttribute.size());
    numWorkers = 1;
    cout << "Number of workers: " << numWorkers << endl;
    Interval initializer{-1, -1, -1, -1, -1};
    vector<Interval> bestIntervals(numWorkers, initializer);

    // setting up our variables which the workers care about
    atomic<int> readyThreads(0);
    char currentPhase = 0;

    // Now declare the unordered_set using the stupid structs from above.
    unordered_set<pair<int,int>, PairHash, PairEq> usedPoints;

    // use this so that we have an early return condition. once we've found that a column doesn't have any more good intervals, we can just skip it.
    vector<char> doneColumns(dataByAttribute.size(), 0);

    vector<thread> workers;
    workers.reserve(numWorkers);

    // launch all our workers using a bunch of nasty parameters.
    for (int i = 0; i < numWorkers; i++) {
        workers.emplace_back(
            intervalHyperWorker,
            ref(dataByAttribute),          // pass dataByAttribute by reference
            ref(bestIntervals[i]),         // pass each Interval by reference
          	i,                            // threadID
          	numWorkers,                   // threadCount
            ref(readyThreads),             // pass atomic<int> by reference
            &currentPhase,                    // pass address of currentPhase (char*)
            ref(usedPoints),              // pass usedPoints by reference
            ref(doneColumns),
            COMMAND_LINE_ARGS_CLASS
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
            cout << "Best size supervisor: " << interval.size << endl;
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

    for (auto &worker : workers) {
        if (worker.joinable())
            worker.join();
    }

    // at the end of that loop, we have a bunch of points which we have not put into blocks. We now make all those guys into their own one point blocks.
    // this is easy, you just find the guys who aren't used yet, and make them their own block to live in.
    // we only have to use one column, since all the data points are in each column.
    vector<pair<int, int>> notUsedPoints;
    for (auto &dataAtt : dataByAttribute[0]) {
        if (dataAtt.used != true) {
            notUsedPoints.push_back({dataAtt.classNum, dataAtt.classIndex});
        }
    }

    // loop through all the points, and find the guys who are not used, and make blocks out of them.
    for (auto &point : notUsedPoints) {
        int classNum = point.first;

        // if we are just doign one class, and that class is not the class of this dataPoint, then we skip it. don't make blocks unnecessarily
        if (COMMAND_LINE_ARGS_CLASS != -1 && COMMAND_LINE_ARGS_CLASS != classNum)
            continue;

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

    cout << "Made it to the end" << endl;
}

// use with the regular interval hyper below. Used with openMP or futures to launch a thread to get longest attribute, but it is inefficient because you make a kill so many threads.
Interval IntervalHyperBlock::longestInterval(vector<DataATTR> &dataByAttribute, int attribute)
{
    cout << "Attribute ran on: " << attribute << endl;
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
        if (!checkValidStart(dataByAttribute, currentStart)) {
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
            // DONE BY SUPERVISOR AND true BY EVERYONE
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

            for (auto & att : remainingData) {
                for (auto & dataAtt : att) {

                    // Check if this DataATTR matches any of the removed points.
                    for (auto &removed : usedIDs) {
                        if (dataAtt.classNum == removed.first && dataAtt.classIndex == removed.second) {
                            // Mark it used in the att-th column
                            dataAtt.used = true;
                            break;
                        }
                    }
                }
            }
        }

        // once all the attributes returned us no intervals or just intervals of one, we break.
        else
            break;
    }

    // at the end of that loop, we have a bunch of points which we have not put into blocks. We now make all those guys into their own one point blocks.
    // this is easy, you just find the guys who aren't used yet, and make them their own block to live in.
    // we only have to use one column, since all the data points are in each column.
    vector<pair<int, int>> notUsedPoints;
    for (auto &dataAtt : remainingData[0]) {
        if (dataAtt.used != true) {
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

    // the two functions use almost identical logic, except that one uses a supervisor thread and workers, instead of
    // constantly launching and killing threads each iteration. Supervisor version works better on any machine except cwu cluster.

    // intervalHyper(data, dataByAttribute, hyperBlocks);
    // intervalHyperSupervisor(data, dataByAttribute, hyperBlocks, COMMAND_LINE_ARGS_CLASS);
    pureBlockIntervalHyper(dataByAttribute, data, hyperBlocks, COMMAND_LINE_ARGS_CLASS);

    cout << "Num blocks after interval: " << hyperBlocks.size() << endl;
    cout << "STARTING MERGING" << endl;
    try{
        merger_cuda(data, hyperBlocks, COMMAND_LINE_ARGS_CLASS);
        // dataByAttribute = separateByAttribute(data, FIELD_LENGTH);
        // mergerNotInCuda(data, hyperBlocks, dataByAttribute);
    } catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
        cout << e.what() << endl;
    }
}

void IntervalHyperBlock::merger_cuda(const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS) {

    int NUM_CLASSES = allData.size();
    int FIELD_LENGTH = allData[0][0].size();

    cout << "Num classes " << NUM_CLASSES << endl;

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : allData) {
        numPoints += classData.size();
    }

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

    vector<vector<HyperBlock>> inputBlocks(NUM_CLASSES);
    vector<vector<HyperBlock>> resultingBlocks(NUM_CLASSES);
    vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (HyperBlock& hyperBlock : hyperBlocks) {
        // store this block in the slot which corresponds to it's class.
        numBlocksOfEachClass[hyperBlock.classNum]++;
        inputBlocks[hyperBlock.classNum].push_back(hyperBlock);
    }

    for (int classN = temp; classN < goToClass; classN++) {

        // set our device based on class. this way even single threaded we use multiple GPUs
        // MORE MULTI GPU BUSINESS
        //cudaSetDevice(classN % deviceCount);

        int totalDataSetSizeFlat = numPoints * PADDED_LENGTH;

        // Compute grid size to cover all HBs. we already know our ideal block size from before.
        int gridSize = ((numBlocksOfEachClass[classN]) + blockSize - 1) / blockSize;

        int currentClassBlockLengthFlattened = inputBlocks[classN].size() * PADDED_LENGTH;

        // Allocate host memory
        vector<float> hyperBlockMinsC(currentClassBlockLengthFlattened);
        vector<float> hyperBlockMaxesC(currentClassBlockLengthFlattened);
        vector<int> deleteFlagsC(currentClassBlockLengthFlattened / PADDED_LENGTH);

        int nSize = allData[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

        // Fill hyperblock array
        for (int i = 0; i < inputBlocks[classN].size(); i++) {
            HyperBlock &h = inputBlocks[classN][i];
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

    // Assign them their size.
    for(HyperBlock& hyperBlock : hyperBlocks) {
        hyperBlock.find_avg_and_size(allData);
    }
}

// helper function. takes in a block, and the DataByAttribute columns. What we do is just check our interval of each attribute.
// we make a list of wrong class points in each column. Then we take that smallest list, and query all those other lists, and if any point
// is inside of all the other lists, (inside our bounds for all attributes) we fail. if every wrong class point is missing from ast least one list, we pass
bool IntervalHyperBlock::checkMergable(vector<vector<DataATTR>> &dataByAttribute, HyperBlock &h) {

    int FIELD_LENGTH = dataByAttribute.size();

    // each column has a local set to themselves
    vector<unordered_set<pair<int, int>, PairHash, PairEq>> localSets(FIELD_LENGTH);

    for (int column = 0; column < FIELD_LENGTH; column++) {
        // one thread per column. we basically just determine if a point is in the bounds of the HB for that attribute, and if so, we put him in our set of terrorists.
        // go through each column's interval of attributes, and find guys who are wrong class.
        // they go on terrorism watchlist, and at the end, everybody finds if they are sharing same guy on a watchlist or not.
        for (int start = h.topBottomPairs[column].first; start <= h.topBottomPairs[column].second; start++) {
            DataATTR &d = dataByAttribute[column][start];
            if (d.classNum != h.classNum) {
                localSets[column].insert({d.classNum, d.classIndex});
            }
        }
    }

    // find the smallest set our of each set from all columns
    auto smallestSet = min_element(
        localSets.begin(),
        localSets.end(),
        [](const unordered_set<pair<int,int>, PairHash, PairEq> &a,
           const unordered_set<pair<int,int>, PairHash, PairEq> &b) {
            return a.size() < b.size();
        }
    );


    if (smallestSet != localSets.end()) {
        // For each of the other sets, remove elements not present in that set
        for (auto it = localSets.begin(); it != localSets.end(); ++it) {
            if (it == smallestSet) {
                continue; // skip comparing smallest set to itself
            }

            // Remove from *smallestSet any element that is NOT in *it
            for (auto stIt = smallestSet->begin(); stIt != smallestSet->end(); ) {
                if (it->find(*stIt) == it->end()) {
                    stIt = smallestSet->erase(stIt);
                } else {
                    ++stIt;
                }
            }

            // If the intersection becomes empty, we can return early
            if (smallestSet->empty()) {
                return true;
            }
        }
    }

    // If after all comparisons we didn't empty the smallest set,
    // it means there's at least one element in all sets => fail
    return smallestSet == localSets.end() || smallestSet->empty();

}

#define KILL 1
#define LIVE 0
void IntervalHyperBlock::mergerNotInCuda(vector<vector<vector<float>>> &trainingData, vector<HyperBlock> &hyperBlocks, vector<vector<DataATTR>> &pointsBrokenUp) {

    int FIELD_LENGTH = hyperBlocks[0].minimums.size();

    // the basic algorithm is this. we are going to maintain a terrorist watchlist.
    // we find all points which are in our bounds for any attributes. when we are expanding our bounds to include
    // new friends, we find all points which now might be in our new bound obviously, and add to watchlist. Then we just check the watchlist.
    // for each block
    vector<vector<HyperBlock>> inputBlocks(trainingData.size());

    for (HyperBlock &h : hyperBlocks) {

        // Ensure we have a pair for each attribute.
        // Using resize is better than reserve here because we want to create FIELD_LENGTH elements,
        // which we then update.
        h.topBottomPairs.resize(FIELD_LENGTH);

        // For each attribute (column), perform a binary search.
        for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

            // Get the sorted vector of DataATTR for this attribute.
            const vector<DataATTR>& columnPoints = pointsBrokenUp[attribute];

            // Our target interval for this attribute (assuming each is stored as a one-element vector)
            float lowerVal = h.minimums[attribute][0];
            float upperVal = h.maximums[attribute][0];

            // Binary search:
            // lower_bound: first element whose value is not less than lowerVal.
            auto lowIt = lower_bound(columnPoints.begin(), columnPoints.end(), lowerVal,
                [](const DataATTR &d, float value) {
                    return d.value < value;
                });

            // upper_bound: first element whose value is greater than upperVal.
            auto highIt = upper_bound(columnPoints.begin(), columnPoints.end(), upperVal,
                [](float value, const DataATTR &d) {
                    return value < d.value;
                });

            // Determine indices:
            // If lowIt reached the end, then no elements are ≥ lowerVal.
            int lowIndex = (lowIt != columnPoints.end()) ? static_cast<int>(lowIt - columnPoints.begin()) : -1;

            // For the topmost index, we want the last index in the range.
            // upper_bound returns the first element greater than upperVal.
            // If highIt equals columnPoints.begin(), then even the first element is greater than upperVal.
            int highIndex = (highIt != columnPoints.begin()) ? static_cast<int>(highIt - columnPoints.begin()) - 1 : -1;

            // Optionally, you might want to check whether these indices make sense.
            // For example, if lowIndex == -1 or highIndex == -1 or if lowIndex > highIndex,
            // it means no valid element was found for that attribute.
            // You can handle that case as needed.

            // Store the pair of indices (bottommost, topmost) for this attribute.
            h.topBottomPairs[attribute] = make_pair(lowIndex, highIndex);
        }

        // add the HyperBlock to inputBlocks
        inputBlocks[h.classNum].emplace_back(h);
    }


    for (int classN = 0; classN < inputBlocks.size(); classN++) {

        vector<HyperBlock>& blocks = inputBlocks[classN];

        vector<char> mergableFlags(blocks.size(), 0);
        vector<char> deleteFlags(blocks.size(), LIVE);

        for (int seed = 0; seed < blocks.size() - 1; seed++) {

            HyperBlock &seedBlock = blocks[seed];

            // now we check if this block is mergeable to all the blocks after it.
            // go through each seed block. now what we do is we are going to have to make that rearranging business happen just like in merger_cuda.
            omp_set_num_threads(min((int)blocks.size() - 1 - seed, omp_get_num_procs()));

            #pragma omp parallel for
            for (int candidateBlock = seed + 1; candidateBlock < blocks.size(); candidateBlock++) {

                HyperBlock &candidate = blocks[candidateBlock];

                // make it a copy of candidate for now. the bounds' values don't matter unless we pass the test, at which point we update them.
                HyperBlock combinedBlock(candidate.minimums, candidate.maximums, candidate.classNum);
                combinedBlock.topBottomPairs.resize(FIELD_LENGTH);

                for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {
                    // new block has max of the two tops, and min of the two bottoms.
                    combinedBlock.topBottomPairs[attribute] = {min(seedBlock.topBottomPairs[attribute].first,candidate.topBottomPairs[attribute].first),
                                                                    max(seedBlock.topBottomPairs[attribute].second, candidate.topBottomPairs[attribute].second)};
                }

                // check merging using set based checking instead of brute force checking the entire dataset.
                if (checkMergable(pointsBrokenUp, combinedBlock)) {
                    mergableFlags[candidateBlock] = true;
                    deleteFlags[seed] = KILL; // we can kill the seedblock if we are able to merge with any blocks.

                    vector<vector<float>> combinedMins(FIELD_LENGTH);
                    vector<vector<float>> combinedMaxes(FIELD_LENGTH);

                    for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {
                        combinedMins[attribute].push_back(min(seedBlock.minimums[attribute][0], candidate.minimums[attribute][0]));
                        combinedMaxes[attribute].push_back(max(seedBlock.maximums[attribute][0], candidate.maximums[attribute][0]));
                    }
                    // copy our new bounds into this block.
                    candidate.minimums = combinedMins;
                    candidate.maximums = combinedMaxes;

                    candidate.topBottomPairs = combinedBlock.topBottomPairs;
                }
            }

            // after we have checked all our candidate blocks, we are going to rearrange the blocks like this.
            // if we merged, we go to the back of the line. BUT, blocks which were earlier in the blocks go to the back, and ones which were later go to front.
            // meaning that if block 1 merged, and block N merged, block 1 gets put in behind block N.

            // Gather indices for blocks after 'seed'
            vector<int> indices;
            for (int i = seed + 1; i < blocks.size(); i++) {
                indices.push_back(i);
            }

            // Sort these indices based on mergableFlags criteria.
            sort(indices.begin(), indices.end(), [&](int a, int b) {
                if (mergableFlags[a] != mergableFlags[b])
                    return mergableFlags[a] < mergableFlags[b]; // false (0) first; i.e., non-mergeable remain at front.
                if (mergableFlags[a]) // if both are true, sort by index descending (later block comes first)
                    return a > b;
                return a < b; // if both are false, keep the original order.
            });

            // Rebuild sorted arrays.
            // First, copy blocks, mergableFlags, and deleteFlags from indices 0 to seed (unchanged)
            vector<HyperBlock> sortedBlocks;
            vector<char> sortedMergable;
            vector<char> sortedDelete;
            for (int i = 0; i <= seed; i++) {
                sortedBlocks.push_back(blocks[i]);
                sortedMergable.push_back(mergableFlags[i]);
                sortedDelete.push_back(deleteFlags[i]);
            }
            
            // Then, append the sorted blocks for indices > seed.
            for (int i : indices) {
                sortedBlocks.push_back(blocks[i]);
                sortedMergable.push_back(mergableFlags[i]);
                sortedDelete.push_back(deleteFlags[i]);
            }

            // Update the originals with the newly ordered data.
            blocks = move(sortedBlocks);
            mergableFlags = move(sortedMergable);
            deleteFlags = move(sortedDelete);

            // Reset all mergeable flags to 0.
            fill(mergableFlags.begin(), mergableFlags.end(), 0);
        }

        // now once we are done with that merging business. we simply remove all the blocks which were marked KILL
        blocks.erase(
        remove_if(blocks.begin(), blocks.end(),
            [&](const HyperBlock& block) {
                size_t index = &block - &blocks[0];
                return deleteFlags[index] == KILL;
            }),
            blocks.end()
        );
    }
    // add all the blocks from each class back to hyperBlocks pointer
    hyperBlocks.clear();
    for (vector<HyperBlock>& blocks : inputBlocks) {
        hyperBlocks.insert(hyperBlocks.end(), blocks.begin(), blocks.end());
    }
}



