#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include "CudaUtil.h"

using namespace std;

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset

// Struct version of DataATTR xrecord
struct DataATTR {
    float value; // Value of one attribute of a point
    int classNum; // The class number of the point
    int classIndex; // The index of point within the class

    DataATTR(float val, int cls, int index) : value(val), classNum(cls), classIndex(index) {}
};

// Interval struct to make interval thing more understandable
struct Interval{
    int size;
    int start;
    int end;
    int attribute;

    Interval(int s, int st, int e, int a) : size(s), start(st), end(e), attribute(a) {}
};


///////////////////////// FUNCTIONS FOR INTERVAL_HYPER IMPLEMENTATION /////////////////////////

 /**
     * Finds largest interval across all dimensions of a set of data.
     * @param data_by_attr all data split by attribute
     * @param acc_threshold accuracy threshold for interval
     * @param existing_hb existing hyperblocks to check for overlap
     * @return largest interval
     */
vector<DataATTR> interval_hyper(vector<vector<DataATTR>>& data_by_attr, float acc_threshold, vector<HyperBlock>& existing_hb){
    vector<Future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -,1);

    // Search each attribute
   // Search each attribute
    for (size_t i = 0; i < data_by_attr.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(launch::async, longest_interval, cref(data_by_attr[i]), acc_threshold, cref(existing_hb), i));
    }

    // Wait for results then find largest interval
    for(auto& future : intervals){
        Interval intr = future.get();
        if(intr.size > 1 && intr.size > best.size){
            best = intr;
            attr = intr.attribute;
        }
    }

    // Construct ArrayList of data
    vector<DataATTR> longest;
    if(best.size != -1){
        for(int i = best.start; i <= best.end; i++){
            longest.push_back(data_by_attr[attr][i]);
        }
    }

    return longest;
}

/**
 * Seperates data into seperate vecs by attribute
 */
vector<vector<DataATTR>> separateByAttribute(vector<vector<vector<float>>>& data){
    vector<vector<DataATTR>> attributes;

    // Go through the attribute columns
    for(int k = 0; k < FIELD_LENGTH; k++){
        vector<DataATTR> tmpField;

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
void sortByColumn(vector<vector<float>>& classData, int colIndex) {
    sort(classData.begin(), classData.end(), [colIndex](const vector<float>& a, const vector<float>& b) {
        return a[colIndex] < b[colIndex];
    });
}

/***
 * Finds the longest interval in a sorted list of data by attribute.
 * @param data_by_attr sorted data by attribute
 * @param acc_threshold accuracy threshold for interval
 * @param existing_hb existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval longest_interval(vector<DataATTR>& data_by_attr, float acc_threshold, vector<HyperBlock>& existing_hb, int attr){
    Interval intr(1, 0, 0, attr);
    Interval max_intr(-1, -1, -1, attr);

    int n = data_by_attr.size();
    float misclassified = 0;

    for(int i = 1; i < n; i++){
        // If current class matches with next
        if(data_by_attr[intr.start].classNum == data_by_attr[i].classNum){
            intr.size++;
        }
        else if( (misclassified+1) / intr.size > acc_threshold){
            // ^ i think this is a poor way to check. but not changing rn for the translation from java
            misclassified++;
            intr.size++;
        }
        else{
            // Remove value from interval if accuracy is below threshold.
            if(data_by_attr[i-1].value == data_by_attr[i].value){
                // remove then skip overlapped values
                remove_value_from_interval(data_by_attr, intr, data_by_attr[i].value);
                i = skip_value_in_interval(data_by_attr, i, data_by_attr[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && check_interval_overlap(data_by_attr, intr, attr, existing_hb)){
                max_intr.start = intr.start;
                max_intr.end = intr.end;
                max_intr.size = intr.size;
            }

            // Reset curr interval
            intr.size = 1;
            intr.start = i;
            misclassified = 0;
        }
        intr.end = i;
    }

    // final check update longest interval if it doesn't overlap
    if(intr.size > max_intr.size && check_interval_overlap(data_by_attr, intr, attr, existing_hb)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    return max_intr;
}


bool check_interval_overlap(vector<DataATTR>& data_by_attr, Interval& intr, int attr, vector<HyperBlock>& existing_hb){
    // interval range of vals
    float intv_min = data_by_attr[intr.start].value;
    float intv_max = data_by_attr[intr.end].value;
   
    /*
    *   check if interval range overlaps with any existing hyperblocks
    * to not overlap the interval maximum must be below all existing hyperblock minimums
    * or the interval minimum must be above all existing hyperblock maximums
    */
    for(const HyperBlock& hb : existing_hb){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    // If unique return true
    return true;
}

//skip_value_in_interval
int skip_value_in_interval(vector<DataATTR>& data_by_attr, int i, float value){
    while(data_by_attr[i].value == value){
        if(i < data_by_attr.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    return i;
}


//remove_value_from_interval
void remove_value_from_interval(vector<DataATTR>& data_by_attr, Interval& intr, float value){
    while(data_by_attr[intr.end].value == value){
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

///////////////////////// END FUNCTIONS FOR INTERVAL_HYPER IMPLEMENTATION /////////////////////////

void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyper_blocks){
    // Hyperblocks generated with this algorithm
    vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    vector<vector<DataATTR>> data_by_attr = separateByAttribute(data);
    vector<vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    vector<vector<vector<float>> > datum;
    vector<vector<vector<float>> > seed_data;
    vector<vector<int>> skips;

    // Initially generate blocks
    try{
        while(data_by_attr[0].size() > 0){
            vector<DataATTR> intv = interval_hyper(data_by_attr, 100, gen_hb);
            all_intv.push_back(intv);

            // if hyperblock is unique then add
            if(intv.size() > 1){
                vector<vector<vector<float>>> hb_data;
                vector<vector<float>> intv_data;

                // Add the points from real data that are in the intervals
                for(DataATTR& dataAttr : intv){
                    intv_data.push_back(data[dataAttr.classNum][dataAttr.classIndex]);
                }

                // add data and hyperblock
                hb_data.push_back(intv_data);
                HyperBlock hb(hb_data, intv[0].classNum);
                gen_hb.push_back(hb);
            }
            else{
                break;
            }
        }

        // Add all hbs from gen_hb to hyper_blocks
        hyper_blocks.insert(hyper_blocks.end(), gen_hb.begin(), gen_hb.end());

        // All data: go through each class and add points from data
        for(const vector<vector<float>>& classData : data){
            datum.push_back(classData);
            seed_data.emplace_back(vector<float>());
            skips.emplace_back(vector<int>());
        }

        // find which data to skip
        for(const vector<DataATTR>& dataAttrs : all_intv){
            for(const DataATTR& dataAttr : dataAttrs){
                skips[dataAttr.classNum].push_back(dataAttr.classIndex);
            }
        }

        // Sort the skips
        for(vector<int>& skip : skips){
            sort(skip.begin(), skip.end())
        }

        for(int i = 0; i < data.size(); i++){
            for(int j = 0; j < data[i].size(); j++){
                if(skips[i].size() > 0){
                    if(j == skips[i][0]){
                        seed_data[i].push_back(data[i][j]);
                    }
                    else{
                        skips[i].erase(skips.begin());
                    }
                }
                else{
                    seed_data[i].push_back(data[i][j]);
                }
            }
        }

        // Sort data by most important attribute
        for(int i = 0; i < datum.size(); i++){
            sortByColumn(datum[i], bestAttribute);
            sortByColumn(seed_data[i], bestAttribute);
        }

    }catch(exception e){
        cout << "Error in generateHBs: intervals" << endl;
    }


    // Call CUDA function.
    try{
        merger_cuda(datum, seed_data, hyper_blocks);
    }catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
    }

}




void print3DVector(const vector<vector<vector<float>>>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "Class " << i << ":" << endl;
        for (const auto& row : vec[i]) {
            cout << "  [";
            for (size_t j = 0; j < row.size(); j++) {
                cout << row[j];
                if (j < row.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        cout << endl;  // Add spacing between classes
    }
}

/*  Returns a class seperated version of the dataset
 *  Each class has an entry in the outer vector with a 2-d vector of its points
 */
vector<vector<vector<float>>> dataSetup(const string& filepath) {
    // 3D vector: data[class][point][attribute]
    vector<vector<vector<float>>> data;

    // Map class labels to indices in `data`
    map<string, int> classMap;

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Failed to open file " << filepath << endl;
        return data;
    }

    int classNum = 0;
    string line;

    // Ignore the header, can use later if needed
    getline(file, line);

    // Read through all rows of CSV
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;

        // Read the entire row, splitting by commas
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Skip empty lines
        if (row.empty()) continue;

        string classLabel = row.back();
        row.pop_back();

        // Check if class exists, else create new entry
        if (classMap.count(classLabel) == 0) {
            classMap[classLabel] = classNum;
            data.emplace_back();
            classNum++;
        }

        int classIndex = classMap[classLabel];

        vector<float> point;
        for (const string& val : row) {
            try {
                point.push_back(stof(val));  // Convert to float and add to the point
            } catch (const invalid_argument&) {
                cerr << "Invalid value '" << val << "' in CSV" << endl;
                point.push_back(0.0f);  // Default to 0 if conversion fails
            }
        }

        // Add the points
        data[classIndex].push_back(point);
    }

    file.close();

    // Set global variables
    FIELD_LENGTH = data.empty() ? 0 : static_cast<int>(data[0][0].size());
    NUM_CLASSES = classNum;

    return data;
}


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
        atomicMin(&mergable[seedBlock], -1);
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

__global__ void rearrangeSeedQueue(int *readSeedQueue,  int *writeSeedQueue, int *deleteFlags, const int numBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;
    // now we are just going to loop through the seed queue and compute each blocks new position in the queue. 
    for(int i = threadID; i < numBlocks; i += globalThreadCount){
        // if the block is dead, we just copy it over. it is dead if we have already used it as a seed block.
        if (deleteFlags[i] < 0){
            writeSeedQueue[i] = readSeedQueue[i];
            continue;
        }
        // if we didn't merge, we are just going to iterate through and our new index is just the amount of numbers <= 0 to our LEFT.
        if (mergable[i] == 0){
            int newIndex = 0;
            for(int j = 0; j < i; j++){
                if (mergable[j] < 0){
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
                if (mergable[j] == 1){
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
__global__ void assignPointsToBlocks(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    float *thisThreadPoint = &dataPointsArray[threadID * numAttributes];

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
            startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
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

/* This needs to be a function to serialize hyperblocks.
 * take in 3-D vector that is the hyperblocks for each class
 * each class gets a dimension, with a 2-d vector for the HBs
 * assumes each row in the 2-D vector is 1 hyperblock
 * the first 1/2 of the row is the max's, the end is the mins.
 */
void saveHyperBlocksToFile(const string& filepath, const vector<vector<vector<float>>>& hyperBlocks) {
    ofstream file(filepath);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filepath << endl;
        return;
    }

    // Loop through each class (outermost vector)
    for (size_t classNum = 0; classNum < hyperBlocks.size(); classNum++) {
        // Loop through each hyperblock (2D vector)
        for (const auto& hyperblock : hyperBlocks[classNum]) {
            // Write hyperblock values
            for (size_t i = 0; i < hyperblock.size(); i++) {
                file << hyperblock[i];
                if (i < hyperblock.size()) file << ", ";
            }
            // Append the class index
            file << ", " << classNum << "\n";
        }
    }

    file.close();
    cout << "Hyperblocks saved to " << filepath << endl;
}


void minMaxNormalization(vector<vector<vector<float>>>& dataset) {
    if (dataset.empty()) return;

    int num_classes = dataset.size();

    // Min and max values for each attribute
    vector<float> min_vals(FIELD_LENGTH, numeric_limits<float>::max());
    vector<float> max_vals(FIELD_LENGTH, numeric_limits<float>::lowest());

    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                min_vals[k] = min(min_vals[k], point[k]);
                max_vals[k] = max(max_vals[k], point[k]);
            }
        }
    }

    // Step 2: Apply Min-Max normalization
    for (auto& class_data : dataset) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                // Avoid div/0
                if (max_vals[k] != min_vals[k]) {
                    point[k] = (point[k] - min_vals[k]) / (max_vals[k] - min_vals[k]);
                } else {
                    cout << "Column found with useless values" << endl;
                    point[k] = 0.5f;
                }
            }
        }
    }
}

// Source
void merger_cuda(const vector<vector<vector<float>>>& data_with_skips, const vector<vector<vector<float>>>& all_data, vector<HyperBlock>& hyper_blocks) {
    // Mark uniform columns
    vector<bool> removed = markUniformColumns(all_data); // Assuming this function exists

    // Initialize CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int totalCores = getCudaCores(prop);
    cout << "Total number of CUDA cores: " << totalCores << endl;

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : all_data) {
        numPoints += classData.size();
    }

    // Count blocks per class
    vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyper_blocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    // Process each class
    for (int classN = 0; classN < NUM_CLASSES; classN++) {
        
        int totalDataSetSizeFlat = numPoints * FIELD_LENGTH;
        int sizeWithoutHBpoints = ((data_with_skips[classN].size() + numBlocksOfEachClass[classN]) * FIELD_LENGTH);

        if (data_with_skips[classN].empty()) {
            sizeWithoutHBpoints = numBlocksOfEachClass[classN] * FIELD_LENGTH;
        }

        // Allocate host memory
        vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        vector<float> combinedMinsC(sizeWithoutHBpoints);
        vector<float> combinedMaxesC(sizeWithoutHBpoints);
        vector<int> deleteFlagsC(sizeWithoutHBpoints / FIELD_LENGTH);

        int nSize = all_data[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * FIELD_LENGTH));

        // Fill data arrays
        int currentClassIndex = 0;
        for (int currentClass = 0; currentClass < data_with_skips.size(); currentClass++) {
            for (const auto& point : data_with_skips[currentClass]) {
                if (currentClass == classN) {
                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        if (removed[attr]) continue;
                        hyperBlockMinsC[currentClassIndex] = point[attr];
                        hyperBlockMaxesC[currentClassIndex] = point[attr];
                        currentClassIndex++;
                    }
                }
            }
        }

        // Process other class points
        int otherClassIndex = 0;
        for (int currentClass = 0; currentClass < all_data.size(); currentClass++) {
            if (currentClass == classN) continue;

            for (const auto& point : all_data[currentClass]) {
                for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                    if (removed[attr]) continue;
                    pointsC[otherClassIndex++] = point[attr];
                }
            }
        }

        // Add the existing blocks from interval_hyper
        for (auto it = hyper_blocks.begin(); it != hyper_blocks.end();) {
            if (it->classNum == classN) {
                for (size_t i = 0; i < it->minimums.size(); i++) {
                    if (removed[i]) continue;
                    hyperBlockMinsC[currentClassIndex] = it->minimums[i][0];
                    hyperBlockMaxesC[currentClassIndex] = it->maximums[i][0];
                    currentClassIndex++;
                }
                it = hyper_blocks.erase(it);
            } else {
                ++it;
            }
        }

        // Allocate device memory
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_combinedMins, *d_combinedMaxes;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;
        float *d_points;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / FIELD_LENGTH) * sizeof(int));
        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / FIELD_LENGTH;
        vector<int> seedQueue(numBlocks);
        for (int i = 0; i < numBlocks; i++) {
            seedQueue[i] = i;
        }

        cudaMalloc(&d_mergable, numBlocks * sizeof(int));
        cudaMalloc(&d_seedQueue, numBlocks * sizeof(int));
        cudaMalloc(&d_writeSeedQueue, numBlocks * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_hyperBlockMins, hyperBlockMinsC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyperBlockMaxes, hyperBlockMaxesC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsC.data(), pointsC.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seedQueue, seedQueue.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int gridSize = 17;
        int sharedMemSize = 2 * FIELD_LENGTH * sizeof(float) + sizeof(int);

        mergerHelper1<<<gridSize, blockSize, sharedMemSize>>>(
            d_hyperBlockMins,
            d_hyperBlockMaxes,
            d_combinedMins,
            d_combinedMaxes,
            d_deleteFlags,
            d_mergable,
            FIELD_LENGTH,
            d_points,
            pointsC.size() / FIELD_LENGTH,
            numBlocks,
            d_seedQueue,
            d_writeSeedQueue
        );

        cudaDeviceSynchronize();

        // Copy results back
        cudaMemcpy(hyperBlockMinsC.data(), d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hyperBlockMaxesC.data(), d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(deleteFlagsC.data(), d_deleteFlags, deleteFlagsC.size() * sizeof(int), cudaMemcpyDeviceToHost);

        // Process results
        for (int i = 0; i < hyperBlockMinsC.size(); i += FIELD_LENGTH) {
            if (deleteFlagsC[i / FIELD_LENGTH] == -1) continue;

            vector<vector<float>> blockMins(FIELD_LENGTH);
            vector<vector<float>> blockMaxes(FIELD_LENGTH);

            int realIndex = 0;
            for (int j = 0; j < FIELD_LENGTH; j++) {
                if (removed[j]) {
                    blockMins[j].push_back(0.0f);
                    blockMaxes[j].push_back(2.0f);
                } else {
                    blockMins[j].push_back(hyperBlockMinsC[i + realIndex]);
                    blockMaxes[j].push_back(hyperBlockMaxesC[i + realIndex]);
                    realIndex++;
                }
            }

            hyper_blocks.emplace_back(blockMaxes, blockMins, classN);
        }

        // Free device memory
        cudaFree(d_hyperBlockMins);
        cudaFree(d_hyperBlockMaxes);
        cudaFree(d_combinedMins);
        cudaFree(d_combinedMaxes);
        cudaFree(d_deleteFlags);
        cudaFree(d_points);
        cudaFree(d_mergable);
        cudaFree(d_seedQueue);
        cudaFree(d_writeSeedQueue);
    }
}

// WE WILL ASSUME WE DONT HAVE A ID COLUMN.
// WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
int main() {
    printf("USER WARNING :: ENSURE THAT THERE IS NO ID COLUMN\n");
    printf("USER WARNING :: ENSURE THAT THE LAST COLUMN IS A CLASS COLUMN\n");

    vector<vector<vector<float>>> data = dataSetup("./datasets/iris.csv");

    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES    : " << NUM_CLASSES << endl;
    print3DVector(data);
    minMaxNormalization(data);
    print3DVector(data);

    return 0;
}