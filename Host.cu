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
#include <future>
#include "CudaUtil.h"
#include "HyperBlock.h"
#include "HyperBlockCuda.cuh"
#include <chrono>
#include <omp.h>
#include <iomanip>  // For setw
#include <algorithm> // For max
using namespace std;

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset
int COMMAND_LINE_ARGS_CLASS = -1;


map<string, int> CLASS_MAP_TRAINING;
map<string, int> CLASS_MAP_TESTING;


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

Interval longestInterval(vector<DataATTR>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB, int attr);
void removeValueFromInterval(vector<DataATTR>& dataByAttribute, Interval& intr, float value);
int skipValueInInterval(vector<DataATTR>& dataByAttribute, int i, float value);
bool checkIntervalOverlap(vector<DataATTR>& dataByAttribute, Interval& intr, int attr, vector<HyperBlock>& existingHB);
void merger_cuda(const vector<vector<vector<float>>>& dataWithSkips, const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks);
void saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks);
void print3DVector(const vector<vector<vector<float>>>& vec);


/***
* We want to go through the hyperBlocks that were generated and write them to a file.
*
*
* This print isn't caring about disjunctive blocks.
*/
void saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks, const string& fileName){
	// Open file for writing
    ofstream file(fileName);
    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

	// min1, min2, min3, ..., minN, max1, max2, max3, ..., maxN, class
	for (const auto& hyperBlock : hyperBlocks) {
        // Write minimums
        for (const vector<float>& min : hyperBlock.minimums) {
            file << min[0] << ",";
        }

        // Write maximums
        for (const vector<float>& max : hyperBlock.maximums) {
            file << max[0] << ",";
        }

        // Write the class number
        file << hyperBlock.classNum << "\n";
    }

    file.close();
}

///////////////////////// FUNCTIONS FOR intervalHyper IMPLEMENTATION /////////////////////////

 /**
     * Finds largest interval across all dimensions of a set of data.
     * @param dataByAttribute all data split by attribute
     * @param accThreshold accuracy threshold for interval
     * @param existingHB existing hyperblocks to check for overlap
     * @return largest interval
     */
vector<DataATTR> intervalHyper(vector<vector<DataATTR>>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB){
    //cout << "Starting interval hyperblock" << endl;
    vector<future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -1);

   // Search each attribute
    for (int i = 0; i < dataByAttribute.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(launch::async, longestInterval, ref(dataByAttribute[i]), accThreshold, ref(existingHB), i));
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
	//cout << "Best.start: " << best.start << "  Best.end: " << best.end <<"  Best.size: " << best.size <<  "  Best.attribute: " << best.attribute << endl;

    // Construct ArrayList of data
    vector<DataATTR> longest;
    if(best.size != -1){
        for(int i = best.start; i <= best.end; i++){
          	//cout << "Data by attribute printing time: " << dataByAttribute[attr][i].classNum << " " << dataByAttribute[attr][i].classIndex << "\n";
            //cout << attr << endl;

            longest.push_back(dataByAttribute[attr][i]);
        }
    }
    //cout << "Finished interval hyperblock" << endl;

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
 * @param dataByAttribute sorted data by attribute
 * @param accThreshold accuracy threshold for interval
 * @param existingHB existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval longestInterval(vector<DataATTR>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB, int attr){
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
                removeValueFromInterval(dataByAttribute, intr, dataByAttribute[i].value);
                i = skipValueInInterval(dataByAttribute, i, dataByAttribute[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
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
    if(intr.size > max_intr.size && checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    //cout << "Finished longest interval \n" << endl;

    return max_intr;
}


bool checkIntervalOverlap(vector<DataATTR>& dataByAttribute, Interval& intr, int attr, vector<HyperBlock>& existingHB){
    //cout << "Started check interval overlap\n" << endl;
    // interval range of vals
    float intv_min = dataByAttribute[intr.start].value;
    float intv_max = dataByAttribute[intr.end].value;
   
    /*
    *   check if interval range overlaps with any existing hyperblocks
    * to not overlap the interval maximum must be below all existing hyperblock minimums
    * or the interval minimum must be above all existing hyperblock maximums
    */
    for(const HyperBlock& hb : existingHB){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    //cout << "Finished check interval overlap\n" << endl;

    // If unique return true
    return true;
}

//skipValueInInterval
int skipValueInInterval(vector<DataATTR>& dataByAttribute, int i, float value){
    //cout << "Starting skip value in interval\n" << endl;

    while(dataByAttribute[i].value == value){
        if(i < dataByAttribute.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    //cout << "Finished skip value in interval\n" << endl;

    return i;
}


//removeValueFromInterval
void removeValueFromInterval(vector<DataATTR>& dataByAttribute, Interval& intr, float value){
    //cout << "Starting remove value from intervals\n" << endl;
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
    //cout << "Finished remove value from intervals\n" << endl;
}

///////////////////////// END FUNCTIONS FOR intervalHyper IMPLEMENTATION /////////////////////////

void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyperBlocks){
  	// "Started generating HBS\n" << endl;
    // Hyperblocks generated with this algorithm
    vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    vector<vector<DataATTR>> dataByAttribute = separateByAttribute(data);
    vector<vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    vector<vector<vector<float>>> datum;
    vector<vector<vector<float>>> seed_data;
    vector<vector<int>> skips;
	// "Initialized datum, seed_data, skips\n" << endl;

    // Initially generate blocks

        //cout << "Starting while loop to generate hyperblocks\n";
		//cout << "dataByAttribute[0].size() = " << dataByAttribute[0].size() << endl;

        while(dataByAttribute[0].size() > 0){
			//cout << "Attempting to go into intervalHyper " << endl;

            vector<DataATTR> intv = intervalHyper(dataByAttribute, 100, gen_hb);
            all_intv.push_back(intv);
			//cout << "Pushed to back of all intervals" << endl;

            // if hyperblock is unique then add
            if(intv.size() > 1){
                //cout << "making hb and intv_data" << endl;
                vector<vector<vector<float>>> hb_data;
                vector<vector<float>> intv_data;


                // Add the points from real data that are in the intervals
                for(DataATTR& dataAttr : intv){
                    /*cout << "Trying to add a dataATTR. " << endl;

					if(dataAttr.classNum > 1 || dataAttr.classNum < 0){
                    	cout << "Invalid classNum: " << dataAttr.classNum << endl;
                        cout << "Value: " << dataAttr.value << endl;
                        continue;
                    }


                    if(dataAttr.classIndex > data[dataAttr.classNum].size() - 1 || dataAttr.classIndex < 0){
                    	cout << "Invalid class index: " << dataAttr.classIndex << endl;
                        continue;
                    }
					*/
                    intv_data.push_back(data[dataAttr.classNum][dataAttr.classIndex]);
                }

                //cout << "Made it past the points from real data thingy" << endl << endl;
                // add data and hyperblock
                hb_data.push_back(intv_data);
                //cout << "Added intv data to hb_data" << endl << endl;

                HyperBlock hb(hb_data, intv[0].classNum);
                //cout << "Made the hyperblock for this interval thing" << endl << endl;

                gen_hb.push_back(hb);
                //cout << "Added results from last intervalHyper" << endl << endl;
            }else{
                //cout << "Breaking because the intv size is < 1" << endl;
                break;
            }
        }

        // Add all hbs from gen_hb to hyperBlocks
        hyperBlocks.insert(hyperBlocks.end(), gen_hb.begin(), gen_hb.end());

        // All data: go through each class and add points from data
        for(const vector<vector<float>>& classData : data){
            datum.push_back(classData);
            seed_data.push_back(vector<vector<float>>());
            skips.push_back(vector<int>());
        }

        // find which data to skip
        for(const vector<DataATTR>& dataAttrs : all_intv){
            for(const DataATTR& dataAttr : dataAttrs){
                skips[dataAttr.classNum].push_back(dataAttr.classIndex);
            }
        }
        // Sort the skips
        for(vector<int>& skip : skips){
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
            sortByColumn(datum[i], 278);
            sortByColumn(seed_data[i], 278);
        }

    // Call CUDA function.
    //cout << "Calling merger_cuda\n" << endl;

    try{
        //cout << "Printing interval hyperblocks:\n\n" << endl;
        //for(const auto& hb : hyperBlocks){
        //    for(const auto& min : hb.minimums){
        //        cout << min[0] << " ";
        //    }
        //   cout << endl;
		//
        //    for(const auto& max : hb.maximums){
        //        cout << max[0] << " ";
        //    }
        //    cout << endl;
        //}
        //cout << "End interval hyperblocks:\n\n" << endl;
		//cout << "DATUM BEING PASSED INTO MERGING:" << endl;
		//print3DVector(datum);

        //cout << "SEED DATA BEING PASSED INTO MERGING:" << endl;
		//print3DVector(seed_data);

        merger_cuda(seed_data, datum, hyperBlocks);
    }catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
    }
}




void print3DVector(const vector<vector<vector<float>>>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << "Class " << i << ":" << endl;
        for (const auto& row : vec[i]) {
            cout << "  [";
            for (int j = 0; j < row.size(); j++) {
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
vector<vector<vector<float>>> dataSetup(const string filepath, map<string, int>& classMap) {
    // 3D vector: data[class][point][attribute]
    vector<vector<vector<float>>> data;

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
            data.push_back(vector<vector<float>>());
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
    for (int classNum = 0; classNum < hyperBlocks.size(); classNum++) {
        // Loop through each hyperblock (2D vector)
        for (const auto& hyperblock : hyperBlocks[classNum]) {
            // Write hyperblock values
            for (int i = 0; i < hyperblock.size(); i++) {
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


vector<HyperBlock> loadBasicHBsFromCSV(const string& fileName) {
    ifstream file(fileName);
    vector<HyperBlock> hyperBlocks;

    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return hyperBlocks;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<vector<float>> minimums, maximums;
        string value;
        vector<float> temp_vals;

        while (getline(ss, value, ',')) {
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (!value.empty()) {
                temp_vals.push_back(stof(value));
            }
        }

        if (temp_vals.empty()) continue;

        int num_attributes = temp_vals.size() / 2;
        int classNum = static_cast<int>(temp_vals.back());
        temp_vals.pop_back(); // Remove classNum from the list

        for (int i = 0; i < num_attributes; ++i) {
            minimums.push_back({ temp_vals[i] });
            maximums.push_back({ temp_vals[i + num_attributes] });
        }

        hyperBlocks.emplace_back(maximums, minimums, classNum);
    }

    file.close();
    return hyperBlocks;
}




/**
* Find the min/max values in each column of data across the dataset.
* Can use this in normalization and also for making sure test set is normalized with
* the same values as the training set.
*/
void findMinMaxValuesInDataset(const vector<vector<vector<float>>>& dataset, vector<float>& minValues, vector<float>& maxValues) {
    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                minValues[k] = min(minValues[k], point[k]);
                maxValues[k] = max(maxValues[k], point[k]);
            }
        }
    }
}




/**
* A function to normalize the test set using the given mins/maxes that were used to normalize the initial set
*/
void normalizeTestSet(vector<vector<vector<float>>>& testSet, const vector<float>& minValues, const vector<float>& maxValues) {
    if (testSet.empty()){
      cout << "Test set was empty when trying to normalize" << endl;
      return;
	}

    // Print out the min and max values first 20
    for (int i = 0; i < 30; i++) {
      cout << minValues[i] << endl;
    }
    cout << endl;

    cout << "Maxes" << endl;


    for (int i = 0; i < 30; i++) {
      cout << maxValues[i] << ",";
    }
    cout << endl;

    for (auto& class_data : testSet) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    point[k] = 0.5f;
                }
            }
        }
    }
}

void minMaxNormalization(vector<vector<vector<float>>>& dataset, const vector<float>& minValues, const vector<float>& maxValues) {
    //cout << "Starting min-max normalization\n" << endl;

    if (dataset.empty()) return;

    int num_classes = dataset.size();

    // Min and max values for each attribute
    //vector<float> minValues(FIELD_LENGTH, std::numeric_limits<float>::infinity());
    //vector<float> maxValues(FIELD_LENGTH, -std::numeric_limits<float>::infinity());

    // Step 1: Find min and max for each attribute
	//findMinMaxValuesInDataset(dataset, minValues, maxValues);

    // Step 2: Apply Min-Max normalization
    for (auto& class_data : dataset) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                // Avoid div/0
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    //cout << "Column found with useless values" << endl;
                    point[k] = 0.5f;
                }
            }
        }
    }
    //cout << "Finished min-max normalization\n" << endl;
}

vector<bool> markUniformColumns(const vector<vector<vector<float>>>& data) {
     // cout << "Starting mark uniform columns\n" << endl;

    if (data.empty() || data[0].empty()) return vector<bool>(); // Handle edge case

    int numCols = data[0][0].size();
    vector<bool> removed(numCols, false);

    // Iterate through each column
    for (int col = 0; col < numCols; col++) {
        float referenceValue = data[0][0][col]; // Use first row of first class as reference
        bool allSame = true;

        // Check across all classes and all rows
        for (const auto& obj : data) {
            for (const auto& row : obj) {
                if (row[col] != referenceValue) {
                    allSame = false;
                    break;
                }
            }
            if (!allSame) break;
        }

        // If the column is uniform across all classes, mark it for removal
        if (allSame) {
            removed[col] = true;
        }
    }

   // cout << "Finished mark uniform columns\n" << endl;

    return removed;
}

// Source
void merger_cuda(const vector<vector<vector<float>>>& dataWithSkips, const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks) {

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

    vector<vector<HyperBlock>> resultingBlocks(NUM_CLASSES);
    
    // get our device count, and max it so that the most we will use is numClasses if we have only 2 classes for example.
    /* MULTI GPU BUSINESS
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    deviceCount = std::min(deviceCount, NUM_CLASSES);


    // Process each class
    // we have our multithreading happen here at this level. set the device to class % deviceCount
    #pragma omp parallel for num_threads(deviceCount)
    */

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

        cout << "Grid size: " << gridSize << endl;
        cout << "Block size: " << blockSize << endl;
        cout << "Shared memory size: " << sharedMemSize << endl;

        // Allocate host memory
        vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        vector<float> combinedMinsC(sizeWithoutHBpoints);
        vector<float> combinedMaxesC(sizeWithoutHBpoints);
        vector<int> deleteFlagsC(sizeWithoutHBpoints / PADDED_LENGTH);

        int nSize = allData[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

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
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_combinedMins, *d_combinedMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / PADDED_LENGTH;
        vector<int> seedQueue(numBlocks);
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
                sharedMemSize,
                d_combinedMins,
                d_combinedMaxes
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
        cudaFree(d_combinedMins);
        cudaFree(d_combinedMaxes);
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


/**
* This will save the normalized dataset back so that we can use the same one in DV with the same normalization.
*/
void saveNormalizedVersionToCsv(string fileName, vector<vector<vector<float>>>& data) {
    ofstream outFile(fileName);

    if (!outFile.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

    // Assuming all classes have at least one point, get feature count from the first point of the first class
    int featureCount = data[0][0].size();

    // Write the header
    for (int i = 0; i < featureCount; i++) {
        outFile << "x" << i << ",";
    }
    outFile << "label\n";  // Add label column

    // Iterate through classes
    for (int i = 0; i < data.size(); i++) {
        // Iterate through points in class
        for (int j = 0; j < data[i].size(); j++) {
            // Iterate through attributes of a point
            for (int k = 0; k < data[i][j].size(); k++) {
                outFile << data[i][j][k] << ",";
            }
            outFile << i << "\n";  // Append class label
        }
    }

    outFile.close();
}


/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
vector<vector<long>> testAccuracyOfHyperBlocks(vector<HyperBlock>& hyperBlocks, vector<vector<vector<float>>> testSet){

  	// Keep track of which points were never inside of a block
    vector<set<int>> pointsNotClassified(CLASS_MAP_TESTING.size(), set<int>(0));
    for(int i = 0; i < CLASS_MAP_TESTING.size(); i++){
    	set.
    }


	// Make a n x n matrix for the confusion matrix
	vector<vector<long>> ultraConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));

    cout << "Testing on " << hyperBlocks.size() << " hyperblocks" << endl;
    cout << "Testing on " << testSet.size() << " classes" << endl;
    cout << "Testing on " << testSet[0].size() << " points in first class." << endl;
    cout << "Testing on " << NUM_CLASSES << " classes" << endl;
    cout << "Testing on " << FIELD_LENGTH << " attributes" << endl;

    vector<string> listTraining(NUM_CLASSES);
    for (const auto& entry : CLASS_MAP_TRAINING) {
        int index = entry.second;
        listTraining[index] = entry.first;
    }

    vector<string> listTesting(CLASS_MAP_TESTING.size());
    for (const auto& entry : CLASS_MAP_TESTING) {
        int index = entry.second;
        listTesting[index] = entry.first;
    }


    bool anyPointWasInside = false;

    // Go through all the blocks
	for(int hb = 0; hb < hyperBlocks.size(); hb++){
        HyperBlock& currBlock = hyperBlocks[hb];
        // Go through all the classes in the testSet
		for(int cls = 0; cls < NUM_CLASSES; cls++){
            // go through all the points in a clases
        	for(int pnt = 0; pnt < testSet[cls].size(); pnt++){
           		const vector<float>& point = testSet[cls][pnt];

                if(currBlock.inside_HB(point.size(), point.data())){
                    // Get the actual class name from the training data class index
					string trainingClassName = listTraining[currBlock.classNum];

					// Find what index this class has in the testing data
					int trainingClassIndexInTestingOrder = CLASS_MAP_TESTING[trainingClassName];

					// The testingClassIndex is just cls (the current class we're testing)
					int testingClassIndex = cls;

					ultraConfusionMatrix[testingClassIndex][trainingClassIndexInTestingOrder]++;
                }
                else{
                	// don't know what to put here because it might not have a good
                }
        	}
     	}
    }

	cout << "Any point was inside" << anyPointWasInside <<  endl;
    return ultraConfusionMatrix;
}

/*
void print2DMatrix(vector<vector<long>>& data){
	vector<string> listTraining(NUM_CLASSES);

    for (const auto& entry : CLASS_MAP_TRAINING) {
        int index = entry.second;
        listTraining[index] = entry.first;
    }


  for(const auto& string: listTraining){
    cout << string << "\t" <<endl;
  }

  for(int i = 0; i < data.size(); i++){
	cout << listTraining[i] << "\t"
    for(int j = 0; j < data[i].size(); j++){
      cout << data[i][j] << "\t";
    }
    cout << endl;
  }
}

 */



// Computes the accuracy of the points classifed
void confusionMatrixAccuracy(){


}



void print2DMatrix(vector<vector<long>>& data) {
    vector<string> classLabels(NUM_CLASSES);

    vector<float> accuracies(NUM_CLASSES, 0.0);

    // Calculate the accuracies of each of the rows.
    // Only the diagonal values are correct predictions

for (int i = 0; i < NUM_CLASSES; ++i) {
    long correct = 0;
    long incorrect = 0;
    long totalClassifications = 0;

    for (int j = 0; j < NUM_CLASSES; ++j) {
        totalClassifications += data[i][j];
        if (i == j) {
            correct += data[i][j];  // Diagonal value indicates correct predictions
        } else {
            incorrect += data[i][j];  // Off-diagonal values are incorrect predictions
        }
    }

    if (totalClassifications > 0) {
        accuracies[i] = (float)correct / totalClassifications;
    }
}


    for (const auto& entry : CLASS_MAP_TESTING) {
        int index = entry.second;
        classLabels[index] = entry.first;
    }

    // Calculate column width based on the longest class name and largest number
    size_t maxWidth = 8; // Minimum width

    for (const auto& name : classLabels) {
        maxWidth = max(maxWidth, name.length() + 2);
    }

    for (const auto& row : data) {
        for (const auto& cell : row) {
            string numStr = to_string(cell);
            maxWidth = max(maxWidth, numStr.length() + 2);
        }
    }

    // Print header row with "Actual\Predicted" in the corner
    cout << setw(maxWidth) << "Act\\Pred" << " |";
    for (const auto& name : classLabels) {
        cout << setw(maxWidth) << name << " |";
    }
    cout << endl;

    // Print separator line
    cout << string(maxWidth, '-') << "-+";
    for (size_t i = 0; i < classLabels.size(); i++) {
        cout << string(maxWidth, '-') << "-+";
    }
    cout << endl;

    // Print each row with row label
    for (size_t i = 0; i < data.size(); i++) {
        cout << setw(maxWidth) << classLabels[i] << " |";

        for (size_t j = 0; j < data[i].size(); j++) {
            cout << setw(maxWidth) << data[i][j] << " |";
        }

        cout << accuracies[i] << endl;
    }
}

// Function to clear the console screen (cross-platform)
void clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

// Function to wait for user input before continuing
void waitForEnter() {
    cout << "\nPress Enter to continue...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

// Function to display the main menu
void displayMainMenu() {
    clearScreen();
    cout << "=== HyperBlock Classification System ===\n\n";
    cout << "1. Import training data.\n";
    cout << "2. Import testing data.\n";
    cout << "3. Save normalized training data.\n";
    cout << endl;
    cout << "4. Import existing hyperblocks.\n";
    cout << "5. Export existing hyperblocks.\n";
    cout << "6. Generate new hyperblocks.\n";
    cout << "7. Simplify hyperblocks.\n";
    cout << "8. Test hyperblocks on dataset.\n";
    cout << endl;
    cout << "9. Exit\n\n";
}


/**
* 	 Assume these for now, implement a better handler later.
*
*    printf("USER WARNING :: ENSURE THAT THERE IS NO ID COLUMN\n");
*    printf("USER WARNING :: ENSURE THAT THE LAST COLUMN IS A CLASS COLUMN\n");
*/
int main(int argc, char* argv[]) {

  	/* File names */
  	string normalizedSaveFile;
	string hyperBlocksImportFileName;
    string trainingDataFileName;
    string testingDataFileName;
	string hyperBlocksExportFileName;

    vector<vector<vector<float>>> testData;
    vector<vector<vector<float>>> trainingData;

    vector<float> minValues(1);	/* Holds the min attribute vals used for normalization */
    vector<float> maxValues(1); /* Holds the max attribute vals used for normalization */

    vector<HyperBlock> hyperBlocks;	/* Store the HyperBlocks we generate or import */


    if(argc == 2){
        COMMAND_LINE_ARGS_CLASS = stoi(argv[1]);
        cout << "Running on class index " << COMMAND_LINE_ARGS_CLASS << endl;
    }

	// In other ML models single point usaully is classified as 1 class only.
    // But in HyperBlocks, a point can be in multiple blocks at the same time
    // thus, we need a more complex matrix to track all of these point classifications.
	vector<vector<long>> ultraConfusionMatrix;

    bool running = true;	// Loop
	int choice;				// Main menu user input "choice"

    // Option to use the command line running style
    if(argc > 1){

    }

    while(running){
       displayMainMenu();
       cin >> choice;
       cin.clear();
       cin.ignore(numeric_limits<streamsize>::max(), '\n');

	   switch (choice) {
          case 1:	// IMPORT TRAINING DATA
			   cout << "Enter training data filename: " << endl;
               system("ls");
               getline(cin, trainingDataFileName);

               // Attempt to read from the file
               trainingData = dataSetup(trainingDataFileName, CLASS_MAP_TRAINING);

                // Reassign them with the correct field length
               minValues.assign(FIELD_LENGTH, std::numeric_limits<float>::infinity());
               maxValues.assign(FIELD_LENGTH, -std::numeric_limits<float>::infinity());
               findMinMaxValuesInDataset(trainingData, minValues, maxValues);

               minMaxNormalization(trainingData, minValues, maxValues);

			   waitForEnter();
               break;

          case 2:	// IMPORT TESTING DATA
		  	 cout << "Enter testing data filename: " << endl;
             system("ls");
             getline(cin, testingDataFileName);

             testData = dataSetup(testingDataFileName, CLASS_MAP_TESTING);
             normalizeTestSet(testData, minValues, maxValues);

             waitForEnter();

		     break;
          case 3:		// SAVE NORMALIZED TRAINING DATA
              cout << "Enter the file to save the normalized training data to: " << endl;

               // Save to the file they input
               getline(cin, normalizedSaveFile);
               saveNormalizedVersionToCsv(normalizedSaveFile, trainingData);

               cout << "Saved normalized training data to: " << normalizedSaveFile << endl;
               waitForEnter();
               break;
          case 4:	// IMPORT EXISTING HYPERBLOCKS
                cout << "Enter existing hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);

                hyperBlocks = loadBasicHBsFromCSV(hyperBlocksImportFileName);

                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;
                waitForEnter();
                break;
          case 5: // EXPORT HYPERBLOCKS
            	cout << "Enter the file to save HyperBlocks to: " << endl;

               // Save to the file they input
               getline(cin, hyperBlocksExportFileName);


			   saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName);
               break;
          case 6:	// GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    waitForEnter();
                } else {
                    hyperBlocks.clear();
                    generateHBs(trainingData, hyperBlocks);
                }
                cout << "Finished Generating HyperBlocks" << endl;

                waitForEnter();
                break;
          case 7:		// SIMPLIFY HYPERBLOCKS
               cout << "Simplifications Interface Not Yet Implemented. " << endl;
               //hyperBlocks = simplifyExistingHyperBlocks(hyperBlocks);
               break;
          case 8:		// TEST HYPERBLOCKS ON DATASET
             cout << "Testing hyperblocks on testing dataset" << endl;
             ultraConfusionMatrix = testAccuracyOfHyperBlocks(hyperBlocks, testData);
             print2DMatrix(ultraConfusionMatrix);
             waitForEnter();
             break;
          case 9:		// EXIT
             running = false;
             break;
          default:
             cout << "\nInvalid choice. Please try again." << endl;
             waitForEnter();
             break;
        }
	}
    return 0;
}
