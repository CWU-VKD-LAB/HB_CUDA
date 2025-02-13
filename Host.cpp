#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <limits>
#include "CudaUtil.h"

using namespace std;

struct HyperBlock {
    vector<vector<float>> maximums;
    vector<vector<float>> minimums;
    int classNum;

    HyperBlock(const vector<vector<float>>& maxs,
               const vector<vector<float>>& mins,
               int cls) : maximums(maxs), minimums(mins), classNum(cls) {}
};

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset

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


// Function to generate hyperblocks
void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyper_blocks) {


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

            vector<vector<float>> blockMins(DataVisualizer::fieldLength);
            vector<vector<float>> blockMaxes(DataVisualizer::fieldLength);

            int realIndex = 0;
            for (int j = 0; j < DataVisualizer::fieldLength; j++) {
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


