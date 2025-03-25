//
// Created by asnyd on 3/20/2025.
//
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include "DataUtil.h"

extern int FIELD_LENGTH;
extern int NUM_CLASSES;

/*  Returns a class seperated version of the dataset
 *  Each class has an entry in the outer vector with a 2-d vector of its points
 */
std::vector<std::vector<std::vector<float>>> DataUtil::dataSetup(const std::string filepath, std::map<std::string, int>& classMap, std::map<int, std::string>& reversedClassMap) {
    // 3D std::std::vector: data[class][point][attribute]
    std::vector<std::vector<std::vector<float>>> data;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filepath << std::endl;
        return data;
    }

    int classNum = 0;
    std::string line;
    // Ignore the header, can use later if needed
    getline(file, line);

    // Read through all rows of CSV
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // Read the entire row, splitting by commas
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Skip empty lines
        if (row.empty()) continue;

        std::string classLabel = row.back();
        row.pop_back();

        // Check if class exists, else create new entry
        if (classMap.count(classLabel) == 0) {
            classMap[classLabel] = classNum;
            data.push_back(std::vector<std::vector<float>>());
            classNum++;
        }

        int classIndex = classMap[classLabel];

        std::vector<float> point;
        for (const std::string& val : row) {
            try {
                point.push_back(stof(val));  // Convert to float and add to the point
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value '" << val << "' in CSV" << std::endl;
                point.push_back(0.0f);  // Default to 0 if conversion fails
            }
        }

        // Add the points
        data[classIndex].push_back(point);
    }

    for (const auto& pair : classMap) {
        reversedClassMap[pair.second] = pair.first;
    }
    file.close();

    FIELD_LENGTH = data[0][0].size();
    NUM_CLASSES = classNum;

    return data;
}


/* This needs to be a function to serialize hyperblocks.
 * take in 3-D vector that is the hyperblocks for each class
 * each class gets a dimension, with a 2-d vector for the HBs
 * assumes each row in the 2-D vector is 1 hyperblock
 * the first 1/2 of the row is the max's, the end is the mins.
 */
void DataUtil::saveHyperBlocksToFile(const std::string& filepath, const std::vector<std::vector<std::vector<float>>>& hyperBlocks) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
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
    std::cout << "Hyperblocks saved to " << filepath << std::endl;
}

/**
* This will save the normalized dataset back so that we can use the same one in DV with the same normalization.
*/
void DataUtil::saveNormalizedVersionToCsv(std::string fileName, std::vector<std::vector<std::vector<float>>>& data) {
    std::ofstream outFile(fileName);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
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

std::vector<HyperBlock> DataUtil::loadBasicHBsFromCSV(const std::string& fileName) {
    std::ifstream file(fileName);
    std::vector<HyperBlock> hyperBlocks;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return hyperBlocks;
    }

    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::vector<float>> minimums, maximums;
        std::string value;
        std::vector<float> temp_vals;

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
* A function to normalize the test set using the given mins/maxes that were used to normalize the initial set
*/
void DataUtil::normalizeTestSet(std::vector<std::vector<std::vector<float>>>& testSet, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH) {
    if (testSet.empty()){
      std::cout << "Test set was empty when trying to normalize" << std::endl;
      return;
	}


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


void DataUtil::minMaxNormalization(std::vector<std::vector<std::vector<float>>>& dataset, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH) {
    std::cout << "Normalizing the dataset" << std::endl;
    if (dataset.empty()) return;

    int num_classes = dataset.size();

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
}


// Function to reorder testing dataset based on training class std::mapping
std::vector<std::vector<std::vector<float>>> DataUtil::reorderTestingDataset(const std::vector<std::vector<std::vector<float>>>& testingData, const std::map<std::string, int>& CLASS_MAP_TRAINING, const std::map<std::string, int>& CLASS_MAP_TESTING) {
    // Create a new std::vector with the same size as the testing data
    std::vector<std::vector<std::vector<float>>> reorderedTestingData(testingData.size());

    // Create a mapping from testing indices to training indices
    std::map<int, int> indexMap;
    // For each (className -> trainingIndex) in the training map
    for (const auto& entry : CLASS_MAP_TRAINING) {
        const std::string& className   = entry.first;
        int trainingIndex = entry.second;

        // Find the same class in the testing map
        auto it = CLASS_MAP_TESTING.find(className);
        if (it != CLASS_MAP_TESTING.end()) {
            int testingIndex = it->second;
            indexMap[testingIndex] = trainingIndex;

            // Ensure our output vector is big enough
            if (trainingIndex >= reorderedTestingData.size()) {
                reorderedTestingData.resize(trainingIndex + 1);
            }
        }
    }

    // Reorder the testing data
    for (int testingIndex = 0; testingIndex < testingData.size(); testingIndex++) {
        auto it = indexMap.find(testingIndex);
        if (it != indexMap.end()) {
            int trainingIndex = it->second;
            reorderedTestingData[trainingIndex] = testingData[testingIndex];
        }
    }

    return reorderedTestingData;
}


/***
* We want to go through the hyperBlocks that were generated and write them to a file.
*
*
* This print isn't caring about disjunctive blocks.
*/
void DataUtil::saveBasicHBsToCSV(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH){
	// Open file for writing
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }

	// min1, min2, min3, ..., minN, max1, max2, max3, ..., maxN, class
	for (const auto& hyperBlock : hyperBlocks) {
        // Write minimums
        for (const std::vector<float>& min : hyperBlock.minimums) {
            file << min[0] << ",";
        }

        // Write maximums
        for (const std::vector<float>& max : hyperBlock.maximums) {
            file << max[0] << ",";
        }

        // Write the class number
        file << hyperBlock.classNum << "\n";
    }

    file.close();
}

/**
* Find the min/max values in each column of data across the dataset.
* Can use this in normalization and also for making sure test set is normalized with
* the same values as the training set.
*/
void DataUtil::findMinMaxValuesInDataset(const std::vector<std::vector<std::vector<float>>>& dataset, std::vector<float>& minValues, std::vector<float>& maxValues, int FIELD_LENGTH) {
    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                minValues[k] = std::min(minValues[k], point[k]);
                maxValues[k] = std::max(maxValues[k], point[k]);
            }
        }
    }
}



std::vector<bool> DataUtil::markUniformColumns(const std::vector<std::vector<std::vector<float>>>& data) {
    // std::cout << "Starting mark uniform columns\n" << std::endl;

    if (data.empty() || data[0].empty()) return std::vector<bool>(); // Handle edge case

    int numCols = data[0][0].size();
    std::vector<bool> removed(numCols, false);

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

    return removed;
}


std::vector<std::vector<float>> DataUtil::flattenMinsMaxesForRUB(std::vector<HyperBlock>& hyper_blocks, int FIELD_LENGTH){
    // Declare std::vectors
    std::vector<float> flatMinsList;
    std::vector<float> flatMaxesList;
    std::vector<float> blockEdges;
    std::vector<float> blockClasses(hyper_blocks.size());

    // First block starts at index 0
    blockEdges.push_back(0.0f);

    // Iterate over each hyper block
    for (size_t hb = 0; hb < hyper_blocks.size(); hb++) {
        // Use a reference so we don't copy the block
        const HyperBlock &block = hyper_blocks[hb];
        blockClasses[hb] = static_cast<float>(block.classNum);

        int blockLength = 0;
        // Process each attribute in this block
        for (size_t m = 0; m < block.minimums.size(); m++) {
            // The number of intervals for the attribute
            float numIntervals = static_cast<float>(block.minimums[m].size());
            blockLength += block.minimums[m].size();
            // Push the number of intervals first, as in the Java version
            flatMinsList.push_back(numIntervals);
            flatMaxesList.push_back(numIntervals);
            // Use the built-in insert method to add the interval values
            flatMinsList.insert(flatMinsList.end(), block.minimums[m].begin(), block.minimums[m].end());
            flatMaxesList.insert(flatMaxesList.end(), block.maximums[m].begin(), block.maximums[m].end());
        }
        // Add the cumulative length of this block (plus FIELD_LENGTH offset, like DV.fieldLength in Java)
        blockEdges.push_back(blockEdges.back() + blockLength + FIELD_LENGTH);
    }

    // Assemble the result using move semantics to avoid extra copies
    std::vector<std::vector<float>> result;
    result.push_back(std::move(flatMinsList));
    result.push_back(std::move(flatMaxesList));
    result.push_back(std::move(blockEdges));
    result.push_back(std::move(blockClasses));
    return result;
}


std::vector<std::vector<float>> DataUtil::flattenDataset(std::vector<std::vector<std::vector<float>>>& data) {
    std::vector<float> dataset;
    std::vector<float> classBorder(data.size() + 1);
    classBorder[0] = 0.0f;

    // For each class
    for (size_t classN = 0; classN < data.size(); ++classN) {
        // Set the end index of the class in the flattened dataset.
        // (data[classN].size() returns the number of points in that class.)
        classBorder[classN + 1] = static_cast<float>(data[classN].size()) + classBorder[classN];

        // For each point in this class, append its attributes to the dataset.
        for (const auto& point : data[classN]) {
            // For each attribute in the point
            for (const auto& attribute : point) {
                dataset.push_back(attribute);
            }
        }
    }
    std::vector<std::vector<float>> result;
    result.push_back(move(dataset));
    result.push_back(move(classBorder));
    return result;
}


// our function to flatten our list of HBs without encoding lengths in. this is what we use for removing attirbutes
std::vector<std::vector<float>> DataUtil::flatMinMaxNoEncode(std::vector<HyperBlock> hyper_blocks, int FIELD_LENGTH) {

    int size = hyper_blocks.size();
    std::vector<float> flatMinsList;
    std::vector<float> flatMaxesList;
    std::vector<float> blockEdges(size + 1, 0.0f);
    std::vector<float> blockClasses(size, 0.0f);
    std::vector<float> intervalCounts(size * FIELD_LENGTH, 0.0f);

    // First block starts at 0.
    blockEdges[0] = 0.0f;
    int idx = 0;

    // Process each hyper block
    for (size_t hb = 0; hb < size; hb++) {
        HyperBlock& block = hyper_blocks[hb];
        // cast the classNum as a float so that we can put it in the std::vector of floats we are returning
        blockClasses[hb] = static_cast<float>(block.classNum);
        int length = 0;

        // Iterate through each attribute in the block
        for (size_t m = 0; m < block.minimums.size(); m++) {
            // Number of possible MIN/MAX values for the attribute
            size_t numIntervals = block.minimums[m].size();
            length += static_cast<int>(numIntervals);

            // Record the count of intervals for the current attribute
            intervalCounts[idx] = static_cast<float>(numIntervals);
            idx++;

            // Add all intervals for the current attribute.
            for (size_t i = 0; i < numIntervals; i++) {
                flatMinsList.push_back(block.minimums[m][i]);
                flatMaxesList.push_back(block.maximums[m][i]);
            }
        }
        // Mark the end of the block by accumulating the length.
        blockEdges[hb + 1] = blockEdges[hb] + length;
    }

    // Return the five arrays in a vector (order matches the original Java return)
    return { flatMinsList, flatMaxesList, blockEdges, blockClasses, intervalCounts };
}
