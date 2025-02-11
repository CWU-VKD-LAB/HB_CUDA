#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <limits>

using namespace std;

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
void generateHBs(vector<vector<float>>& data_with_skips, vector<vector<float>>& data_without_skips) {


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


