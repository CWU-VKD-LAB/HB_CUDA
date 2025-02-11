#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset

void print2DVector(const vector<vector<float>>& vec) {
    for (const auto& row : vec) {
        cout << "[";
        for (const auto& val : row) {
            cout << val << ", ";
        }
        cout << "]" << endl;
    }
}

/*  Returns a flattened version of the dataset
 *  each inner list is flattened dataset for ONE class
 */
vector<vector<float>> dataSetup(const string& filepath) {
    vector<vector<float>> data;
    // Assign a string to int index in the data vector.
    map<string, int> classMap;

    ifstream file(filepath);
    if(!file.is_open()) {
        cerr << "Failed to open file " << filepath << endl;
    }

    int classNum = 0;
    string line;

    // Ignore the header, can use later if needed
    getline(file, line);


    // Read through all rows of CSV
    while(getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;

        while(getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        string s = row.back();
        // if we dont yet have the current class
        if(classMap.count(s) == 0) {
            classMap[s] = classNum;
            data.emplace_back();
            classNum++;
        }

        // Iterate through the columns except the last one (the class)
        for (int i = 0; i < row.size() - 1; i++) {
            try {
                // Try to convert string to float
                float value = stof(row.at(i));
                data[classMap[s]].push_back(value);
            } catch (const invalid_argument& e) {
                cerr << "Invalid value '" << row.at(i) << "' at row " << classNum << endl;
            }
        }

        // dont ask.
        FIELD_LENGTH = static_cast<int>(row.size()) - 1;
    }
    file.close();

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




// WE WILL ASSUME WE DONT HAVE A ID COLUMN.
// WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
int main() {
    printf("USER WARNING :: ENSURE THAT THERE IS NO ID COLUMN\n");
    printf("USER WARNING :: ENSURE THAT THE LAST COLUMN IS A CLASS COLUMN\n");

    vector<vector<float>> data = dataSetup("./datasets/iris.csv");

    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES    : " << NUM_CLASSES << endl;
    print2DVector(data);
    return 0;
}


