#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <future>
#include "./lda/LDA.cpp"
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "./cuda_util/CudaUtil.h"
#include "./hyperblock_generation/MergerHyperBlock.cuh"
#include "./hyperblock/HyperBlock.h"
#include "./interval_hyperblock/IntervalHyperBlock.h"
#include "./knn/Knn.h"
#include "./screen_output/PrintingUtil.h"
#include "./data_utilities/DataUtil.h"
#include "./simplifications/Simplifications.h"
using namespace std;

#ifdef _WIN32
    const string PATH_SEPARATOR = "\\";
#else
    const string PATH_SEPARATOR = "/";
#endif

#define LDA_ORDERING true

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset
int COMMAND_LINE_ARGS_CLASS = -1; // used for when we are splitting up generation one class per machine. This lets us run on many computers at once.

map<string, int> CLASS_MAP;
map<string, int> CLASS_MAP_TESTING;

map<int, string> CLASS_MAP_INT;
map<int, string> CLASS_MAP_TESTING_INT;

/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
vector<vector<long>> testAccuracyOfHyperBlocks(vector<HyperBlock>& hyperBlocks, vector<vector<vector<float>>> &testSet){

  	// Keep track of which points were never inside of a block, when a point is classifed we increment the map internal vectors correct positon
    // there should be CLASS_NUM unordered_maps or just hashmaps, in each will hold a vector<point_index, vector<int> of len(class_num)>
    vector<unordered_map<int, vector<int>>> pointsNotClassified(CLASS_MAP.size());

    // Go through each class
    for(int cls = 0; cls < NUM_CLASSES; cls++){
        // Put the index of each point in each class into a set, this is how we will track which points were never classified.
        for(int j = 0; j < testSet[cls].size(); j++){
            pointsNotClassified[cls][j] = vector<int>(NUM_CLASSES);
        }
    }

	// Make a n x n matrix for the confusion matrix
	vector<vector<long>> ultraConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));
    vector<vector<long>> regularConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));

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

					ultraConfusionMatrix[cls][currBlock.classNum]++;

                    // Go to the actual class, to the right points entry, and increment the "predicted" class (the hb it was in).
                    pointsNotClassified[cls][pnt][currBlock.classNum]++;
                }
        	}
     	}
    }

    // Lets count how many points fell into blocks of multiple classes
    for(int i = 0; i < NUM_CLASSES; i++){
       int numPointsInMultipleClasses = 0;
       int numPointsInNoBlocks = 0;

       // Go through all the points in a class.
       for(int pnt = 0; pnt < testSet[i].size(); pnt++){
           char in = 0;

           // Go through the classification vector for the point
           for(int cls = 0; cls < NUM_CLASSES; cls++){
               if(pointsNotClassified[i][pnt][cls] > 0){
                  in++;
               }

               // Means it fell into multiple of the same.
               if(in > 1){
                   break;
               }
           }

           if(in > 1) {
               numPointsInMultipleClasses++;
           }

           if(in == 0) numPointsInNoBlocks++;
       }

       cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN MULTIPLE CLASSES BLOCKS: " << numPointsInMultipleClasses << endl;
       cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN NO BLOCKS: " << numPointsInNoBlocks << endl;
    }

    vector<vector<vector<float>>> unclassifiedPointVec(NUM_CLASSES, vector<vector<float>>()); // [class][pointIdx][attr]

    // Lets count how many points fell into blocks of multiple classes
    for(int i = 0; i < NUM_CLASSES; i++){
        // Go through all the points in a class.
        for(int pnt = 0; pnt < testSet[i].size(); pnt++){
            int majorityClass = -1;
            int max = 0;

            // Go through the classification vector for the point
            for(int cls = 0; cls < NUM_CLASSES; cls++){
                if(pointsNotClassified[i][pnt][cls] > max){
                   max = pointsNotClassified[i][pnt][cls];
                   majorityClass = cls;
                }
            }

            // The majority was the one they are actually predicted to be in
            if(majorityClass != -1){
                regularConfusionMatrix[i][majorityClass]++;

            }else{
                // Put the point that wasn't classified into the vector to go to Knn
                unclassifiedPointVec[i].push_back(testSet[i][pnt]);
            }
        }
    }

    cout << "\n\n\n\n" << endl;
    cout << "============================ REGULAR CONFUSION MATRIX ==================" << endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END CONFUSION MATRIX ======================" << endl;

	cout << "Any point was inside" << anyPointWasInside <<  endl;

    cout << "\n\n\n\n" << endl;
    cout << "============================ K-NN CONFUSION MATRIX ==================" << endl;
    int k = 1;
    vector<vector<long>> secondConfusionMatrix = Knn::kNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
     PrintingUtil::printConfusionMatrix(secondConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END K-NN MATRIX ======================" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }

    cout << "\n\n\n\n" << endl;
    cout << "============================ DISTINCT POINT CONFUSION MATRIX ==================" << endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END DISTINCT POINT MATRIX ======================" << endl;
    cout << "\n\n\n\n" << endl;

    return ultraConfusionMatrix;
}

// This function computes the LDA ordering for a given training dataset.
// It sets up the bestVectors, bestVectorsIndexes, and eachClassBestVectorIndex.
// best vectors is the weights of each coefficient from the LDF function
// bestVectorsIndexes is just the indexes that correspond to those weights from the function, since we are sorting them
// eachClassBestVectorIndex is the one best attribute for each class, we sort by this when generating blocks, and it helps a bit.
void computeLDAOrdering(const vector<vector<vector<float>>>& trainingData, vector<vector<float>>& bestVectors, vector<vector<int>>& bestVectorsIndexes, vector<int>& eachClassBestVectorIndex) {
    // Run LDA on the training data.
    bestVectors = linearDiscriminantAnalysis(trainingData);

    // Resize our index containers.
    bestVectorsIndexes.assign(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    eachClassBestVectorIndex.assign(NUM_CLASSES, 0);

    // For each class, initialize the indexes and then sort (if desired)
    // and determine the index with the largest absolute LDA coefficient.
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Populate with initial indices: 0, 1, 2, ... FIELD_LENGTH - 1.
        for (int j = 0; j < FIELD_LENGTH; j++) {
            bestVectorsIndexes[i][j] = j;
        }

#ifdef LDA_ORDERING
        // Optionally sort the indexes for class i based on the absolute value of the LDA coefficients.
        sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
             [&](int a, int b) {
                 return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
             });
#endif
        // Find the index (from bestVectorsIndexes) corresponding to the largest absolute LDA coefficient.
        // We use the values in bestVectors[i] for comparison.
        auto it = max_element(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
                              [&](int a, int b) {
                                  return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
                              });
        eachClassBestVectorIndex[i] = distance(bestVectorsIndexes[i].begin(), it);
    }
}

void runKFold(vector<vector<vector<float>>> &dataset) {
    if (dataset.empty()) {
        cout << "Please enter a training dataset before using K Fold validation" << endl;
        return;
    }

    cout << "Please Enter a K value:\t";
    int k;
    cin >> k;

    // Clear the newline from the input buffer.
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    if (cin.fail() || k < 2) {
        cout << "Error: Invalid input. Please enter a valid integer greater than 1." << endl;
        // Clear the error state and ignore any remaining input.
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        return;
    }

    vector<vector<vector<vector<float>>>> kFolds = DataUtil::splitDataset(dataset, k);


    // generate blocks with a training set which is all folds except i. using i as the test dataset.
    for (int i = 0; i < k; i++) {

        // trainingData will store all folds except the i-th as training data.
        vector<vector<vector<float>>> trainingData(NUM_CLASSES);

        // Loop through all folds except i and accumulate points by class.
        for (int fold = 0; fold < k; fold++) {
            if (fold == i) continue; // skip test fold

            for (int cls = 0; cls < NUM_CLASSES; cls++) {
                // Append all points from kFolds[fold][cls] to trainingData[cls]
                trainingData[cls].insert(trainingData[cls].end(), kFolds[fold][cls].begin(), kFolds[fold][cls].end());
            }
        }

        // The test dataset for this iteration is simply fold i.
        vector<vector<vector<float>>> testData = kFolds[i];

        // now that our data is set up with training and testing, we simply do business as usual. we are going to do our LDA on the train data, then just do our block generation and simplification
        // Run LDA on the training data.
        vector<vector<float>>bestVectors;

        // Initialize indexes for each class
        vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
        vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);

        computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);


        cout << "----------------------------FOLD " << (i + 1) << " RESULTS----------------------------------" << endl;

        // ------------------------------------------
        // GENERATING BLOCKS BUSINESS AS USUAL
        vector<HyperBlock> hyperBlocks;

        IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
        cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
        cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << endl;

        vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
        int totalPoints = 0;
        for (const auto &c : trainingData)
            totalPoints += c.size();
        cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
        cout << "Ran simplifications: " << result[0] << " Times" << endl;
        cout << "We had: " << totalPoints << " points" << endl;

        testAccuracyOfHyperBlocks(hyperBlocks, testData);
    } // end of one train/test loop
}

// -------------------------------------------------------------------------
// Asynchronous mode: run when argc >= 2
int runAsync(int argc, char* argv[]) {
    // Local variables for async mode
    string normalizedSaveFile;
    string hyperBlocksImportFileName;
    string trainingDataFileName;
    string testingDataFileName;
    string hyperBlocksExportFileName;

    // 3-D DATASETS
    vector<vector<vector<float>>> testData;
    vector<vector<vector<float>>> trainingData;

    // Normalization vectors (will be resized later)
    vector<float> minValues;
    vector<float> maxValues;

    // Store our HyperBlocks
    vector<HyperBlock> hyperBlocks;

    // Ultra confusion matrix
    vector<vector<long>> ultraConfusionMatrix;

    // Variables to be set by LDA

    if (argc > 3) {
        cout << "TOO MANY ARGUMENTS!" << endl;
        exit(1);
    }

    if (argc == 3) {
        // Set a global or externally-declared variable
        COMMAND_LINE_ARGS_CLASS = stoi(argv[2]);
        cout << "Running on class index " << COMMAND_LINE_ARGS_CLASS << endl;
    }

    // Process training data from file provided as first argument
    trainingData = DataUtil::dataSetup(argv[1], CLASS_MAP, CLASS_MAP_INT);
    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES : " << NUM_CLASSES << endl;

    // Resize normalization vectors based on FIELD_LENGTH
    minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
    maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());

    DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
    DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

    // Run LDA on the training data.
    vector<vector<float>>bestVectors;
    // Initialize indexes for each class
    vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);
    computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);

    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
    cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
    cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << endl;

    vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    cout << "Ran simplifications: " << result[0] << " Times" << endl;
    cout << "We had: " << totalPoints << " points\n";

     DataUtil::saveBasicHBsToCSV(hyperBlocks, "AsyncBlockOutput", FIELD_LENGTH);
    return 0;
}

// -------------------------------------------------------------------------
// Interactive mode: run when argc < 2
void runInteractive() {
    // Local variables for interactive mode
    string normalizedSaveFile;
    string hyperBlocksImportFileName;
    string trainingDataFileName;
    string testingDataFileName;
    string hyperBlocksExportFileName;
    vector<vector<vector<float>>> testData;
    vector<vector<vector<float>>> trainingData;

    vector<float> minValues;
    vector<float> maxValues;

    vector<HyperBlock> hyperBlocks;

    vector<vector<long>> ultraConfusionMatrix;

    vector<vector<float>> bestVectors;
    vector<vector<int>> bestVectorsIndexes;
    vector<int> eachClassBestVectorIndex;

    bool running = true;
    int choice;
    while (running) {
        PrintingUtil::displayMainMenu();
        cin >> choice;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        switch (choice) {
            case 1: { // IMPORT TRAINING DATA
                cout << "Enter training data filename: " << endl;
                system("ls datasets");
                getline(cin, trainingDataFileName);

                string fullPath = "datasets" + string(PATH_SEPARATOR) + trainingDataFileName;

                CLASS_MAP.clear();
                CLASS_MAP_INT.clear();

                trainingData = DataUtil::dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                // Resize normalization vectors based on FIELD_LENGTH
                minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
                maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());
                DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
                DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

                computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);


                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                cout << "Enter testing data filename: " << endl;
                system("ls datasets");
                getline(cin, testingDataFileName);
                string fullPath = "datasets" + string(PATH_SEPARATOR) + testingDataFileName;

                // clear these two maps to prevent issues when using a second test set.
                CLASS_MAP_TESTING.clear();
                CLASS_MAP_TESTING_INT.clear();

                testData = DataUtil::dataSetup(fullPath, CLASS_MAP_TESTING, CLASS_MAP_TESTING_INT);

                // Normalize and reorder testing data as needed.
                DataUtil::normalizeTestSet(testData, minValues, maxValues, FIELD_LENGTH);
                testData = DataUtil::reorderTestingDataset(testData, CLASS_MAP, CLASS_MAP_TESTING);
                PrintingUtil::waitForEnter();
                break;
            }
            case 3: { // SAVE NORMALIZED TRAINING DATA
                cout << "Enter the file to save the normalized training data to: " << endl;
                getline(cin, normalizedSaveFile);
                DataUtil::saveNormalizedVersionToCsv(normalizedSaveFile, trainingData);
                cout << "Saved normalized training data to: " << normalizedSaveFile << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 4: { // IMPORT EXISTING HYPERBLOCKS
                cout << "Enter existing hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                hyperBlocks = DataUtil::loadBasicHBsFromCSV(hyperBlocksImportFileName);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 5: { // EXPORT HYPERBLOCKS
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
                break;
            }
            case 6: { // GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    PrintingUtil::waitForEnter();
                } else {
                    hyperBlocks.clear();
                    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
                }
                cout << "Finished Generating HyperBlocks" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 7: {
                // SIMPLIFY HYPERBLOCKS
                vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
                int totalPoints = 0;

                for (const auto &c : trainingData) totalPoints += c.size();

                cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
                cout << "We got a final total of: " << hyperBlocks.size() << " blocks." << endl;
                cout << "We had: " << totalPoints << " points of training data\n";
                PrintingUtil::waitForEnter();
                break;
            }
            case 8: { // TEST HYPERBLOCKS ON DATASET
                cout << "Testing hyperblocks on testing dataset" << endl;
                ultraConfusionMatrix = testAccuracyOfHyperBlocks(hyperBlocks, testData);
                PrintingUtil::waitForEnter();
                break;
            }
            case 9: {
                runKFold(trainingData);
                PrintingUtil::waitForEnter();
                break;
            }
            case 10: {
                running = false;
                break;
            }
            default: {
                cout << "\nInvalid choice. Please try again." << endl;
                PrintingUtil::waitForEnter();
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main entry point: choose mode based on argc.
int main(int argc, char* argv[]) {

    // Command line input mode, allows you to specify in command line what to do
    if (argc >= 2)
        return runAsync(argc, argv);

    // Interactive input loop, options to import data, train, test, save, etc
    runInteractive();
    return 0;
}