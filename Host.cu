
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
#include "./cuda_util/CudaUtil.h"
#include "./hyperblock_generation/MergerHyperBlock.cuh"
#include "./hyperblock/HyperBlock.h"
#include "./interval_hyperblock/IntervalHyperBlock.h"
#include "./knn/Knn.h"
#include "./screen_output/PrintingUtil.h"
#include "./data_utilities/DataUtil.h"
#include "./simplifications/Simplifications.h""

#ifdef _WIN32
    const std::string PATH_SEPARATOR = "\\";
#else
    const std::string PATH_SEPARATOR = "/";
#endif

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset
int COMMAND_LINE_ARGS_CLASS = -1;

std::map<std::string, int> CLASS_MAP;
std::map<std::string, int> CLASS_MAP_TESTING;

std::map<int, std::string> CLASS_MAP_INT;
std::map<int, std::string> CLASS_MAP_TESTING_INT;


/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
std::vector<std::vector<long>> testAccuracyOfHyperBlocks(std::vector<HyperBlock>& hyperBlocks, std::vector<std::vector<std::vector<float>>> testSet){

  	// Keep track of which points were never inside of a block, when a point is classifed we increment the map internal std::vectors correct positon
    // there should be CLASS_NUM unordered_maps or just hashmaps, in each will hold a std::vector<point_index, std::vector<int> of len(class_num)>
    std::vector<std::unordered_map<int, std::vector<int>>> pointsNotClassified(CLASS_MAP.size());

    // Go through each class
    for(int cls = 0; cls < NUM_CLASSES; cls++){
        // Put the index of each point in each class into a set, this is how we will track which points were never classified.
        for(int j = 0; j < testSet[cls].size(); j++){
            pointsNotClassified[cls][j] = std::vector<int>(NUM_CLASSES);
        }
    }

	// Make a n x n matrix for the confusion matrix
	std::vector<std::vector<long>> ultraConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));
    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    bool anyPointWasInside = false;

    // Go through all the blocks
	for(int hb = 0; hb < hyperBlocks.size(); hb++){
        HyperBlock& currBlock = hyperBlocks[hb];
        // Go through all the classes in the testSet
		for(int cls = 0; cls < NUM_CLASSES; cls++){
            // go through all the points in a clases
        	for(int pnt = 0; pnt < testSet[cls].size(); pnt++){
           		const std::vector<float>& point = testSet[cls][pnt];

                if(currBlock.inside_HB(point.size(), point.data())){

					ultraConfusionMatrix[cls][currBlock.classNum]++;


                    // Go to the actual class, to the right points entry, and increment the "predicted" class (the hb it was in).
                    pointsNotClassified[cls][pnt][currBlock.classNum]++;
                }
        	}
     	}
    }

    for(int i = 0; i < NUM_CLASSES; i++){
        std::cout << pointsNotClassified[0][0][i] << std::endl;
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

           if(in > 1) numPointsInMultipleClasses++;

           if(in == 0) numPointsInNoBlocks++;
       }

       std::cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN MULTIPLE CLASSES BLOCKS: " << numPointsInMultipleClasses << std::endl;
       std::cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN NO BLOCKS: " << numPointsInNoBlocks << std::endl;
    }


    std::vector<std::vector<std::vector<float>>> unclassifiedPointVec(NUM_CLASSES, std::vector<std::vector<float>>()); // [class][pointIdx][attr]

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

    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ REGULAR CONFUSION MATRIX ==================" << std::endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    std::cout << "============================ END CONFUSION MATRIX ======================" << std::endl;

	std::cout << "Any point was inside" << anyPointWasInside <<  std::endl;

    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ K-NN CONFUSION MATRIX ==================" << std::endl;
    int k = 1;
    std::vector<std::vector<long>> secondConfusionMatrix = Knn::kNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
     PrintingUtil::printConfusionMatrix(secondConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    std::cout << "============================ END K-NN MATRIX ======================" << std::endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }

    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ DISTINCT POINT CONFUSION MATRIX ==================" << std::endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    std::cout << "============================ END DISTINCE POINT MATRIX ======================" << std::endl;
    std::cout << "\n\n\n\n" << std::endl;

    return ultraConfusionMatrix;
}


// -------------------------------------------------------------------------
// Asynchronous mode: run when argc >= 2
int runAsync(int argc, char* argv[]) {
    // Local variables for async mode
    std::string normalizedSaveFile;
    std::string hyperBlocksImportFileName;
    std::string trainingDataFileName;
    std::string testingDataFileName;
    std::string hyperBlocksExportFileName;

    // 3-D DATASETS
    std::vector<std::vector<std::vector<float>>> testData;
    std::vector<std::vector<std::vector<float>>> trainingData;

    // Normalization std::vectors (will be resized later)
    std::vector<float> minValues;
    std::vector<float> maxValues;

    // Store our HyperBlocks
    std::vector<HyperBlock> hyperBlocks;

    // Ultra confusion matrix
    std::vector<std::vector<long>> ultraConfusionMatrix;

    // Variables to be set by LDA

    if (argc > 3) {
        std::cout << "TOO MANY ARGUMENTS!" << std::endl;
        exit(1);
    }

    if (argc == 3) {
        // Set a global or externally-declared variable
        COMMAND_LINE_ARGS_CLASS = std::stoi(argv[2]);
        std::cout << "Running on class index " << COMMAND_LINE_ARGS_CLASS << std::endl;
    }

    // Process training data from file provided as first argument
    trainingData = DataUtil::dataSetup(argv[1], CLASS_MAP, CLASS_MAP_INT);
    std::cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << std::endl;
    std::cout << "NUM CLASSES : " << NUM_CLASSES << std::endl;

    // Resize normalization vectors based on FIELD_LENGTH
    minValues.assign(FIELD_LENGTH, std::numeric_limits<float>::infinity());
    maxValues.assign(FIELD_LENGTH, -std::numeric_limits<float>::infinity());

    DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
    DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

    // Run LDA on the training data.
    std::vector<std::vector<float>>bestVectors = linearDiscriminantAnalysis(trainingData);

    // Initialize indexes for each class
    std::vector<std::vector<int>> bestVectorsIndexes = std::vector<std::vector<int> >(NUM_CLASSES, std::vector<int>(FIELD_LENGTH, 0));
    std::vector<int> eachClassBestVectorIndex = std::vector<int>(NUM_CLASSES);

    // sort our vectors from the LDA by their coefficients so that we can determine an ordering for removing and sorting by best columns in generation
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < FIELD_LENGTH; j++) {
            bestVectorsIndexes[i][j] = j;
        }
        // Sort indices by absolute value of the coefficients for the current class.
        sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
             [&](int a, int b) {
                 return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
             });
        eachClassBestVectorIndex[i] = bestVectorsIndexes[i][0];
    }

    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
    std::cout << "HYPERBLOCK GENERATION FINISHED!" << std::endl;
    std::cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << std::endl;

    std::vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    std::cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    std::cout << "Ran simplifications: " << result[0] << " Times" << std::endl;
    std::cout << "We had: " << totalPoints << " points\n";

     DataUtil::saveBasicHBsToCSV(hyperBlocks, "AsyncBlockOutput", FIELD_LENGTH);
    return 0;
}

// -------------------------------------------------------------------------
// Interactive mode: run when argc < 2
void runInteractive() {
    // Local variables for interactive mode
    std::string normalizedSaveFile;
    std::string hyperBlocksImportFileName;
    std::string trainingDataFileName;
    std::string testingDataFileName;
    std::string hyperBlocksExportFileName;
    std::vector<std::vector<std::vector<float>>> testData;
    std::vector<std::vector<std::vector<float>>> trainingData;

    std::vector<float> minValues;
    std::vector<float> maxValues;

    std::vector<HyperBlock> hyperBlocks;

    std::vector<std::vector<long>> ultraConfusionMatrix;

    std::vector<std::vector<float>> bestVectors;
    std::vector<std::vector<int>> bestVectorsIndexes;
    std::vector<int> eachClassBestVectorIndex;

    bool running = true;
    int choice;
    while (running) {
        PrintingUtil::displayMainMenu();
        std::cin >> choice;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch (choice) {
            case 1: { // IMPORT TRAINING DATA
                std::cout << "Enter training data filename: " << std::endl;
                system("ls DATASETS");
                std::getline(std::cin, trainingDataFileName);

                std::string fullPath = "DATASETS" + std::string(PATH_SEPARATOR) + trainingDataFileName;
                trainingData = DataUtil::dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                FIELD_LENGTH = trainingData[0][0].size();
                NUM_CLASSES = trainingData.size();

                // Resize normalization vectors based on FIELD_LENGTH
                minValues.assign(FIELD_LENGTH, std::numeric_limits<float>::infinity());
                maxValues.assign(FIELD_LENGTH, -std::numeric_limits<float>::infinity());
                DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
                DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

                /* Run LDA on the training data.
                bestVectors = linearDiscriminantAnalysis(trainingData);

                bestVectorsIndexes = std::vector<std::vector<int>>(NUM_CLASSES, std::vector<int>(FIELD_LENGTH, 0));
                eachClassBestVectorIndex = std::vector<int>(NUM_CLASSES);

                for (int i = 0; i < NUM_CLASSES; i++) {
                    for (int j = 0; j < FIELD_LENGTH; j++) {
                        bestVectorsIndexes[i][j] = j;
                    }
                    sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
                         [&](int a, int b) {
                             return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
                         });
                    eachClassBestVectorIndex[i] = bestVectorsIndexes[i][0];
                }
                 */
                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                std::cout << "Enter testing data filename: " << std::endl;
                system("ls");
                getline(std::cin, testingDataFileName);
                testData = DataUtil::dataSetup(testingDataFileName.c_str(), CLASS_MAP_TESTING, CLASS_MAP_TESTING_INT);

                // Normalize and reorder testing data as needed.
                DataUtil::normalizeTestSet(testData, minValues, maxValues, FIELD_LENGTH);
                testData = DataUtil::reorderTestingDataset(testData, CLASS_MAP, CLASS_MAP_TESTING);
                PrintingUtil::waitForEnter();
                break;
            }
            case 3: { // SAVE NORMALIZED TRAINING DATA
                std::cout << "Enter the file to save the normalized training data to: " << std::endl;
                getline(std::cin, normalizedSaveFile);
                DataUtil::saveNormalizedVersionToCsv(normalizedSaveFile, trainingData);
                std::cout << "Saved normalized training data to: " << normalizedSaveFile << std::endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 4: { // IMPORT EXISTING HYPERBLOCKS
                std::cout << "Enter existing hyperblocks file name: " << std::endl;
                getline(std::cin, hyperBlocksImportFileName);
                hyperBlocks = DataUtil::loadBasicHBsFromCSV(hyperBlocksImportFileName);
                std::cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << std::endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 5: { // EXPORT HYPERBLOCKS
                std::cout << "Enter the file to save HyperBlocks to: " << std::endl;
                getline(std::cin, hyperBlocksExportFileName);
                DataUtil::saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
                break;
            }
            case 6: { // GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    std::cout << "\nError: Please import training data first." << std::endl;
                    PrintingUtil::waitForEnter();
                } else {
                    hyperBlocks.clear();
                    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
                }
                std::cout << "Finished Generating HyperBlocks" << std::endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 7: { // SIMPLIFY HYPERBLOCKS
                std::vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
                int totalPoints = 0;

                for (const auto &c : trainingData) totalPoints += c.size();

                std::cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
                std::cout << "Ran simplifications: " << result[0] << " Times" << std::endl;
                std::cout << "We had: " << totalPoints << " points\n";
                PrintingUtil::waitForEnter();
                break;
            }
            case 8: { // TEST HYPERBLOCKS ON DATASET
                std::cout << "Testing hyperblocks on testing dataset" << std::endl;
                ultraConfusionMatrix = testAccuracyOfHyperBlocks(hyperBlocks, testData);
                //printConfusionMatrix(ultraConfusionMatrix);
                PrintingUtil::waitForEnter();
                break;
            }
            case 9: {
                running = false;
                break;
            }
            default: {
                std::cout << "\nInvalid choice. Please try again." << std::endl;
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