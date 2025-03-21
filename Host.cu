
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
#include <queue>


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



// Source
void merger_cuda(const std::vector<std::vector<std::vector<float>>>& dataWithSkips, const std::vector<std::vector<std::vector<float>>>& allData, std::vector<HyperBlock>& hyperBlocks) {

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : allData) {
        numPoints += classData.size();
    }

    // Count blocks per class
    std::vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyperBlocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    std::vector<std::vector<HyperBlock>> resultingBlocks(NUM_CLASSES);

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

        #ifdef DEBUG
        std::cout << "Grid size: " << gridSize << std::endl;
        std::cout << "Block size: " << blockSize << std::endl;
        std::cout << "Shared memory size: " << sharedMemSize << std::endl;
        #endif

        // Allocate host memory
        std::vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        std::vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        std::vector<int> deleteFlagsC(sizeWithoutHBpoints / PADDED_LENGTH);

        int nSize = allData[classN].size();
        std::vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

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
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / PADDED_LENGTH;
        std::vector<int> seedQueue(numBlocks);
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

        std::cout << "Launched a kernel for class: " << classN << std::endl;

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
        cudaMemcpy(hyperBlockMinsC.data(), d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hyperBlockMaxesC.data(), d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(deleteFlagsC.data(), d_deleteFlags, deleteFlagsC.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // Process results
        for (int i = 0; i < hyperBlockMinsC.size(); i += PADDED_LENGTH) {
            
            if (deleteFlagsC[i / PADDED_LENGTH] == -1) continue;  // -1 is a seed block which was merged to. so it doesn't need to be copied back.

            std::vector<std::vector<float>> blockMins(FIELD_LENGTH);
            std::vector<std::vector<float>> blockMaxes(FIELD_LENGTH);
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
    for(const std::vector<HyperBlock>& classBlocks : resultingBlocks) {
      hyperBlocks.insert(hyperBlocks.end(), classBlocks.begin(), classBlocks.end());
    }
}


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
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix);
    std::cout << "============================ END CONFUSION MATRIX ======================" << std::endl;

	std::cout << "Any point was inside" << anyPointWasInside <<  std::endl;

    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ K-NN CONFUSION MATRIX ==================" << std::endl;
    std::vector<std::vector<long>> secondConfusionMatrix = Knn::kNN(unclassifiedPointVec, hyperBlocks, 5);
     PrintingUtil::printConfusionMatrix(secondConfusionMatrix);
    std::cout << "============================ END K-NN MATRIX ======================" << std::endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }

    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ DISTINCT POINT CONFUSION MATRIX ==================" << std::endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix);
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

    DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues);
    DataUtil::minMaxNormalization(trainingData, minValues, maxValues);

    // Run LDA on the training data.
    std::vector<std::vector<float>>bestVectors = linearDiscriminantAnalysis(trainingData);

    // Initialize indexes for each class
    std::vector<std::vector<int> > bestVectorsIndexes = std::vector<std::vector<int> >(NUM_CLASSES, std::vector<int>(FIELD_LENGTH, 0));
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

    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex);
    std::cout << "HYPERBLOCK GENERATION FINISHED!" << std::endl;
    std::cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << std::endl;

    std::vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    std::cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    std::cout << "Ran simplifications: " << result[0] << " Times" << std::endl;
    std::cout << "We had: " << totalPoints << " points\n";

     DataUtil::saveBasicHBsToCSV(hyperBlocks, "AsyncBlockOutput");
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
                system("ls DATASETS");  // list available DATASETS
                std::getline(std::cin, trainingDataFileName);
                // Prepend the directory (adjust PATH_SEPARATOR as needed)
                std::string fullPath = "DATASETS" + std::string(PATH_SEPARATOR) + trainingDataFileName;
                trainingData = DataUtil::dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                // Resize normalization vectors based on FIELD_LENGTH
                minValues.assign(FIELD_LENGTH, std::numeric_limits<float>::infinity());
                maxValues.assign(FIELD_LENGTH, -std::numeric_limits<float>::infinity());
                DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues);
                DataUtil::minMaxNormalization(trainingData, minValues, maxValues);

                // Run LDA on the training data.
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
                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                std::cout << "Enter testing data filename: " << std::endl;
                system("ls");
                getline(std::cin, testingDataFileName);
                testData = DataUtil::dataSetup(testingDataFileName.c_str(), CLASS_MAP_TESTING, CLASS_MAP_TESTING_INT);

                // Normalize and reorder testing data as needed.
                DataUtil::normalizeTestSet(testData, minValues, maxValues);
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
                DataUtil::saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName);
                break;
            }
            case 6: { // GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    std::cout << "\nError: Please import training data first." << std::endl;
                    PrintingUtil::waitForEnter();
                } else {
                    hyperBlocks.clear();
                    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex);
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