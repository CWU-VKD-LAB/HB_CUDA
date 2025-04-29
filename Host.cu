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
#include <optional>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <filesystem>

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


void evaluateOneToOneHyperBlocks(const std::vector<std::vector<HyperBlock>>& oneToOneHBs,const std::vector<std::vector<std::vector<float>>>& testSet,const std::vector<std::pair<int, int>>& classPairs, int numClasses);

/**
 * This is the OneToSome based method of HyperBlocks.
 *
 * It will generate hyperblocks for some ordering of the classes.
 * It should be used as a sieve.
 *
    // Based on ordering, each should train on only the ones in front of them.
    // Ex 0, 1, 2, 3, 4, 5
    // 1st round =>  0 : 1, 2, 3, 4, 5
    // 2nd round =>  1 : 2, 3, 4, 5
    // 3rd round =>  2 : 3, 4, 5
 *
 */
vector<vector<HyperBlock>> oneToSomeHyper(const vector<int>& ordering, const vector<vector<vector<float>>>& trainSet, vector<int> ecBestVecIdx) {
    vector<vector<HyperBlock>> oneToSomeBlocks;
    vector<HyperBlock> tempHBs;

    cout << ordering.size() << endl;
    cout << trainSet.size() << endl;
    cout << ecBestVecIdx.size() << endl;

    const int numClasses = trainSet.size();

    for(int i = 0; i < numClasses; i++) {
        cout << "Training Class (REAL LABEL): " << CLASS_MAP_INT[ordering[i]] << "  " << numClasses - i << " more classes." << endl;
        tempHBs.clear();
        // Train with this class as first class, and ALL others (remaining) as the second class.
        vector<vector<vector<float>>> trainingData(2);
        trainingData[0] = trainSet[ordering[i]];

        // Add the training data for each of the remaining classes. (flatten each and put in "point-wise")
        for(int j = i + 1; j < numClasses; j++) {
            for(const auto& point : trainSet[ordering[j]]) {
                trainingData[1].push_back(point);
            }
        }
        vector<int> bestVecs(2);
        bestVecs[0] = ecBestVecIdx[ordering[i]];
        bestVecs[1] = ecBestVecIdx[ordering[i]]; // won't really be used since only gen for first class, so set to same
        int genForClass = 0;

        // Now we should call
        IntervalHyperBlock::generateHBs(trainingData, tempHBs, bestVecs, FIELD_LENGTH, genForClass); // only go to class "0" for generating blocks.
        oneToSomeBlocks.push_back(tempHBs);

        // Set the HBs generated to have the correct class number.
        for(auto& hb : oneToSomeBlocks[i]) {
            hb.classNum = ordering[i];
        }
    }


    return oneToSomeBlocks;
}

/* If we have classes a, b, c. then we should generate blocks to change this to a multi-step 2 class problem
   we will make blocks for a pair at a time.

   a vs b, a vs c, b vs c.

   We will now be able to pass a test set point to be evaluated by each set INDEPENDENTLY.
   We will either treat it as a sieve, ex use the set of blocks that is LEAST over-generalized.
   Future Work: or it can be treated as a voting type of system ex point p gets evaluated on ALL sets
   if it scores a, c, c for the 3 pairs specified above we might want to put it in class c.

   https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
*/
vector<vector<HyperBlock>> oneToOneHyper(const vector<vector<vector<float>>>& trainingData, vector<int> eachClassBestVectorIndex, std::vector<pair<int,int>> &classPairs){
      vector<vector<HyperBlock>> oneToOneHyperBlocks;
      vector<HyperBlock> pairHyperBlocks;
      const int numClasses = trainingData.size();
      cout << "Num Classes" << endl;

    // Make the call to interval hyper FOR EACH set
      for(int i = 0; i < numClasses; i++){
          for(int j = i + 1; j < numClasses; j++){
              pairHyperBlocks.clear();
              cout << i << ", " << j << endl;
              vector<vector<vector<float>>> binaryTrainingData(2);
              binaryTrainingData[0] = trainingData[i];
              binaryTrainingData[1] = trainingData[j];

              // Pass binary training data in . ex class 0 and 1.
              IntervalHyperBlock::generateHBs(binaryTrainingData, pairHyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

              // Copy all the newly added blocks to the right index in the fin
              oneToOneHyperBlocks.push_back(pairHyperBlocks);
              classPairs.emplace_back(i, j);
              cout << "pDone" << classPairs.size() << endl;
          }
      }

      cout << "Done generating on-to-one HBs, now we are setting their classes to be correct..." << endl;
      // Now what we do.
      for(int i = 0; i < oneToOneHyperBlocks.size(); i++){
         for(int j = 0; j < oneToOneHyperBlocks[i].size(); j++){
            HyperBlock& hb = oneToOneHyperBlocks[i].at(j);

            hb.classNum = (hb.classNum == 0) ? classPairs[i].first : classPairs[i].second;
         }
      }

    return oneToOneHyperBlocks;
}











/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
float testAccuracyOfHyperBlocks(std::vector<HyperBlock>& hyperBlocks, std::vector<std::vector<std::vector<float>>> &testSet, std::vector<std::vector<std::vector<float>>> &trainingSet, std::vector<int> order){

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
for (int i = 0; i < NUM_CLASSES; i++) {
    for (int pnt = 0; pnt < testSet[i].size(); pnt++) {
        int assignedClass = -1;

        // Use classOrder to prioritize better-separated classes
        for (int rank = 0; rank < NUM_CLASSES; rank++) {
            int candidateClass = order[rank];

            // If this point falls into a block of candidateClass
            if (pointsNotClassified[i][pnt][candidateClass] > 0) {
                assignedClass = candidateClass;
                break; // Assign to the highest-ranked class it overlaps
            }
        }

        if (assignedClass != -1) {
            regularConfusionMatrix[i][assignedClass]++;
        } else {
            // Still unclassified → add to kNN fallback
            unclassifiedPointVec[i].push_back(testSet[i][pnt]);
        }
    }
}


    cout << "\n\n\n\n" << endl;
    cout << "============================ REGULAR CONFUSION MATRIX ==================" << endl;
    PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END CONFUSION MATRIX ======================" << endl;

	cout << "Any point was inside" << anyPointWasInside <<  endl;


    std::cout << "\n\n\n\n" << std::endl;
    std::cout << "============================ K-NN CONFUSION MATRIX ==================" << std::endl;

    // Find the closest actual cases to the edge of each block.
    for(HyperBlock& hb: hyperBlocks){
        hb.tameBounds(trainingSet);
    }

    int k = 3;
    //std::vector<std::vector<long>> secondConfusionMatrix = Knn::closeToInkNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
    // std::vector<std::vector<long>> secondConfusionMatrix = Knn::closestBlock(unclassifiedPointVec, hyperBlocks, NUM_CLASSES);
    std::vector<std::vector<long>> secondConfusionMatrix = Knn::pureKnn(unclassifiedPointVec, trainingSet,  k, NUM_CLASSES);
    //std::vector<std::vector<long>> secondConfusionMatrix = Knn::bruteMergable(unclassifiedPointVec, trainingSet, hyperBlocks, 5, NUM_CLASSES);

    PrintingUtil::printConfusionMatrix(secondConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END K-NN MATRIX ======================" << endl;

    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }

    cout << "\n\n\n\n" << endl;
    cout << "============================ DISTINCT POINT CONFUSION MATRIX ==================" << endl;
    float accuracy = PrintingUtil::printConfusionMatrix(regularConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END DISTINCT POINT MATRIX ======================" << endl;
    cout << "\n\n\n\n" << endl;

    return accuracy;
}





/* This function computes the LDA ordering for a given training dataset.
 * It sets up the bestVectors, bestVectorsIndexes, and eachClassBestVectorIndex.
 * best vectors is the weights of each coefficient from the LDF function
 * bestVectorsIndexes is just the indexes that correspond to those weights from the function, since we are sorting them
 * eachClassBestVectorIndex is the one best attribute for each class, we sort by this when generating blocks, and it helps a bit.
*
 * returns the class accuracy ordering so we can use the "sift approach" for HBs ex {0, 4, 2, 1, 5, 6, 7, 8, 9} for MNIST classes 0 is best seperable, 9 would be worst.
 */
std::vector<int> computeLDAOrdering(const vector<vector<vector<float>>>& trainingData, vector<vector<float>>& bestVectors, vector<vector<int>>& bestVectorsIndexes, vector<int>& eachClassBestVectorIndex) {
    // Run LDA on the training data.
    std::pair<std::vector<std::vector<float>>, std::vector<int>> result = linearDiscriminantAnalysis(trainingData);
    std::vector<int> classOrder;
    bestVectors = result.first;

    // Accuracy ordering of the LDA on the training data.
    classOrder = result.second;

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

    return classOrder;
}

void runKFold(vector<vector<vector<float>>> &dataset, bool oneToMany, std::vector<std::pair<int,int>>& classPairs) {
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

    // stats trackers for cross folds.
    float acc = 0.0f;
    int blockCount = 0;
    int cCount = 0;

    // generate blocks with a training set which is all folds except i. using i as the test dataset.
    for (int i = 0; i < k; i++) {
        // trainingData will store all folds except the i-th as training data.
        vector<vector<vector<float>>> trainingData(NUM_CLASSES);

        // Loop through all folds except i and accumulate points by class.
        for (int fold = 0; fold < k; fold++) {
            if (fold == i) continue; // skip test fold

            // build our training data
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

        if (oneToMany) {
            // ------------------------------------------
            // GENERATING BLOCKS BUSINESS AS USUAL
            vector<HyperBlock> hyperBlocks;

            IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
            //cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
            //cout << "GENERATED WITH " << hyperBlocks.size() << " BLOCKS" << endl;
            vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);

            int totalPoints = 0;
            for (const auto &c : trainingData)
                totalPoints += c.size();

            //cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS AFTER REMOVING USELESS BLOCKS!" << endl;
            //cout << "After removing useless blocks we have: " << result[1] << " clauses\n";

            int clauseCount = 0;
            for (const auto &hb : hyperBlocks) {
                for (int a = 0; a < FIELD_LENGTH; a++) {
                    if (hb.minimums[a][0] == 0.0f && hb.maximums[a][0] == 1.0f)
                        continue;

                    clauseCount++;
                }
            }


            //       acc += testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData);
            blockCount += hyperBlocks.size();
            cCount += clauseCount;

        } // end of one train/test loop
        else {
            vector<vector<HyperBlock>> oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex, classPairs);
            evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairs, NUM_CLASSES);
            //IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
        }


    }
    cout << "OVERALL ACCURACY " << float(acc) / float(k) << endl;
    cout << "Average block count " << float(blockCount) / float (k) << endl;
    cout << "Average clause count " << float(cCount) / float (k) << endl;
    
}



float evaluateOneToSomeHBs(const vector<vector<HyperBlock>>& oneToSomeBlocks, const vector<vector<vector<float>>> testData) {
    vector<vector<int>> confusionMatrix(NUM_CLASSES, vector<int>(NUM_CLASSES, 0));
    int totalPoints = 0;
    for(const auto& c : testData) {
        totalPoints += c.size();
    }

    int incorrect = 0;
    int correct = 0;
    int pointsTested = 0;
    // Go through the classes
    for(int i = 0; i < NUM_CLASSES; i++) {
        for(int j = 0; j < testData[i].size(); j++) {
            pointsTested++;
            const auto& point = testData[i][j];
            bool classified = false;

            for(int sieveLvl = 0; sieveLvl < NUM_CLASSES; sieveLvl++) {
                for(const auto& hb : oneToSomeBlocks[sieveLvl]) {
                    if(hb.inside_HB(point.size(), point.data())) {
                        classified = true;

                        // If it is of the wrong class.
                        if(hb.classNum != i) {
                            incorrect++;
                        }
                        else {
                            correct++;
                        }

                        confusionMatrix[i][hb.classNum]++;
                        break;
                    }
                }
                if (classified) break; // do not keep checking once classified
            }
        }
    }

    for(const auto& row : confusionMatrix) {
        for(int i = 0; i < row.size(); i++) {
            cout << row[i] << " ";
        }
        cout << endl;
    }

    // Calculate the coverage.
    float coverage = static_cast<float>(incorrect + correct) / static_cast<float>(totalPoints);
    float acc = static_cast<float>(correct) / static_cast<float>(correct + incorrect);

    cout << "Coverage %: " << coverage * 100.0f << "%" << endl;
    cout << "Coverage total: " << (incorrect + correct) << " out of " << totalPoints << endl;
    cout << "Overall Accuracy: " << acc * 100.0f << "%" << endl;
    cout << "Total number of points tested: " << pointsTested << endl;
    return acc;
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

    cout << "RUNNING LDA" << endl;
    // Run LDA on the training data.
    vector<vector<float>>bestVectors;
    // Initialize indexes for each class
    vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);
    computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);

    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
    cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
    cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << endl;
    cout << "BEGINNING SIMPLIFICATIONS" << endl;

    string nonSimplified = string("NonSimplifiedBlocksClass") + to_string(COMMAND_LINE_ARGS_CLASS);
    DataUtil::saveBasicHBsToCSV(hyperBlocks, nonSimplified, FIELD_LENGTH);

    vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    cout << "Ran simplifications: " << result[0] << " Times" << endl;
    cout << "We had: " << totalPoints << " points\n";

    string simplified = string("SimplifiedBlocks") + to_string(COMMAND_LINE_ARGS_CLASS);
    DataUtil::saveBasicHBsToCSV(hyperBlocks, simplified, FIELD_LENGTH);
    return 0;
}


void evaluateOneToOneHyperBlocks(
    const std::vector<std::vector<HyperBlock>>& oneToOneHBs,
    const std::vector<std::vector<std::vector<float>>>& testSet,
    const std::vector<std::pair<int, int>>& classPairs,
    int numClasses
) {
    int totalPoints = 0;
    int correctPoints = 0;

    std::vector<std::vector<int>> confusion(numClasses, std::vector<int>(numClasses, 0));
    std::vector<int> correctPerClass(numClasses, 0);
    std::vector<int> totalPerClass(numClasses, 0);


    for (int actualClass = 0; actualClass < numClasses; ++actualClass) {
        for (const auto& point : testSet[actualClass]) {
            std::vector<int> votes(numClasses, 0);

            // Run all pairwise comparisons
            for (int i = 0; i < numClasses; ++i) {
                for (int j = i + 1; j < numClasses; ++j) {
                    // Find the correct index using classPairs
                    int findPairIndex = -1;
                    for (int idx = 0; idx < classPairs.size(); ++idx) {
                        if ((classPairs[idx].first == i && classPairs[idx].second == j) ||
                            (classPairs[idx].first == j && classPairs[idx].second == i)) {
                            findPairIndex = idx;
                            break;
                            }
                    }
                    if (findPairIndex == -1) {
                        std::cerr << "Error: could not find block set for classes " << i << " and " << j << "\n";
                        continue;
                    }

                    const auto& pairHBs = oneToOneHBs[findPairIndex];

                    bool inClassI = false, inClassJ = false;

                    for (const auto& block : pairHBs) {
                        if (block.classNum == i && block.inside_HB(point.size(), point.data())) {
                            inClassI = true;
                        }
                        if (block.classNum == j && block.inside_HB(point.size(), point.data())) {
                            inClassJ = true;
                        }
                    }

                    if (inClassI && !inClassJ) votes[i]++;
                    else if (!inClassI && inClassJ) votes[j]++;
                    else if (inClassI && inClassJ) {
                        votes[i]++;
                        votes[j]++;
                    }
                }
            }

            // Majority vote
            int predictedClass = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
            if (votes[predictedClass] == 0) {
                // No vote: optionally handle as unclassified
                continue;
            }

            confusion[actualClass][predictedClass]++;
            totalPerClass[actualClass]++;
            if (predictedClass == actualClass) {
                correctPerClass[actualClass]++;
                correctPoints++;
            }
            totalPoints++;
        }
    }

    // Output
    std::cout << "\nConfusion Matrix:\n";
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            std::cout << confusion[i][j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-Class Accuracy:\n";
    for (int i = 0; i < numClasses; ++i) {
        float acc = totalPerClass[i] ? static_cast<float>(correctPerClass[i]) / totalPerClass[i] : 0.0f;
        std::cout << "Class " << i << ": " << acc * 100.0f << "%\n";
    }

    std::cout << "\nOverall Accuracy: " << (static_cast<float>(correctPoints) / totalPoints) * 100.0f << "%\n";
}

/**
 *  We will attempt to find some "best" ordering of the classes by training normal HBs, then calling this function,
 *  this function will return the order that we believe is best to be used by the one-to-some method.
 *
 *  This is similar to using the LDA accuracies to sort the classes as best -> worst accuracy, but
 *  it is closer to what we should actually expect our blocks to do. The LDA is inherently different from
 *  how our blocks classify, and thus this should hopefully give us a better order.
 *
 *  Order will be determined by seeing which classes HyperBlocks swallow the most points from the wrong classes.
 *  (we also keep track of how many times a point of a class falls into wrong blocks)
 *
 * @param validationData The validation data set.
 * @param hyperBlocks The HyperBlocks which were trained WITHOUT using the validation points.
 * @return a vector<int> which is the recommended order to create one-to-some blocks in.
 */
vector<int> findOneToSomeOrder(vector<vector<vector<float>>>& validationData, vector<HyperBlock>& hyperBlocks) {
    int numClasses = validationData.size();

    // Keep track of how many times a block from each class picks up points from other classes.
    vector<int> hbClassOverclaims(numClasses, 0);  // based on the hyperblock.class
    // This one will keep track of how many times A POINT from a class falls into multiple blocks.
    vector<vector<int>> sneakyPointCount(numClasses, vector<int>(numClasses, 0));

    for(int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < validationData[i].size(); ++j) {
            const auto& point = validationData[i][j];

            for(const auto& block : hyperBlocks) {
                if(block.inside_HB(point.size(), point.data())) {
                    if(i != block.classNum) { // Incorrect classification.
                        hbClassOverclaims[block.classNum]++;    // How many times HBs classify the WRONG point!
                        sneakyPointCount[i][block.classNum]++;  ;  // How many times POINTS of a class fall into wrong class HBs
                    }
                }
            }
        }
    }

    // Now we should look at the numbers for hbClassOverclaims to analyze which ordering would be the best.
    // Save raw counts (temps)
    std::ofstream outFile("hbClassOverclaims.csv"); if (!outFile.is_open()) {std::cerr << "Error opening file for writing: " << "hbClassOverclaims.csv" << std::endl;}for (size_t i = 0; i < hbClassOverclaims.size(); ++i) {outFile << i << "," << hbClassOverclaims[i] << "\n";}
    std::ofstream outFile2("sneakyPointCount.csv");if (!outFile2.is_open()) {std::cerr << "Error opening file for writing: " << "sneakyPointCount.csv" << std::endl;}for (size_t i = 0; i < sneakyPointCount.size(); ++i) {for (size_t j = 0; j < sneakyPointCount[i].size(); ++j) {outFile2 << sneakyPointCount[i][j];if (j + 1 != sneakyPointCount[i].size())outFile2 << ",";}outFile2 << "\n";}


    // Build mistake counts
    vector<std::pair<int, int>> pointMistakes;   // (count, class)
    vector<std::pair<int, int>> blockMistakes;   // (count, class)
    vector<std::pair<int, int>> combinedMistakes; // (count, class)

    for (int c = 0; c < numClasses; ++c) {
        int pointMistakeSum = 0;
        for (int other = 0; other < numClasses; ++other) {
            pointMistakeSum += sneakyPointCount[c][other];
        }

        int blockMistake = hbClassOverclaims[c];
        int combined = pointMistakeSum + blockMistake;

        pointMistakes.emplace_back(pointMistakeSum, c);
        blockMistakes.emplace_back(blockMistake, c);
        combinedMistakes.emplace_back(combined, c);
    }

    // Now sort each list
    auto sorter = [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first; // sort by mistake counts ascending
    };

    std::sort(pointMistakes.begin(), pointMistakes.end(), sorter);
    std::sort(blockMistakes.begin(), blockMistakes.end(), sorter);
    std::sort(combinedMistakes.begin(), combinedMistakes.end(), sorter);

    // Extract orders
    vector<int> pointOrder, blockOrder, combinedOrder;
    for (auto& p : pointMistakes) pointOrder.push_back(p.second);
    for (auto& p : blockMistakes) blockOrder.push_back(p.second);
    for (auto& p : combinedMistakes) combinedOrder.push_back(p.second);

    // Save the ordering. (temp)
    std::ofstream orderFile("orderings.csv"); if (!orderFile.is_open()) { std::cerr << "Error opening file for writing orderings.csv" << std::endl; } else { orderFile << "pointOrder:"; for (size_t i = 0; i < pointOrder.size(); ++i) { orderFile << " " << pointOrder[i]; if (i + 1 != pointOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile << "blockOrder:"; for (size_t i = 0; i < blockOrder.size(); ++i) { orderFile << " " << blockOrder[i]; if (i + 1 != blockOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile << "combinedOrder:"; for (size_t i = 0; i < combinedOrder.size(); ++i) { orderFile << " " << combinedOrder[i]; if (i + 1 != combinedOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile.close(); }

    return blockOrder;
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
    vector<vector<vector<float>>> validationData;

    vector<float> minValues;
    vector<float> maxValues;

    vector<HyperBlock> hyperBlocks;

    vector<vector<long>> ultraConfusionMatrix;

    vector<vector<float>> bestVectors;
    vector<vector<int>> bestVectorsIndexes;
    vector<int> eachClassBestVectorIndex;

    // Class ordering
    vector<int> order;
    vector<vector<HyperBlock>> oneToOneBlocks;
    std::vector<std::pair<int, int>> classPairsOut;

    vector<vector<HyperBlock>> oneToSomeBlocks;


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
                #ifdef _WIN32
                    system("dir DATASETS");
                #else
                    system("ls DATASETS");
                #endif
                getline(cin, trainingDataFileName);

                string fullPath = "DATASETS" + string(PATH_SEPARATOR) + trainingDataFileName;

                CLASS_MAP.clear();
                CLASS_MAP_INT.clear();

                trainingData = DataUtil::dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                // Resize normalization vectors based on FIELD_LENGTH
                minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
                maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());
                DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
                DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

                order = computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);


                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                cout << "Enter testing data filename: " << endl;
                system("ls DATASETS");
                getline(cin, testingDataFileName);
                string fullPath = "DATASETS" + string(PATH_SEPARATOR) + testingDataFileName;

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

				//try_expand_blocks_and_save(hyperBlocks, trainingData);

                for(HyperBlock& hb: hyperBlocks){
                  hb.find_avg_and_size(trainingData);
                }

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

                //for(HyperBlock& hb: hyperBlocks) hb.findSize(trainingData);

                cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
                cout << "We got a final total of: " << hyperBlocks.size() << " blocks." << endl;
                cout << "We had: " << totalPoints << " points of training data\n";
                PrintingUtil::waitForEnter();
                break;
            }
            case 8: { // TEST HYPERBLOCKS ON DATASET
                std::cout << "Testing hyperblocks on testing dataset" << std::endl;
                testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData, order);
                PrintingUtil::waitForEnter();
                break;
            }
            case 9: {
                evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairsOut, NUM_CLASSES);

                PrintingUtil::waitForEnter();
                break;
            }
            case 10: {
                runKFold(trainingData, true, classPairsOut);
                PrintingUtil::waitForEnter();
                break;
            }
            case 11: {
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    PrintingUtil::waitForEnter();
                } else {
                    oneToOneBlocks.clear();
                    oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex, classPairsOut);
                }

                cout << "Finished Generating 1-1 HyperBlocks" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 12: {
                //      oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex);

                // Import 1-1 Hyperblocks
                cout << "Enter 1-1 Hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                oneToOneBlocks = DataUtil::loadOneToOneHBsFromCSV(hyperBlocksImportFileName, classPairsOut);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;

                /* Might still want this but has to be wrapped in loop through each pair.
                for(HyperBlock& hb: hyperBlocks){
                    hb.find_avg_and_size(trainingData);
                }

                blockSizeDistribution(hyperBlocks);
                */

                PrintingUtil::waitForEnter();
                break;
            }
            case 13: {
                // Export 1-1 Hyperblocks
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveOneToOneHBsToCSV(oneToOneBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
                break;
            }
            case 14: {
                // RUN K FOLD FOR THE ONE TO ONE TYPE BLOCKS.
                runKFold(trainingData, false, classPairsOut);
                PrintingUtil::waitForEnter();
                break;
            }
            case 15: {

                // Split out the validation data we will use to train the ordering.
                float validationPercent = 0.10f;
                int seed = 42;
                //float percentTrainData = .0f;

                //DataUtil::splitTrainTestByPercent(trainingData, testData, percentTrainData);
                std::vector<std::vector<std::vector<float>>> fullTrainingData = trainingData;
                DataUtil::createValidationSplit(trainingData, validationData, validationPercent, seed);

                // Generate the "basic hyperblocks" that we will test against validation
                hyperBlocks.clear();
                IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

                int numVal = 0;
                for(int i = 0; i < validationData.size(); i++) { numVal += validationData[i].size(); }
                cout << "Validation is " << numVal << " points" << endl;

                // Now we want to test this based on the validation set to see which ordering to use.
                order = findOneToSomeOrder(validationData, hyperBlocks );
                oneToSomeBlocks.clear();

                // Train the oneToSome Blocks
                auto start = std::chrono::high_resolution_clock::now();

                int numTrain = 0;
                for(int i = 0; i < fullTrainingData.size(); i++) { numTrain += fullTrainingData[i].size(); }
                oneToSomeBlocks = oneToSomeHyper(order, fullTrainingData, eachClassBestVectorIndex);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end - start;

                for(int i = 0; i < oneToSomeBlocks.size(); i++) {
                    const auto& blockSet = oneToSomeBlocks[i];
                    DataUtil::saveBasicHBsToCSV(blockSet, std::to_string(order[i]) + "replaceable.csv", FIELD_LENGTH);
                }

                int numTest = 0;
                for(int i = 0; i < testData.size(); i++) { numTest += testData[i].size(); }


                cout << "Testing is " << numTest << " points" << endl;
                cout << "Validation is " << numVal << " points" << endl;
                cout << "Training is " << numTrain << " points" << endl;

                evaluateOneToSomeHBs(oneToSomeBlocks, testData);

                cout << "Finished Generating one to Some blocks." << endl;
                std::cout << "Elapsed time: " << diff.count() << " seconds\n";

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