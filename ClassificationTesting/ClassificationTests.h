//
// Created by Ryan Gallagher on 5/5/25.
//
#ifndef CLASSIFICATIONTESTS_H
#define CLASSIFICATIONTESTS_H

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <iterator>
#include "../knn/Knn.h"
#include "../data_utilities/StatStructs.h"

#include "../hyperblock/HyperBlock.h"
using namespace std;

class ClassificationTests {
public:
    enum classifiers {

        // regular classification mode. this is the whole point of the program
        // everything else comes after this
        HYPERBLOCKS = 1,
        PURE_KNN = 2,                   // VERY HIGH ACCURACY, BUT SHOULDN'T BE USED OUTSIDE OF IMAGE DATA.
        CLOSEST_BLOCK = 3,             // takes the block which our point has least distance to the bounds. just a simple addition of the amount you missed each interval by
        BRUTE_MERGABLE = 4,
        THRESHOLD_KNN = 5,             // BEST FOR EXPLAINABILITY ON MOST datasets!!!!!! JUST LIKE A REGULAR KNN, BUT USES STANDARD DEVIATION BASED SIMILARITY SCORES TO COMPARE POINTS
        OLD_KNN = 7,                   // nearest HB  based
        MERGABLE_KNN = 8,               // determines which block is "most mergeable" to this point. And uses that classification
        PRECISION_WEIGHTED = 9          // Uses HBs with a precison weighting mechanism. Allows for HB class X to partially vote for any class based on validation scores.
    };

    // main function which takes in points we need to classify, and classifies them using whichever mode we are in.
    static vector<vector<long>> buildConfusionMatrix(vector<HyperBlock> &hyperBlocks, const vector<vector<vector<float>>> &trainingData, vector<vector<vector<float>>> &pointsToClassify, int classificationMode, vector<vector<vector<float>>> &pointsWeCantClassify, const int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries, int k = 5, float threshold = 0.25);
    static pair<int, vector<BlockInfo>> precisionWeightedHBs(const vector<float> &point, vector<vector<vector<float>>>& testData, vector<HyperBlock>& hyperBlocks, int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries);

    static pair<int, vector<BlockInfo>> predictWithHBs(const vector<HyperBlock> &hyperBlocks, const vector<float> &point, int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries);

};



#endif //CLASSIFICATIONTESTS_H
