//
// Created by Ryan Gallagher on 5/5/25.
//

#include "ClassificationTests.h"


/*
 * takes in our hyperblocks by reference. takes in the points we need to classify. and takes in a mode, as well as a reference to another container for points.
 * the points we can't classify is for those which fall out of all HBs. this allows us to easily make a list of points we couldn't handle, and we would just call the function
 * again with another mode, to reclassify those points specifically.
 */
vector<vector<long>> ClassificationTests::buildConfusionMatrix(vector<HyperBlock> &hyperBlocks, const vector<vector<vector<float>>> &trainingData, vector<vector<vector<float>>> &testingData, int classificationMode, vector<vector<vector<float>>> &pointsWeCantClassify, const int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries,int k, float threshold) {

    vector<vector<long>> confusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));

    int totalPointsToDo = 0;
    for (const auto &classPoints : testingData) {
        totalPointsToDo += classPoints.size();
    }

    if (totalPointsToDo == 0)
        return confusionMatrix;

    // go through all classes
    for(int cls = 0; cls < NUM_CLASSES; cls++) {

        // all points each class. we are going to use whichever classifier we have called our function with to build the matrix
        for(int point = 0; point < testingData[cls].size(); point++) {

            // get our reference to the point
            const auto &p = testingData[cls][point];
            int predictedClass = -1;
            vector<BlockInfo> blockHits{};
            pair<int, vector<BlockInfo>> prediction;

            // call whichever classifier we are using now.
            switch (classificationMode) {

                // regular old HBs case. this is the original case, where we just take the classification of whichever block it falls in.
                case HYPERBLOCKS:
                    // call the function which predicts one point
                    prediction = predictWithHBs(hyperBlocks, p, NUM_CLASSES, pointSummaries);
                    predictedClass = prediction.first;
                    blockHits = prediction.second;
                    break;

                case PURE_KNN:
                    predictedClass = Knn::pureKnn(p, trainingData, NUM_CLASSES, k);
                    break;

                case CLOSEST_BLOCK:
                    predictedClass = Knn::closeToInkNN(p, hyperBlocks, k, NUM_CLASSES);
                    break;

                case BRUTE_MERGABLE:
                    predictedClass = Knn::bruteMergable(p, trainingData, hyperBlocks, k, NUM_CLASSES);
                    break;

                // our explainable distance based KNN. takes a similarity of our point, then all train data, and uses k nearest to vote
                // IMPORTANT. WE ARE GOING TO SEG FAULT IN THIS CASE IF NOT HANDLED RIGHT.
                // THE STD DEVIATION IS CALCULATED ONLY ONCE, WE NEED TO RESET IT
                case THRESHOLD_KNN:
                    predictedClass = Knn::thresholdKNN(p, trainingData, NUM_CLASSES, k, threshold);
                    break;

                case OLD_KNN:
                    predictedClass = Knn::kNN(p, hyperBlocks, k, NUM_CLASSES);
                    break;

                case MERGABLE_KNN:
                    predictedClass = Knn::mergableKNN(p, trainingData, hyperBlocks, NUM_CLASSES);
                    break;

                case PRECISION_WEIGHTED:
                    prediction = precisionWeightedHBs(p, testingData, hyperBlocks, NUM_CLASSES, pointSummaries);
                    predictedClass = prediction.first;
                    blockHits = prediction.second;
                    break;
                default:
                    throw new runtime_error("Unknown classification mode");
            }
            // if we didn't classify it, put it in the not classified points. should only happen when we use HB mode.
            if (predictedClass == -1) {
                pointsWeCantClassify[cls].push_back(p);
            }
            else {
                confusionMatrix[cls][predictedClass]++;
                PointSummary& summary = pointSummaries[make_pair(cls, point)];
                summary.classIdx = cls;
                summary.pointIdx = point;
                summary.predictedIdx = predictedClass;
                summary.blockHits = blockHits;
            }
        }
    }

    // now we can just return our matrix
    return confusionMatrix;
}


pair<int, vector<BlockInfo>> ClassificationTests::precisionWeightedHBs(const vector<float> &point, std::vector<std::vector<std::vector<float>>>& testData, std::vector<HyperBlock>& hyperBlocks, int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries) {

    // Precision lost will hold each class of HB precison lost stats for each individual class
    /// Ex cls 0:  [0, .04, .10] indicates the precision lost from 0 was none, 1 was 4% and 2 was 10%
    /// hbPrecision is the precision score of the HB set itself. If class 0 has 100% that means it didn't misclassify anything as a 0 in validation.

    // Find how many HBs there are from each class to weight by
    std::vector<int> totalHBsPerClass(NUM_CLASSES);
    for(const auto& hb : hyperBlocks) {
        totalHBsPerClass[hb.classNum]++;
    }

    std::vector<float> floatVote(NUM_CLASSES, 0.0f);
    vector<BlockInfo> blockHits;

    for(int i = 0; i < hyperBlocks.size(); i++) {
        const auto& hb = hyperBlocks[i];

        // If the point is within the HB
        if(hb.inside_HB(point.size(), point.data())) {
            blockHits.push_back(BlockInfo{hb.classNum, hb.blockId, hb.size, -1});

            // We vote for the right class using precision
            if(totalHBsPerClass[hb.classNum] > 0) {
                floatVote[hb.classNum] += (hb.blockPrecision / totalHBsPerClass[hb.classNum]);

                // We vote for the possible other classes using the precision lost metric.
                for(int conf = 0; conf < NUM_CLASSES; conf++) {
                    // Vote for other classes is HB precision total * other class precision lost,
                    floatVote[conf] += hb.precisionLostByClass[conf] / totalHBsPerClass[hb.classNum];
                }
            }
        }
    }

    // Decide which one we want to vote for
    float max = 0.0f;
    int winningClass = -1;
    for(int i = 0; i < NUM_CLASSES; i++) {
        if(floatVote[i] > max) {
            max = floatVote[i];
            winningClass = i;
        }
    }

    if (max == 0.0f) {
        return {-1, blockHits};
    }


    return {winningClass, blockHits};
}




pair<int, vector<BlockInfo>> ClassificationTests::predictWithHBs(
    const vector<HyperBlock> &hyperBlocks, const vector<float> &point,
    int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries) {

    vector<int> numHbsPerClass(NUM_CLASSES, 0);
    for (const auto &hb : hyperBlocks) {
        numHbsPerClass[hb.classNum]++;
    }

    vector<float> votes(NUM_CLASSES, 0.0f);
    vector<BlockInfo> blockHits;

    for (int i = 0; i < hyperBlocks.size(); i++) {
        const auto &block = hyperBlocks[i];
        if (block.inside_HB(point.size(), point.data())) {
            float weight = 1.0f / numHbsPerClass[block.classNum];
            votes[block.classNum] += weight;
            blockHits.push_back(BlockInfo{block.classNum, block.blockId, block.size, -1});
        }
    }

    float maxVote = *std::max_element(votes.begin(), votes.end());

    if (maxVote == 0.0f) {
        return {-1, blockHits};
    }

    // Case 2: check for ties
    int winner = -1;
    int countMax = 0;
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        if (votes[cls] == maxVote) {
            ++countMax;
            winner = cls;  // last one seen; we only care that there's >1
        }
    }

    // if more than one class had that same amount of votes we return -1.
    if (countMax > 1) {
        // Tie between multiple classes
        return {-1, blockHits};
    }

    // if we had a real winner we can return it.
    return {winner, blockHits};
}




