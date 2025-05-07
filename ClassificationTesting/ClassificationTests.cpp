//
// Created by Ryan Gallagher on 5/5/25.
//

#include "ClassificationTests.h"


/*
 * takes in our hyperblocks by reference. takes in the points we need to classify. and takes in a mode, as well as a reference to another container for points.
 * the points we can't classify is for those which fall out of all HBs. this allows us to easily make a list of points we couldn't handle, and we would just call the function
 * again with another mode, to reclassify those points specifically.
 */
vector<vector<long>> ClassificationTests::buildConfusionMatrix(vector<HyperBlock> &hyperBlocks, const vector<vector<vector<float>>> &trainingData, const vector<vector<vector<float>>> &pointsToClassify, int classificationMode, vector<vector<vector<float>>> &pointsWeCantClassify, const int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries, int k, float threshold) {

    vector<vector<long>> confusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));

    // go through all classes
    for(int cls = 0; cls < NUM_CLASSES; cls++) {

        // all points each class. we are going to use whichever classifier we have called our function with to build the matrix
        for(int point = 0; point < pointsToClassify[cls].size(); point++) {

            // get our reference to the point
            const auto &p = pointsToClassify[cls][point];
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

pair<int, vector<BlockInfo>> ClassificationTests::predictWithHBs(const vector<HyperBlock> &hyperBlocks, const vector<float> &point, int NUM_CLASSES, map<pair<int, int>, PointSummary>& pointSummaries) {

    vector<int> votes(NUM_CLASSES, 0);
    vector<BlockInfo> blockHits;
    // if we instead wanted to just take the first block it falls in, we would just return early when we find a block we fall into.
    // this voting version works better in testing though.
    for (int i = 0; i < hyperBlocks.size(); i++) {
        const auto &block = hyperBlocks[i];
        if (block.inside_HB(point.size(), point.data())) {
            votes[block.classNum]++;
            blockHits.push_back(BlockInfo{block.classNum, i, block.size, -1});
        }
    }

    // return the class which had most votes, and all the block info from the blocks that we hit.
    return {std::distance(votes.begin(),std::max_element(votes.begin(), votes.end())), blockHits};
}





