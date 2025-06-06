//
// Created by asnyd on 3/20/2025.
//
#pragma once

#ifndef KNN_H
#define KNN_H

#include <vector>
#include "../hyperblock/HyperBlock.h"
#include "../interval_hyperblock/DataAttr.h"

class Knn {
    public:

    static float euclideanDistanceBounds(const std::vector<float>& blockBound, const std::vector<float>& point, int FIELD_LENGTH);
    static int kNN(const std::vector<float> &point, const std::vector<HyperBlock>& hyperBlocks, int k, const int NUM_CLASSES);

    static float euclideanDistancePoints(const std::vector<float>& point2, const std::vector<float>& point, int FIELD_LENGTH);

    static int closeToInkNN(const std::vector<float> &point, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);

    static bool isInside(const std::vector<float>& point, const std::vector<std::vector<float>>& fMins, const std::vector<std::vector<float>>& fMaxes);

    static int bruteMergable(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &classifiedData, std::vector<HyperBlock>& hyperBlocks,int k,int NUM_CLASSES);

    static int mergableKNN(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &trainingData, std::vector<HyperBlock> &hyperBlocks, int NUM_CLASSES);

    static float mergeCheck(std::vector<int> &insertIdx, HyperBlock &h, std::vector<std::vector<DataATTR>> &dataByAttribute);

    static int pureKnn(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &classifiedData, const int NUM_CLASSES, const int k);

    static std::vector<float> losslessDistance(const std::vector<float> &seedPoint, const std::vector<float> &trainPoint);

    static bool deviationsComputed;
    static std::vector<float> computeStdDeviations(const std::vector<std::vector<std::vector<float>>> &trainData);

    static int thresholdKNN(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &classifiedData, int NUM_CLASSES, int k, float threshold);
};

#endif //KNN_H
