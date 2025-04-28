//
// Created by asnyd on 3/20/2025.
//
#pragma once

#ifndef KNN_H
#define KNN_H

#include <vector>
#include "../hyperblock/HyperBlock.h"
#include "../data_utilities/DataUtil.h"
#include "../interval_hyperblock/DataAttr.h"
#include "../interval_hyperblock/IntervalHyperBlock.h"

class Knn {
    public:
       static float euclideanDistanceBounds(const std::vector<float>& blockBound, const std::vector<float>& point, int FIELD_LENGTH);
       static std::vector<std::vector<long>> kNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);
       static float euclideanDistancePoints(const std::vector<float>& point2, const std::vector<float>& point, int FIELD_LENGTH);
       static std::vector<std::vector<long>> blockPointkNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<std::vector<std::vector<float>>> classifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);
       static std::vector<std::vector<long>> closeToInkNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);
       static std::vector<std::vector<long>> mostAttributesInKnn(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);
       static std::vector<std::vector<long>> closestBlock(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int NUM_CLASSES);
       static std::vector<std::vector<long>> mergableKNN(std::vector<std::vector<std::vector<float>>> &unclassifiedData, std::vector<std::vector<std::vector<float>>> &trainingData, std::vector<HyperBlock> &hyperBlocks, int NUM_CLASSES);
       static float mergeCheck(std::vector<int> &insertIdx, HyperBlock &h, std::vector<std::vector<DataATTR>> &dataByAttribute);
       static bool isInside(const std::vector<float>& point, const std::vector<std::vector<float>>& fMins, const std::vector<std::vector<float>>& fMaxes);
        static std::vector<std::vector<long>> Knn::bruteMergable(std::vector<std::vector<std::vector<float>>> unclassifiedData,std::vector<std::vector<std::vector<float>>> classifiedData,std::vector<HyperBlock>& hyperBlocks,int k,int NUM_CLASSES);
    static std::vector<std::vector<long>> Knn::pureKnn(
        std::vector<std::vector<std::vector<float>>> unclassifiedData,
        std::vector<std::vector<std::vector<float>>> classifiedData,
        int k, int NUM_CLASSES);
};



#endif //KNN_H
