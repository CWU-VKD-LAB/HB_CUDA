//
// Created by asnyd on 3/20/2025.
//
#pragma once
#include <vector>
#include "../hyperblock/HyperBlock.h"
#ifndef KNN_H
#define KNN_H



class Knn {
    public:
       static float euclideanDistance(const std::vector<float>& hbCenter, const std::vector<float>& point, int FIELD_LENGTH);
       static std::vector<std::vector<long>> kNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES);


};



#endif //KNN_H
