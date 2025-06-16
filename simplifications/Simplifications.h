// Created by Austin Snyder on 3/20/2025.

#pragma once
#include "../hyperblock/HyperBlock.h"
#include <cuda_runtime.h>
#include "../data_utilities/DataUtil.h"
#include "../hyperblock_generation/MergerHyperBlock.cuh"
#include <algorithm>
#include <vector>

#ifndef SIMPLIFICATIONS_H
#define SIMPLIFICATIONS_H
class Simplifications {
    public:
        static int REMOVAL_COUNT;
        static void removeUselessBlocks(vector<vector<vector<float>>> &data, vector<HyperBlock>& hyper_blocks);
        static vector<int> runSimplifications(vector<HyperBlock> &hyperBlocks, vector<vector<vector<float>>> &trainData, vector<vector<int>> &bestAttributeOrdering);
        static void removeUselessAttr(vector<HyperBlock> &hyper_blocks, vector<vector<vector<float>>> &data, vector<vector<int>> &attributeOrderings);
        static void Simplifications::removeUselessAttrNoDisjunction(std::vector<HyperBlock>& hyper_blocks, std::vector<std::vector<std::vector<float>>>& data, std::vector<std::vector<int>>& attributeOrderings);
};



#endif //SIMPLIFICATIONS_H
