
// Created by asnyd on 3/20/2025.

#pragma once
#include <vector>
#include "../hyperblock/HyperBlock.h"
#include <vector>
#include <cuda_runtime.h>

#ifndef SIMPLIFICATIONS_H
#define SIMPLIFICATIONS_H

class Simplifications {
    public:
        static void removeUselessBlocks(std::vector<std::vector<std::vector<float>>> &data, std::vector<HyperBlock>& hyper_blocks);
        static std::vector<int> runSimplifications(std::vector<HyperBlock> &hyperBlocks, std::vector<std::vector<std::vector<float>>> &trainData, std::vector<std::vector<int>> &bestAttributeOrderings);
        static void removeUselessAttributesCUDA(std::vector<HyperBlock> &hyper_blocks, std::vector<std::vector<std::vector<float>>> &data, std::vector<std::vector<int>> &attributeOrderings);
};



#endif //SIMPLIFICATIONS_H
