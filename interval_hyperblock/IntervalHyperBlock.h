//
// Created by asnyd on 3/20/2025.
//
#pragma once
#include "Interval.h"
#include "DataAttr.h"
#include "../hyperblock/HyperBlock.h"

#ifndef INTERVALHYPERBLOCK_H
#define INTERVALHYPERBLOCK_H

class IntervalHyperBlock {
  public:

    static Interval longestInterval(std::vector<DataATTR> &dataByAttribute, int attribute);

    static void intervalHyper(std::vector<std::vector<std::vector<float>>> &realData, std::vector<std::vector<DataATTR>> remainingData, std::vector<HyperBlock> &hyperBlocks);

    static std::vector<std::vector<DataATTR>> separateByAttribute(std::vector<std::vector<std::vector<float>>>& data, int FIELD_LENGTH);

    static void sortByColumn(std::vector<std::vector<float>>& classData, int colIndex);

    static void generateHBs(std::vector<std::vector<std::vector<float>>>& data, std::vector<HyperBlock>& hyperBlocks, std::vector<int> &bestAttributes,int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS);

	static void merger_cuda(const std::vector<std::vector<std::vector<float>>>& allData, std::vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS);
};

#endif //INTERVALHYPERBLOCK_H
