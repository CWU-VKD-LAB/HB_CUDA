//
// Created by asnyd on 3/20/2025.
//
#pragma once
#include "../hyperblock/HyperBlock.h"
#include <atomic>
#include <unordered_set>
#include <vector>
#include <future>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <map>
#include <ostream>
#include <unordered_set>
#include <utility>
#include <thread>

#include "Interval.h"
#include "DataAttr.h"
#include "../hyperblock_generation/MergerHyperBlock.cuh"

#ifndef INTERVALHYPERBLOCK_H
#define INTERVALHYPERBLOCK_H

class IntervalHyperBlock {
  public:

    // stupid structs because we can't just use a simple hash for some reason.
    // just using these for the set of used points.
    struct PairHash {
        std::size_t operator()(const std::pair<int,int> &p) const {
            // hash function.
            return static_cast<std::size_t>(p.first) * 809ULL + static_cast<std::size_t>(p.second); // using 809 because mnist is 784 attributes, so that would maybe be an issue if smaller?
        }
    };

    // default equality operator should be fine for int and int. just checks if the two numbers are equal
    struct PairEq {
        bool operator()(const std::pair<int,int> &a, const std::pair<int,int> &b) const {
            return a == b;
        }
    };

    static void intervalHyperWorker(std::vector<std::vector<DataATTR>> &attributeColumns, Interval &threadBestInterval, int threadID, int threadCount, std::atomic<int> &readyThreadsCount, char *currentPhase, std::unordered_set<std::pair<int, int>, PairHash, PairEq> &usedPoints, std::vector<char> &doneColumns);

    static void intervalHyperSupervisor(std::vector<std::vector<std::vector<float>>> &realData, std::vector<std::vector<DataATTR>> &dataByAttribute, std::vector<HyperBlock> &hyperBlocks);

    static Interval longestInterval(std::vector<DataATTR> &dataByAttribute, int attribute);

    static void intervalHyper(std::vector<std::vector<std::vector<float>>> &realData, std::vector<std::vector<DataATTR>> &remainingData, std::vector<HyperBlock> &hyperBlocks);

    static std::vector<std::vector<DataATTR>> separateByAttribute(std::vector<std::vector<std::vector<float>>>& data, int FIELD_LENGTH);

    static void sortByColumn(std::vector<std::vector<float>>& classData, int colIndex);

    static void generateHBs(std::vector<std::vector<std::vector<float>>>& data, std::vector<HyperBlock>& hyperBlocks, std::vector<int> &bestAttributes,int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS);

	static void merger_cuda(const std::vector<std::vector<std::vector<float>>>& allData, std::vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS);
};

#endif //INTERVALHYPERBLOCK_H
