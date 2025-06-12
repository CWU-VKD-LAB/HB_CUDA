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
#include <iostream>
#include <ostream>
#include <utility>
#include <thread>
#include <numeric>
#include <omp.h>
#include "Interval.h"
#include "DataAttr.h"
#include "../hyperblock_generation/MergerHyperBlock.cuh"

#ifndef INTERVALHYPERBLOCK_H
#define INTERVALHYPERBLOCK_H

using namespace std;
class IntervalHyperBlock {
  public:

    // stupid structs because we can't just use a simple hash for some reason.
    // classNum, classIndex
    struct PairHash {
        size_t operator()(const pair<int,int> &p) const {
            // hash function.
            return static_cast<size_t>(p.first) * 809ULL + static_cast<size_t>(p.second); // using 809 because mnist is 784 attributes, so that would maybe be an issue if smaller?
        }
    };

    // default equality operator should be fine for int and int. just checks if the two numbers are equal
    struct PairEq {
        bool operator()(const pair<int,int> &a, const pair<int,int> &b) const {
            return a == b;
        }
    };


    static void pureBlockIntervalHyper(vector<vector<DataATTR>> &dataByAttribute, vector<vector<vector<float>>> &trainingData, vector<HyperBlock> &hyperBlocks, int COMMAND_LINE_ARGS_CLASS);

    static void intervalHyperWorker(vector<vector<DataATTR>> &attributeColumns, Interval &threadBestInterval, int threadID, int threadCount, atomic<int> &readyThreadsCount, char *currentPhase, unordered_set<pair<int, int>, PairHash, PairEq> &usedPoints, vector<char> &doneColumns, int COMMAND_LINE_ARGS_CLASS);

    static void intervalHyperSupervisor(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> &dataByAttribute, vector<HyperBlock> &hyperBlocks, int COMMAND_LINE_ARGS_CLASS);

    static Interval longestInterval(vector<DataATTR> &dataByAttribute, int attribute);

    static void intervalHyper(vector<vector<vector<float>>> &realData, vector<vector<DataATTR>> &remainingData, vector<HyperBlock> &hyperBlocks);

    static vector<vector<DataATTR>> separateByAttribute(const vector<vector<vector<float>>>& data, int FIELD_LENGTH);

    static void sortByColumn(vector<vector<float>>& classData, int colIndex);

    static void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyperBlocks, vector<int> &bestAttributes,int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS);

	static void merger_cuda(const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks, int COMMAND_LINE_ARGS_CLASS);

    static void mergerNotInCuda(vector<vector<vector<float>>> &trainingData, vector<HyperBlock> &hyperBlocks, vector<vector<DataATTR>> &pointsBrokenUp);

    static bool checkMergable(vector<vector<DataATTR>> &dataByAttribute, HyperBlock &h);

    static vector<vector<vector<float>>> increaseLevelOfTrainingSet(vector<HyperBlock> &hyperBlocks, vector<vector<vector<float>>> &inputTrainingData, int FIELD_LENGTH);

    static unordered_set<pair<int,int>, PairHash, PairEq> findHBEnvelopeCases(HyperBlock &hb, vector<vector<DataATTR>> &dataByAttribute);

    static vector<vector<vector<float>>> generateNextLevelHBs(vector<vector<vector<float>>> &trainingData, vector<HyperBlock> &inputBlocks, vector<HyperBlock> &nextLevelBlocks, vector<int> &bestAttributes, int FIELD_LENGTH, int COMMAND_LINE_ARGS_CLASS);

};


#endif //INTERVALHYPERBLOCK_H
