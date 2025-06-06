//
// Created by asnyd on 3/20/2025.
//
#pragma once
#include <map>
#include <vector>
#include <string>
#include "../hyperblock/HyperBlock.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <random>
#include <algorithm>
#ifndef DATAUTIL_H
#define DATAUTIL_H
using namespace std;

class DataUtil {
    public:
        static vector<vector<vector<float>>> dataSetup(const string filepath, map<string, int>& classMap, map<int, string>& reversedClassMap);
        static void normalizeTestSet(vector<vector<vector<float>>>& testSet, const vector<float>& minValues, const vector<float>& maxValues, int FIELD_LENGTH);
        static void minMaxNormalization(vector<vector<vector<float>>>& dataset, const vector<float>& minValues, const vector<float>& maxValues, int FIELD_LENGTH);
        static vector<vector<vector<float>>> reorderTestingDataset(const vector<vector<vector<float>>>& testingData, const map<string, int>& CLASS_MAP_TRAINING, const map<string, int>& CLASS_MAP_TESTING);
        static void findMinMaxValuesInDataset(const vector<vector<vector<float>>>& dataset, vector<float>& minValues, vector<float>& maxValues, int FIELD_LENGTH);
        static vector<bool> markUniformColumns(const vector<vector<vector<float>>>& data);

        static vector<vector<float>> flattenDataset(vector<vector<vector<float>>>& data);
        static vector<vector<float>> flatMinMaxNoEncode(vector<HyperBlock> hyper_blocks, int FIELD_LENGTH);
        static vector<vector<float>> flattenMinsMaxesForRUB(vector<HyperBlock>& hyper_blocks, int FIELD_LENGTH);
        static vector<vector<vector<vector<float>>>> splitDataset(const vector<vector<vector<float>>> &dataset, int k);

        static void saveHyperBlocksToFile(const string& filepath, const vector<vector<vector<float>>>& hyperBlocks);
        static void saveNormalizedVersionToCsv(string fileName, vector<vector<vector<float>>>& data);

        static void saveOneToOneHBsToCSV(const vector<vector<HyperBlock>>& oneToOneHBs, const string& fileName, int FIELD_LENGTH);
        static vector<vector<HyperBlock>> loadOneToOneHBsFromCSV(const string& fileName,vector<pair<int, int>>& classPairsOut);

        static void createValidationSplit(vector<vector<vector<float>>>& trainingData, vector<vector<vector<float>>>& validationData,float validationFraction = 0.05f,unsigned int randomSeed = 42);
        static void splitTrainTestByPercent(vector<vector<vector<float>>>& trainingData, vector<vector<vector<float>>>& testingData, float percentTrain);

        // SAVE AND LOAD THE HYPER-BLOCKS IN A BINARY FORMAT, HOPEFULLY PREVENTING ROUNDING ERRORS ON OUR BLOCKS
        static vector<HyperBlock> loadBasicHBsFromBinary(const string& fileName);
        static void saveBasicHBsToBinary(const vector<HyperBlock>& hyperBlocks, const string& fileName, int FIELD_LENGTH);

        // SAVES BASIC HBS (WITHOUT DISJUNCTIONS) TO THE FILE INPUT AS CSV.
        static void saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks, const string& fileName, int FIELD_LENGTH);
        static vector<HyperBlock> loadBasicHBsFromCSV(const string& fileName);
        static vector<vector<HyperBlock>> loadOneToSomeBlocksFromBinary(const string& fileName);

        static void saveOneToOneHBsToBinary(const vector<vector<HyperBlock>>& oneToOneHBs, const string& fileName);
        static vector<vector<HyperBlock>> loadOneToOneHBsFromBinary(const string& fileName, vector<pair<int, int>>& classPairsOut);
};

#endif //DATAUTIL_H
