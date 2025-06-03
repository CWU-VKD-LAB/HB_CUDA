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

class DataUtil {
    public:
        static std::vector<std::vector<std::vector<float>>> dataSetup(const std::string filepath, std::map<std::string, int>& classMap, std::map<int, std::string>& reversedClassMap);
        static void normalizeTestSet(std::vector<std::vector<std::vector<float>>>& testSet, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH);
        static void minMaxNormalization(std::vector<std::vector<std::vector<float>>>& dataset, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH);
        static std::vector<std::vector<std::vector<float>>> reorderTestingDataset(const std::vector<std::vector<std::vector<float>>>& testingData, const std::map<std::string, int>& CLASS_MAP_TRAINING, const std::map<std::string, int>& CLASS_MAP_TESTING);
        static void findMinMaxValuesInDataset(const std::vector<std::vector<std::vector<float>>>& dataset, std::vector<float>& minValues, std::vector<float>& maxValues, int FIELD_LENGTH);
        static std::vector<bool> markUniformColumns(const std::vector<std::vector<std::vector<float>>>& data);

        static std::vector<std::vector<float>> flattenDataset(std::vector<std::vector<std::vector<float>>>& data);
        static std::vector<std::vector<float>> flatMinMaxNoEncode(std::vector<HyperBlock> hyper_blocks, int FIELD_LENGTH);
        static std::vector<std::vector<float>> flattenMinsMaxesForRUB(std::vector<HyperBlock>& hyper_blocks, int FIELD_LENGTH);
        static std::vector<std::vector<std::vector<std::vector<float>>>> splitDataset(const std::vector<std::vector<std::vector<float>>> &dataset, int k);

        static void saveHyperBlocksToFile(const std::string& filepath, const std::vector<std::vector<std::vector<float>>>& hyperBlocks);
        static void saveNormalizedVersionToCsv(std::string fileName, std::vector<std::vector<std::vector<float>>>& data);

        static void saveOneToOneHBsToCSV(const std::vector<std::vector<HyperBlock>>& oneToOneHBs, const std::string& fileName, int FIELD_LENGTH);
        static std::vector<std::vector<HyperBlock>> loadOneToOneHBsFromCSV(const std::string& fileName,std::vector<std::pair<int, int>>& classPairsOut);

        static void createValidationSplit(std::vector<std::vector<std::vector<float>>>& trainingData, std::vector<std::vector<std::vector<float>>>& validationData,float validationFraction = 0.05f,unsigned int randomSeed = 42);
        static void splitTrainTestByPercent(std::vector<std::vector<std::vector<float>>>& trainingData, std::vector<std::vector<std::vector<float>>>& testingData, float percentTrain);

        // SAVE AND LOAD THE HYPER-BLOCKS IN A BINARY FORMAT, HOPEFULLY PREVENTING ROUNDING ERRORS ON OUR BLOCKS
        static std::vector<HyperBlock> loadBasicHBsFromBinary(const std::string& fileName);
        static void saveBasicHBsToBinary(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH);

        // SAVES BASIC HBS (WITHOUT DISJUNCTIONS) TO THE FILE INPUT AS CSV.
        static void saveBasicHBsToCSV(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH);
        static std::vector<HyperBlock> loadBasicHBsFromCSV(const std::string& fileName);
        static std::vector<std::vector<HyperBlock>> loadOneToSomeBlocksFromBinary(const std::string& fileName);

        static void saveOneToOneHBsToBinary(const std::vector<std::vector<HyperBlock>>& oneToOneHBs, const std::string& fileName);
        static std::vector<std::vector<HyperBlock>> loadOneToOneHBsFromBinary(const std::string& fileName, std::vector<std::pair<int, int>>& classPairsOut);
};

#endif //DATAUTIL_H
