//
// Created by asnyd on 3/20/2025.
//
#pragma once
#include <map>
#include <vector>
#include <string>
#include "../hyperblock/Hyperblock.h"
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


      static std::vector<HyperBlock> loadBasicHBsFromCSV(const std::string& fileName);
      static void saveBasicHBsToCSV(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH);
      static void saveHyperBlocksToFile(const std::string& filepath, const std::vector<std::vector<std::vector<float>>>& hyperBlocks);
      static void saveNormalizedVersionToCsv(std::string fileName, std::vector<std::vector<std::vector<float>>>& data);
    };



#endif //DATAUTIL_H
