//
// Created by Austin Snyder on 3/20/2025.
//
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <iomanip>

#ifndef PRINTINGUTIL_H
#define PRINTINGUTIL_H

class PrintingUtil {
  public:
    static void clearScreen();
    static void waitForEnter();
    static void displayMainMenu();
    static float printConfusionMatrix(std::vector<std::vector<long>>& data, const int NUM_CLASSES, std::map<int, std::string>& CLASS_MAP_INT);
    static void printDataset(const std::vector<std::vector<std::vector<float>>>& vec);
};



#endif //PRINTINGUTIL_H
