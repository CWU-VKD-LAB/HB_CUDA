//
// Created by asnyd on 5/5/2025.
//

#ifndef STATSTRUCTS_H
#define STATSTRUCTS_H
#include <map>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>

// MOST OF THIS IS JUST FOR DEBUGGING OR FIGURING OUT WHY ACCURACY IS GOOD (OR NOT). NOT MISSION CRITICAL STUFF, JUST DEBUGS TO DETERMINE WHERE MISCLASSIFICATION IS FROM BASICALLY.

struct BlockInfo {
    int blockClass;
    int blockIdx;
    int blockSize;
    float blockDensity;
};

// Keep track of the classification behavior of our points into blocks with this struct.
struct PointSummary {
    int classIdx;
    int pointIdx;
    int predictedIdx;         // THE CLASS IDX IT WAS PREDICTED AS OVERALL!!!!
    std::vector<BlockInfo> blockHits;
};

inline void printPointSummariesToCSV(const std::map<std::pair<int, int>, PointSummary>& pointSummaries, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "classIdx,pointIdx,predIdx,blockClassIdx,blockIdx,blockSize,blockDensity\n";

    std::map<std::pair<int, int>, PointSummary>::const_iterator it;
    for (it = pointSummaries.begin(); it != pointSummaries.end(); ++it) {
        const PointSummary& summary = it->second;
        for (size_t i = 0; i < summary.blockHits.size(); ++i) {
            const BlockInfo& hit = summary.blockHits[i];
            file << summary.classIdx << ","
                 << summary.pointIdx << ","
                 << summary.predictedIdx << ","
                 << hit.blockClass << ","
                 << hit.blockIdx << ","
                 << hit.blockSize << ","
                 << hit.blockDensity << "\n";
        }
    }

    file.close();
}

#endif //STATSTRUCTS_H
