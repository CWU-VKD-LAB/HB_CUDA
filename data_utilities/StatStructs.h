//
// Created by asnyd on 5/5/2025.
//

#ifndef STATSTRUCTS_H
#define STATSTRUCTS_H
#include <vector>

struct BlockInfo {
    int blockClass;
    int blockIdx;
    float blockSize;
    float blockDensity;
};

// Keep track of the classification behavior of our points into blocks with this struct.
struct PointSummary {
    int classIdx;
    int pointIdx;
    int predictedIdx;         // THE CLASS IDX IT WAS PREDICTED AS OVERALL!!!!
    std::vector<BlockInfo> blockHits;
};

#endif //STATSTRUCTS_H
