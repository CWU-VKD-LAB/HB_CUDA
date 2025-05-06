#ifndef HYPERBLOCK_H
#define HYPERBLOCK_H

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <utility>

class HyperBlock {
private:
public:
    std::vector<std::vector<float>> maximums;
    std::vector<std::vector<float>> minimums;
    int classNum;
    int size;

    std::vector<float> tamedMin;
    std::vector<float> tamedMax;


    // This will hold the indices of which points are in the block [classIdx][pointIdx]
    std::vector<std::vector<int>> pointIndices;

    std::vector<float> avgPoint;

    // top and bottom pairs is the indexes which are top and bottom of the sorted list of dataATTR's in the interval and merging without cuda.
    std::vector<std::pair<int, int>> topBottomPairs; // first is bottom, second is top of interval

    // Constructor
    HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls);
    HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls);
    float distance_to_HB_Edge(int numAttributes, const float* point) const;
    float distance_to_HB_Avg(int numAttributes, const float* point) const;
    float distance_to_HB_Combo(int numAttributes, const float* point) const;
    void tameBounds(const std::vector<std::vector<std::vector<float>>>& trainingData);


    bool inside_HB(int numAttributes, const float* point) const;
    void find_avg_and_size(const std::vector<std::vector<std::vector<float>>>& data);
    int inside_N_Bounds(int numAttributes, const float* point);

};

#endif // HYPERBLOCK_H
