#ifndef HYPERBLOCK_H
#define HYPERBLOCK_H

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
class HyperBlock {
private:
public:
    std::vector<std::vector<float>> maximums;
    std::vector<std::vector<float>> minimums;
    int classNum;
    int size;

    std::vector<float> avgPoint;

    // Constructor
    HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls);
    HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls);
    float distance_to_HB_Edge(int numAttributes, const float* point) const;
    float distance_to_HB_Avg(int numAttributes, const float* point) const;
    float distance_to_HB_Combo(int numAttributes, const float* point) const;

    bool inside_HB(int numAttributes, const float* point);
    void find_avg_and_size(const std::vector<std::vector<std::vector<float>>>& data);

};

#endif // HYPERBLOCK_H
