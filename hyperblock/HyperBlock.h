#ifndef HYPERBLOCK_H
#define HYPERBLOCK_H

#include <vector>
#include <cmath>
class HyperBlock {
private:
public:
    std::vector<std::vector<float>> maximums;
    std::vector<std::vector<float>> minimums;
    int classNum;
    
    // Constructor
    HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls);
    HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls);
    float distance_to_HB(int numAttributes, const float* point) const;
    bool inside_HB(int numAttributes, const float* point);

};

#endif // HYPERBLOCK_H
