#ifndef HYPERBLOCK_H
#define HYPERBLOCK_H

#include <vector>

class HyperBlock {
private:
public:
    std::vector<std::vector<float>> maximums;
    std::vector<std::vector<float>> minimums;
    int classNum;
    
    // Constructor
    HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls);
    HyperBlock(const std::vector<std::vector<float>>& hb_data, int cls);
};

#endif // HYPERBLOCK_H
