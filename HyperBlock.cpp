#include "HyperBlock.h"

// Constructor definition
HyperBlock::HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls) : maximums(maxs), minimums(mins), classNum(cls) {}

HyperBlock::HyperBlock(const std::vector<std::vector<std::vector<float>>>& hb_data, int cls){
    size_t attr_count = hb_data[0].size(); // Number of attributes

    // Initialize maxes and mins with size and initial values
    std::vector<std::vector<float>> maxes(attr_count, std::vector<float>(1, std::numeric_limits<float>::lowest()));
    std::vector<std::vector<float>> mins(attr_count, std::vector<float>(1, std::numeric_limits<float>::max()));

    // Find max and min for each attribute
    for(const std::vector<float>& point : hb_data[classNum]){
        for(size_t i = 0; i < point.size(); i++){
            if(point[i] > maxes[i][0]){
                maxes[i][0] = point[i];
            }
            if(point[i] < mins[i][0]){
                mins[i][0] = point[i];
            }
        }
    }

    maximums = maxes;
    minimums = mins;
    classNum = cls;
}
