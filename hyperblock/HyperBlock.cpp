#include "HyperBlock.h"
#include <limits>
#include <iostream>
// Constructor definition
HyperBlock::HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls) : maximums(maxs), minimums(mins), classNum(cls) {}

HyperBlock::HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls){
    int attr_count = hb_data[0][0].size(); // Number of attributes
    //std::cout << "print the vector\n" << std::endl;
    // Initialize maxes and mins with size and initial values
    std::vector<std::vector<float>> maxes(attr_count, std::vector<float>(1, -std::numeric_limits<float>::infinity()));
    std::vector<std::vector<float>> mins(attr_count, std::vector<float>(1, std::numeric_limits<float>::infinity()));

    // Find max and min for each attribute
    for(const std::vector<float>& point : hb_data[0]){
        for(int i = 0; i < point.size(); i++){
            if(point[i] > maxes[i][0]){
                maxes[i][0] = point[i];
            }
            if(point[i] < mins[i][0]){
                mins[i][0] = point[i];
            }
        }
    }

    size = -1;
    maximums = maxes;
    minimums = mins;
    classNum = cls;
}


void HyperBlock::findSize(const std::vector<std::vector<std::vector<float>>>& data){
    int size = 0;

    #pragma omp parallel for reduction(+:size)
    for(int i = 0; i < data.size(); i++){
      for(const auto& point: data[i]){
           size += inside_HB(point.size(), point.data());
      }
    }

    this->size = size;
}


bool HyperBlock::inside_HB(int numAttributes, const float* point) {
    constexpr float EPSILON = 1e-6f;  // Small tolerance value
    
    for (int i = 0; i < numAttributes; i++) {
        bool inAnInterval = false;

        for (int j = 0; j < maximums[i].size(); j++) {
            // Adjust comparisons with EPSILON to prevent floating-point issues
            if ((point[i] + EPSILON >= minimums[i][j]) && (point[i] - EPSILON <= maximums[i][j])) {
                inAnInterval = true;
                break;
            }
        }

        if (!inAnInterval) {
            return false;
        }
    }

    return true;  
}

float HyperBlock::distance_to_HB(int numAttributes, const float* point) const {
    constexpr float EPSILON = 1e-6f;
    float totalDistanceSquared = 0.0f;
    for (int i = 0; i < numAttributes; i++) {
        // If point is below the min bound (with epsilon tolerance)
        if (point[i] < minimums[i][0] - EPSILON) {
            float dist = minimums[i][0] - point[i];
            totalDistanceSquared += dist * dist;
        }
        // If point is above the max bound (with epsilon tolerance)
        else if (point[i] > maximums[i][0] + EPSILON) {
            float dist = point[i] - maximums[i][0];
            totalDistanceSquared += dist * dist;
        }
        // If point is within bounds (including epsilon tolerance), distance is 0
        // So we don't add anything to totalDistanceSquared
    }

    return std::sqrt(totalDistanceSquared);
}