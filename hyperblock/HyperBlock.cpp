#include "HyperBlock.h"
#include <limits>
// Constructor definition
HyperBlock::HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls) : maximums(maxs), minimums(mins), classNum(cls) {}

HyperBlock::HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls){
    int attr_count = hb_data[0][0].size(); // Number of attributes
    //std::cout << "print the vector\n" << std::endl;
    // Initialize maxes and mins with size and initial values
    std::vector<std::vector<float>> maxes(attr_count, std::vector<float>(1, -std::numeric_limits<float>::infinity()));
    std::vector<std::vector<float>> mins(attr_count, std::vector<float>(1, std::numeric_limits<float>::infinity()));

    //std::cout << "Trying to access class number: " << cls << "\n" << std::endl;
    //std::cout << "Number of attributes" << attr_count << std::endl;

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

    maximums = maxes;
    minimums = mins;
    classNum = cls;
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
        float minDistance = std::numeric_limits<float>::max();

        // Find the closest interval in this dimension
        
        float minVal = minimums[i][0];
        float maxVal = maximums[i][0];

        // Compute the shortest distance to the nearest bound
        float lowerDist = std::max(0.0f, minVal - point[i]);  
        float upperDist = std::max(0.0f, point[i] - maxVal);
        float closestDist = std::max(lowerDist, upperDist);

        minDistance = std::min(minDistance, closestDist);

        // Accumulate squared distance for Euclidean norm
        totalDistanceSquared += minDistance * minDistance;
    }

    return std::sqrt(totalDistanceSquared); 
}