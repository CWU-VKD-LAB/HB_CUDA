#include "HyperBlock.h"

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

float HyperBlock::distance_to_HB_Edge(int numAttributes, const float* point) const {
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

float HyperBlock::distance_to_HB_Avg(int numAttributes, const float* point) const{
    constexpr float EPSILON = 1e-6f;
    float totalDistanceSquared = 0.0f;
    for (int i = 0; i < numAttributes; i++) {
        // If point is below the min bound (with epsilon tolerance)
        if (point[i] < avgPoint[i] - EPSILON) {
            float dist = avgPoint[i] - point[i];
            totalDistanceSquared += dist * dist;
        }
        // If point is above the max bound (with epsilon tolerance)
        else if (point[i] > avgPoint[i] + EPSILON) {
            float dist = point[i] - avgPoint[i];
            totalDistanceSquared += dist * dist;
        }
    }
    return std::sqrt(totalDistanceSquared);
}


float HyperBlock::distance_to_HB_Combo(int numAttributes, const float* point) const {
    constexpr float EPSILON = 1e-6f;
    float totalDistanceSquared = 0.0f;
    for (int i = 0; i < numAttributes; i++) {
        // If point is below the min bound (with epsilon tolerance)
        if (point[i] < minimums[i][0] - EPSILON) {
            float dist = avgPoint[i] - point[i];
            totalDistanceSquared += dist * dist;
        }
        // If point is above the max bound (with epsilon tolerance)
        else if (point[i] > maximums[i][0] + EPSILON) {
            float dist = point[i] - avgPoint[i];
            totalDistanceSquared += dist * dist;
        }
        // If point is within bounds (including epsilon tolerance), distance is 0
        // So we don't add anything to totalDistanceSquared
    }

    return std::sqrt(totalDistanceSquared);
}






/**
* Previously this was just used to finmd the size of the hyperblock, but now we need to use it to find the average
* point too. This should be done by adding all the points together and then at the end dividing each attribute by size.
* this will allow for us to know where the "true" center of the block is.
*/
void HyperBlock::find_avg_and_size(const std::vector<std::vector<std::vector<float>>>& data) {
    int size = 0;
    std::vector<float> sumPoint(data[0][0].size(), 0.0f); // Initialize sum vector with zeros

    #pragma omp parallel
    {
        std::vector<float> localSum(data[0][0].size(), 0.0f);
        int localSize = 0;

        #pragma omp for nowait
        for (int i = 0; i < data.size(); i++) {
            for (const auto& point : data[i]) {
                if (inside_HB(point.size(), point.data())) {
                    localSize++;
                    for (size_t j = 0; j < point.size(); j++) {
                        localSum[j] += point[j];
                    }
                }
            }
        }

        #pragma omp critical
        {
            size += localSize;
            for (size_t j = 0; j < sumPoint.size(); j++) {
                sumPoint[j] += localSum[j];
            }
        }
    }

    // Compute the average point
    if (size > 0) {
        for (float& val : sumPoint) {
            val /= size;
        }
    }

    this->size = size;
    this->avgPoint = sumPoint;
}