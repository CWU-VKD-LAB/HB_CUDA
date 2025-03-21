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

bool HyperBlock::inside_HB(int numAttributes, const float* point){
    bool inside = true;

    // Go through all attributes
    for (int i = 0; i < numAttributes; i++)
    {
        bool inAnInterval = false;

        // Go through all intervals the hyperblock allows for the attribute
        for(int j = 0; j < maximums[i].size(); j++){
            // If the datapoints value falls inside one of the intervals.
            if (point[i] >= minimums[i][j] && point[i] <= maximums[i][j]) {
                inAnInterval = true;
                break;
            }
        }

        if (!inAnInterval) {
            inside = false;
            break;
        }
    }

    // Should return true if the point is inside at least 1 interval for all attributes.
    return inside;
}
