#include "HyperBlock.h"

#include "../data_utilities/StatStructs.h"

// Constructor definition
HyperBlock::HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls) : maximums(maxs), minimums(mins), classNum(cls) {
    topBottomPairs.resize(maxs.size());
}

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
    pointIndices = std::vector<std::vector<int>>();
    maximums = maxes;
    minimums = mins;
    classNum = cls;
    topBottomPairs.resize(attr_count);
}


bool HyperBlock::inside_HB(int numAttributes, const float* point) const {
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


// Returns how many bounds the point was in.
int HyperBlock::inside_N_Bounds(int numAttributes, const float* point) {
    constexpr float EPSILON = 1e-6f;
    int numIn = 0;

    for (int i = 0; i < numAttributes; i++)
        for (int j = 0; j < maximums[i].size(); j++)
            if ((point[i] + EPSILON >= minimums[i][j]) && (point[i] - EPSILON <= maximums[i][j]))
                numIn++;

    return numIn;
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


/**
 * Calculate precision metrics for this specific HyperBlock based on validation data
 * Sets this->blockPrecision and populates this->precisionLostByClass
 * @param summaries Point summaries from validation run (blockId -> PointSummary mapping)
 * @param NUM_CLASSES Total number of classes in the problem
 * @param blockIdx The unique index id of this HyperBlock
 * @param useVotedResult If this is true, the algorithm should use the final prediction to not penalize blocks that were wrong but didn't result in bad classificatrion.
 * @return The precision of this HyperBlock (TP / (TP + FP))
 */
void HyperBlock::setHBPrecisions(std::map<std::pair<int, int>, PointSummary> summaries, int NUM_CLASSES, bool useVotedResult) {
    // Initialize the precisionLostByClass vector
    this->precisionLostByClass.assign(NUM_CLASSES, 0.0f);

    int TP = 0;  // True positives: points of this block's class that fell into this block
    int FP = 0;  // False positives: points of other classes that fell into this block
    std::vector<int> FP_by_class(NUM_CLASSES, 0);  // Track FP by each class

    // Go through all point summaries to find points that fell into this block
    for (const auto& entry : summaries) {
        const auto& key = entry.first;
        const auto& pointSummary = entry.second;

        int actualClass = key.first;

        // Check if this point fell into our block
        bool pointInThisBlock = false;
        for (const auto& blockInfo : pointSummary.blockHits) {
            if (blockInfo.blockIdx == this->blockId) {
                pointInThisBlock = true;
                break;
            }
        }

        if (pointInThisBlock) {
            if (actualClass == this->classNum) {
                TP++;
            } else {
                FP++;
                FP_by_class[actualClass]++;
            }
        }
    }

    // Set the block's precision
    int total = TP + FP;
    if (total > 0) {
        this->blockPrecision = static_cast<float>(TP) / total;
    } else {
        this->blockPrecision = 1.0f;
    }

    // Calculate precision lost by each class and populate the member variable
    if (FP > 0) {
        float totalPrecisionLoss = static_cast<float>(FP) / total;

        for (int i = 0; i < NUM_CLASSES; i++) {
            if (i != this->classNum && FP_by_class[i] > 0) {
                // Proportion of total loss attributed to class i
                this->precisionLostByClass[i] = totalPrecisionLoss * static_cast<float>(FP_by_class[i]) / FP;
            }
        }
    }
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
    int totalSize = 0;
    std::vector<float> sumPoint(data[0][0].size(), 0.0f); // Initialize sum vector with zeros
    pointIndices.clear();
    pointIndices.resize(data.size());

    // Thread-local storage
    #pragma omp parallel
    {
        std::vector<float> localSum(data[0][0].size(), 0.0f);
        int localSize = 0;
        std::vector<std::vector<int>> localIndices(data.size()); // Thread-local indices

        #pragma omp for nowait
        for (int classIdx = 0; classIdx < data.size(); classIdx++) {
            for (int pointIdx = 0; pointIdx < data[classIdx].size(); pointIdx++) {
                const auto& point = data[classIdx][pointIdx];
                if (inside_HB(point.size(), point.data())) {
                    localSize++;
                    for (size_t j = 0; j < point.size(); j++) {
                        localSum[j] += point[j];
                    }
                    localIndices[classIdx].push_back(pointIdx); // Track which point matched
                }
            }
        }

        // Merge thread-local results into shared variables
        #pragma omp critical
        {
            totalSize += localSize;
            for (size_t j = 0; j < sumPoint.size(); j++) {
                sumPoint[j] += localSum[j];
            }
            for (int classIdx = 0; classIdx < data.size(); classIdx++) {
                pointIndices[classIdx].insert(
                    pointIndices[classIdx].end(),
                    localIndices[classIdx].begin(),
                    localIndices[classIdx].end()
                );
            }
        }
    }

    // Compute average point
    if (totalSize > 0) {
        for (float& val : sumPoint) {
            val /= totalSize;
        }
    }

    this->size = totalSize;
    this->avgPoint = sumPoint;
}

void HyperBlock::tameBounds(const std::vector<std::vector<std::vector<float>>>& trainingData) {
    // Ensure we are working with non-disjunctive bounds
    int numDims = minimums.size();
    int numPoints = pointIndices[classNum].size();

    if (numPoints == 0) return;

    // Gather points in the block
    std::vector<std::vector<float>> inBlockPoints(numPoints);
    for (int i = 0; i < numPoints; i++) {
        int pointIdx = pointIndices[classNum][i];
        inBlockPoints[i] = trainingData[classNum][pointIdx];
    }

    // Compute dominance count for min (dominated by others)
    std::vector<int> minDominanceCounts(numPoints, 0);
    // Compute dominance count for max (dominates others)
    std::vector<int> maxDominanceCounts(numPoints, 0);

    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < numPoints; j++) {
            if (i == j) continue;

            bool jAboveI = true;
            bool jBelowI = true;

            for (int d = 0; d < numDims; d++) {
                if (inBlockPoints[j][d] < inBlockPoints[i][d]) {
                    jAboveI = false;
                }
                if (inBlockPoints[j][d] > inBlockPoints[i][d]) {
                    jBelowI = false;
                }
            }

            if (jAboveI) minDominanceCounts[i]++;
            if (jBelowI) maxDominanceCounts[i]++;
        }
    }

    // Find the best min and max index
    int bestMinIdx = 0;
    int bestMaxIdx = 0;
    int maxDomBy = -1;
    int maxDomOver = -1;

    for (int i = 0; i < numPoints; i++) {
        if (minDominanceCounts[i] > maxDomBy) {
            maxDomBy = minDominanceCounts[i];
            bestMinIdx = i;
        }
        if (maxDominanceCounts[i] > maxDomOver) {
            maxDomOver = maxDominanceCounts[i];
            bestMaxIdx = i;
        }
    }

    // Save tamed min and max to class fields
    tamedMin = inBlockPoints[bestMinIdx];
    tamedMax = inBlockPoints[bestMaxIdx];
}