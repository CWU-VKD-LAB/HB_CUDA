#ifndef HYPERBLOCK_H
#define HYPERBLOCK_H

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <utility>
#include <map>
#include "../data_utilities/StatStructs.h"

class HyperBlock {
private:
public:

    /**
    *   Min and Max bounds for the HB. (2-D because disjunctive HBs will require it.)
    *
    *   If: maxs = [[1], [1], [1]]
    *       mins = [[0], [0], [0]]
    *
    *   Then it would cover the entire feature space when there are 3 features (A useless block but used for the example.)
    */
    std::vector<std::vector<float>> maximums;
    std::vector<std::vector<float>> minimums;

    /**
    * The class number of the Hyperblock. Corresponds to trainingData[classNum]
    */
    int classNum;
    int size;

    // These are barely used, can remove in future if you want.
    std::vector<float> tamedMin;
    std::vector<float> tamedMax;

    /**
    * These are for the precision weighted HB voting, essentially an HB
    * will be able to vote based on its precision on the validation set.
    *
    * Additionally, if it misclassified a class X, then it will vote a small amount for class X.
    */
    float blockPrecision;
    std::vector<float> precisionLostByClass;


    // This will hold the indices of which points are in the block [classIdx][pointIdx]
    std::vector<std::vector<int>> pointIndices;
    std::vector<float> avgPoint;

    // Unqiue identifier for HBs.
    int blockId;

    // top and bottom pairs is the indexes which are top and bottom of the sorted list of dataATTR's in the interval and merging without cuda.
    std::vector<std::pair<int, int>> topBottomPairs; // first is bottom, second is top of interval

    // Constructor
    HyperBlock(const std::vector<std::vector<float>>& maxs, const std::vector<std::vector<float>>& mins, int cls);
    HyperBlock(std::vector<std::vector<std::vector<float>>>& hb_data, int cls);
    float distance_to_HB_Edge(int numAttributes, const float* point) const;
    float distance_to_HB_Avg(int numAttributes, const float* point) const;
    float distance_to_HB_Combo(int numAttributes, const float* point) const;
    void tameBounds(const std::vector<std::vector<std::vector<float>>>& trainingData);

    void setHBPrecisions(std::map<std::pair<int, int>, PointSummary> summaries, int NUM_CLASSES, bool voted = false);

    bool inside_HB(int numAttributes, const float* point) const;
    void find_avg_and_size(const std::vector<std::vector<std::vector<float>>>& data);
    int inside_N_Bounds(int numAttributes, const float* point);

};

#endif // HYPERBLOCK_H
