//
// Created by asnyd on 3/20/2025.
//

#include "Knn.h"
#include <vector>
#include <pair>

/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
std::vector<std::vector<long>> kNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k){

    if(k > hyperBlocks.size()) k = (int) sqrt(hyperBlocks.size());

    // Keep track of assignments with something
    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    // Flatten out the hyperBlocks into their centers
    std::vector<std::vector<std::vector<float>>> hyperBlockCentroids(NUM_CLASSES);    //[class][block][attribute]

    for(const auto& hyperBlock : hyperBlocks){
        // Get the center of the block
        std::vector<float> blockCenter(FIELD_LENGTH, 0);
        for(int i = 0; i < FIELD_LENGTH; i++){
            blockCenter[i] = (hyperBlock.maximums[i][0] + hyperBlock.minimums[i][0]) / 2.0f;
        }

        hyperBlockCentroids[hyperBlock.classNum].push_back(blockCenter);
    }

    // For each class of points
    for(int i = 0; i < NUM_CLASSES; i++){

        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){
            // Use a priority queue to keep track of the top k best distances
            priority_queue<pair<float, int>> kNearest;


            // Go through all the blocks and find the distances to their centers
            for(int blockClass = 0; blockClass < NUM_CLASSES; blockClass++){
                for(const auto& currHBCenter : hyperBlockCentroids[blockClass]){
                    // Find the distance between the HB center and the unclassified data point
                    float distance = euclideanDistance(currHBCenter, unclassifiedData[i][point]);

                    if(kNearest.size() < k){    // always add when queue is not at k yet.
                        kNearest.push(std::make_pair(distance, blockClass));
                    }
                    else if(distance < kNearest.top().first){ // Queue is big enough, and this distance is better than the worst in queue
                        kNearest.pop();    // pop the max (worst distance)
                        kNearest.push(std::make_pair(distance, blockClass));    // push the better distance.
                    }
                }
            }

            // Count votes for each class
            std::vector<int> votes(NUM_CLASSES, 0);
            while(!kNearest.empty()){
                votes[kNearest.top().second]++;
                kNearest.pop();
            }


            int majorityClass = 5;
            int maxVotes = 0;

            for(int c = 0; c < NUM_CLASSES; c++){
                if(votes[c] > maxVotes){
                   maxVotes = votes[c];
                   majorityClass = c;
                }
            }

            // WE WILL ASSUME WE DONT HAVE A ID COLUMN.
            // WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
            classifications[i][point] = majorityClass;
        }
    }

    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    // Go through the classes.
    for(int classN = 0; classN < NUM_CLASSES; classN++){
        for(int point = 0; point < classifications[classN].size(); point++){
            regularConfusionMatrix[classN][classifications[classN][point]]++;
        }
    }

    return regularConfusionMatrix;
}


//EUCLIDEAN DISTANCE OF TWO VECTORS.
float euclideanDistance(const std::vector<float>& hbCenter, const std::vector<float>& point){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = hbCenter[i] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}
