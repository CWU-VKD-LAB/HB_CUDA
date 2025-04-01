//
// Created by asnyd on 3/20/2025.
//

#include "Knn.h"
#include <vector>
#include <utility>
#include <queue>
#include <iostream>
#include <cmath>
#include <unordered_map>



// Lets make a K-nn that goes through the unclassified points and sees how close they are to being
// inside of each of the blocks. If the value for a attribute is within the bounds of the block we wont add any
// distance to the sum. If the value is outside the bounds we will add the distance to the sum.
std::vector<std::vector<long>> Knn::closeToInkNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES){
    // Basically we will do the same thing, we will just need to change our distancce thingy around.

    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    std::cout << "Field Length: " << FIELD_LENGTH << std::endl;
    if(k > hyperBlocks.size()) k = (int) sqrt(hyperBlocks.size());

    // Keep track of assignments with something
    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    // For each class of points
    for(int i = 0; i < NUM_CLASSES; i++){

        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){
            // Use a priority queue to keep track of the top k best distances
            std::priority_queue<std::pair<float, int>> kNearest;

            // Go through all the blocks and find the disstances to their centers
            for(const HyperBlock& hyperBlock : hyperBlocks){
                // Find the distance between the HB center and the unclassified data point
                
                float distance =  hyperBlock.distance_to_HB(FIELD_LENGTH, unclassifiedData[i][point].data());

                if(kNearest.size() < k){    // always add when queue is not at k yet.
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));
                }
                else if(distance < kNearest.top().first){ // Queue is big enough, and this distance is better than the worst in queue
                    kNearest.pop();    // pop the max (worst distance)
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));    // push the better distance.
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


/////
/////
/////
/////
/////
////
/////


std::vector<std::vector<long>> Knn::blockPointkNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<std::vector<std::vector<float>>> classifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES){
    
    if(k > hyperBlocks.size()) k = (int) sqrt(hyperBlocks.size());
    int FIELD_LENGTH = hyperBlocks[0].maximums.size();

    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    // Go through all the classified data of each class and mark for each point EVERY BLOCK IT IS IN
    std::unordered_map<int, std::vector<int>> pointsInBlocks;
    for(int i = 0; i < NUM_CLASSES; i++){
        for(int p = 0; p < classifiedData[i].size(); p++){
            for(int block = 0; block < hyperBlocks.size(); block++){
                if(hyperBlocks[block].inside_HB(FIELD_LENGTH, classifiedData[i][p].data())){
                    pointsInBlocks[p].push_back(block);
                }
            }
        }
    }

    // Now we want to run the K-nn on the unclassified points
     // For each class of points
    for(int i = 0; i < NUM_CLASSES; i++){

        // For each point in unclassified points
        for(int uPoint = 0; uPoint < unclassifiedData[i].size(); uPoint++){
            // Use a priority queue to keep track of the top k best distances
            std::priority_queue<std::pair<float, int>> kNearest;

            // Go through all the classified points to find the closest one and whatever class thats in use.
            for(int cPoint = 0; cPoint < classifiedData[i].size(); cPoint++){
                float distance = Knn::euclideanDistancePoints(unclassifiedData[i][uPoint], classifiedData[i][cPoint], FIELD_LENGTH);

                if(kNearest.size() < k){
                    kNearest.push(std::make_pair(distance, cPoint));
                }
                else if(distance < kNearest.top().first){
                    kNearest.pop();
                    kNearest.push(std::make_pair(distance, cPoint));
                }
            }

            // Count votes for each class
            std::vector<int> votes(NUM_CLASSES, 0);
            while(!kNearest.empty()){
                // kNearest.top().second is the index of the classified point,
                // we need to increment for each of the blocks that point falls into. (pointsInBlocks)
                for(int block : pointsInBlocks[kNearest.top().second]){
                    votes[hyperBlocks[block].classNum]++;
                }
                kNearest.pop();
            }

            int majorityClass = -1;
            int maxVotes = -1;

            for(int c = 0; c < NUM_CLASSES; c++){
                if(votes[c] > maxVotes){
                   maxVotes = votes[c];
                   majorityClass = c;
                }
            }

            // WE WILL ASSUME WE DONT HAVE A ID COLUMN.
            // WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
            classifications[i][uPoint] = majorityClass;
        }
    }

    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    // Go through the classes.
    for(int classN = 0; classN < NUM_CLASSES; classN++){
        for(int point = 0; point < classifications[classN].size(); point++)
            regularConfusionMatrix[classN][classifications[classN][point]]++;  
    }

    return regularConfusionMatrix;
}



/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
std::vector<std::vector<long>> Knn::kNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES){

    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    std::cout << "Field Length: " << FIELD_LENGTH << std::endl;

    if(k > hyperBlocks.size()) k = (int) sqrt(hyperBlocks.size());

    // Keep track of assignments with something
    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    // For each class of points
    for(int i = 0; i < NUM_CLASSES; i++){

        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){
            // Use a priority queue to keep track of the top k best distances
            std::priority_queue<std::pair<float, int>> kNearest;

            // Go through all the blocks and find the disstances to their centers
            for(const auto& hyperBlock : hyperBlocks){
                // Find the distance between the HB center and the unclassified data point
                float bottomDist = Knn::euclideanDistanceBounds(hyperBlock.minimums, unclassifiedData[i][point], FIELD_LENGTH);
                float topDist = Knn::euclideanDistanceBounds(hyperBlock.maximums, unclassifiedData[i][point], FIELD_LENGTH);


                float distance = std::min(bottomDist, topDist);

                if(kNearest.size() < k){    // always add when queue is not at k yet.
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));
                }
                else if(distance < kNearest.top().first){ // Queue is big enough, and this distance is better than the worst in queue
                    kNearest.pop();    // pop the max (worst distance)
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));    // push the better distance.
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


//EUCLIDEAN DISTANCE OF TWO VECTORS, comparing a point to a block bound (2-D vector for disjunctions)
float Knn::euclideanDistanceBounds(const std::vector<std::vector<float>>& blockBound, const std::vector<float>& point, int FIELD_LENGTH){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = blockBound[i][0] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}


//EUCLIDEAN DISTANCE OF TWO VECTORS, comparing a point to a point
float Knn::euclideanDistancePoints(const std::vector<float>& point2, const std::vector<float>& point, int FIELD_LENGTH){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = point2[i] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}

