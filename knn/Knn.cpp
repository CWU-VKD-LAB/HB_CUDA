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
#include <omp.h>
#include <algorithm>


constexpr float EPSILON = 1e-6f;

int Knn::closeToInkNN(const std::vector<float>& point, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES) {
    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    if (k > hyperBlocks.size())
        k = static_cast<int>(std::sqrt(hyperBlocks.size()));

    std::priority_queue<std::pair<float, int>> kNearest;
    for (const HyperBlock& hb : hyperBlocks) {
        float dist = hb.distance_to_HB_Avg(FIELD_LENGTH, point.data());
        if (kNearest.size() < k) {
            kNearest.emplace(dist, hb.classNum);
        } else if (dist < kNearest.top().first) {
            kNearest.pop();
            kNearest.emplace(dist, hb.classNum);
        }
    }

    std::vector<float> weightedVotes(NUM_CLASSES, 0.0f);
    while (!kNearest.empty()) {
        float dist = kNearest.top().first;
        int cls   = kNearest.top().second;
        kNearest.pop();
        float weight = (dist == 0.0f) ? 1.0f : (1.0f / (dist * dist));
        weightedVotes[cls] += weight;
    }

    return static_cast<int>(
        std::distance(
            weightedVotes.begin(),
            std::max_element(weightedVotes.begin(), weightedVotes.end())
        )
    );
}

int Knn::pureKnn(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &classifiedData, const int NUM_CLASSES, const int k) {

    int FIELD_LENGTH = point.size();

    std::priority_queue<std::pair<float, int>> kNearest; // {distance, classLabel}
    for (int trainClass = 0; trainClass < NUM_CLASSES; ++trainClass) {
        // Compare to all classified points
        for (int c = 0; c < classifiedData[trainClass].size(); ++c) {
            float distance = Knn::euclideanDistancePoints(point, classifiedData[trainClass][c], FIELD_LENGTH);

            if (kNearest.size() < k) {
                kNearest.push({distance, trainClass});
            } else if (distance < kNearest.top().first) {
                kNearest.pop();
                kNearest.push({distance, trainClass});
            }
        }
    }

    // Tally votes from the k nearest neighbors
    std::vector<int> votes(NUM_CLASSES, 0);
    while (!kNearest.empty()) {
        int cls = kNearest.top().second;
        votes[cls]++;
        kNearest.pop();
    }

    // Return the class with the highest vote count
    auto maxIt = std::max_element(votes.begin(), votes.end());
    return static_cast<int>(std::distance(votes.begin(), maxIt));

}


/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
int Knn::kNN(const std::vector<float> &point, const std::vector<HyperBlock>& hyperBlocks, int k, const int NUM_CLASSES)
{
    int FIELD_LENGTH = point.size();

    if (k > hyperBlocks.size())
        k = (int)std::sqrt(hyperBlocks.size());

    // Store all distances
    std::vector<std::pair<float, int>> allDistances;

    for (const auto& hyperBlock : hyperBlocks) {
        float bottomDist = Knn::euclideanDistanceBounds(hyperBlock.tamedMin, point, FIELD_LENGTH);
        float topDist = Knn::euclideanDistanceBounds(hyperBlock.tamedMax, point, FIELD_LENGTH);

        float distance = std::min(bottomDist, topDist);
        allDistances.emplace_back(distance, hyperBlock.classNum);
    }

    // Get the k closest elements
    std::nth_element(allDistances.begin(), allDistances.begin() + k, allDistances.end());
    std::vector<std::pair<float, int>> kNearest(allDistances.begin(), allDistances.begin() + k);

    // Find max distance in kNearest to ignore
    auto worstIt = std::max_element(kNearest.begin(), kNearest.end());
    kNearest.erase(worstIt);  // remove worst

    // Count votes excluding the farthest
    std::vector<int> votes(NUM_CLASSES, 0);
    for (const auto& pair : kNearest) {
        votes[pair.second]++;
    }

    int majorityClass = -1;
    int maxVotes = 0;

    for (int c = 0; c < NUM_CLASSES; c++) {
        if (votes[c] > maxVotes) {
            maxVotes = votes[c];
            majorityClass = c;
        }
    }

    return majorityClass;
}

//EUCLIDEAN DISTANCE OF TWO VECTORS, comparing a point to a block bound (2-D vector for disjunctions)
float Knn::euclideanDistanceBounds(const std::vector<float>& blockBound, const std::vector<float>& point, int FIELD_LENGTH){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = blockBound[i] - point[i];
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

/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
int Knn::mergableKNN(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>> &trainingData, std::vector<HyperBlock> &hyperBlocks, int NUM_CLASSES) {

    int FIELD_LENGTH = point.size();

    // now we just take our training data, and first break it up by column exactly how we did in in the interval HB generation portion.
    std::vector<std::vector<DataATTR>> dataByAttribute = IntervalHyperBlock::separateByAttribute(trainingData, FIELD_LENGTH);
    for (HyperBlock &block : hyperBlocks) {
        // now we set up the top and bottom pairs for each block just to make sure that the bounds are made.
        block.topBottomPairs.resize(FIELD_LENGTH);

        // For each attribute (column), perform a binary search. we are just making the bounds of each attribute for each HB in index form.
        for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

            // Get the sorted vector of DataATTR for this attribute.
            const std::vector<DataATTR>& columnPoints = dataByAttribute[attribute];

            // Our target interval for this attribute (assuming each is stored as a one-element vector)
            float lowerVal = block.minimums[attribute][0];
            float upperVal = block.maximums[attribute][0];

            // lower_bound: first element whose value is not less than lowerVal.
            auto lowIt = lower_bound(columnPoints.begin(), columnPoints.end(), lowerVal,
                [](const DataATTR &d, float value) {
                    return d.value < value;
            });

            // upper_bound: first element whose value is greater than upperVal.
            auto highIt = upper_bound(columnPoints.begin(), columnPoints.end(), upperVal,
                [](float value, const DataATTR &d) {
                    return value < d.value;
            });

            // Determine indices:
            // If lowIt reached the end, then no elements are ≥ lowerVal.
            int lowIndex = (lowIt != columnPoints.end()) ? static_cast<int>(lowIt - columnPoints.begin()) : -1;

            // For the topmost index, we want the last index in the range.
            // upper_bound returns the first element greater than upperVal.
            // If highIt equals columnPoints.begin(), then even the first element is greater than upperVal.
            int highIndex = (highIt != columnPoints.begin()) ? static_cast<int>(highIt - columnPoints.begin()) - 1 : -1;

            // Store the pair of indices (bottommost, topmost) for this attribute.
            block.topBottomPairs[attribute] = std::make_pair(lowIndex, highIndex);
        }
    }

    // tells us which index this point would be in in the sorted columns if inserted. we don't actually put it in though, that would change the data.
    std::vector<int> pointIndicesByColumn(FIELD_LENGTH, -1);

    // set up our vector which tells us which indices the point would land at.
    for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

        float pointVal = point[attribute];

        const std::vector<DataATTR> &columnPoints = dataByAttribute[attribute];
        // find the index which our point would go at in this column.
        auto indexIt = std::lower_bound(
            columnPoints.begin(), columnPoints.end(), pointVal,
            [](const DataATTR &d, float value) { return d.value < value; });

        int index = std::distance(columnPoints.begin(), indexIt);
        if (index == columnPoints.size())
            index--;
        pointIndicesByColumn[attribute] = index;
    }

    int bestClass = -1;
    float bestAcc = 0.0f;
    int bestBlockSize = 0;
    for (HyperBlock &block : hyperBlocks) {

        // now that our block is made, we can determine how many wrong points are going to fall in if we included this guy.
        // use the merge check to determine how "mergeable" this point is to each block, and we use the "most mergeable" class.
        float acc = mergeCheck(pointIndicesByColumn, block, dataByAttribute);

        // if the accuracy is better, or within .1% and we have a bigger block, we are using this block.
        if (acc > bestAcc || (std::fabs(acc - bestAcc) < .001f && block.size > bestBlockSize) ) {
            bestClass = block.classNum;
            bestAcc = acc;
            bestBlockSize = block.size;
        }
    }

    return bestClass;
}

// returns amount of wrong class points which would fall into our HB bounds if we did include this point in the block.
// doesn't change the block itself, just changes it hypothetically. we will call this the justin herbert algorithm.
// same logic as the checking mergeable we implemented for the interval Hyper without cuda merging stuff, but returns the wrong count instead of just true or false.
//  ───────────────────────────────────────────────────────────────
//  Evaluate the accuracy of HyperBlock `hb` if we enlarged it just
//  enough to include the unclassified point whose per‑column index
//  is given in insertIdx
//  ───────────────────────────────────────────────────────────────
float Knn::mergeCheck(std::vector<int> &insertIdx, HyperBlock &hb, std::vector<std::vector<DataATTR>> &columns) {
    const int D = columns.size();

    // 1.  Make a local copy of bounds and enlarge with the new point.
    std::vector<std::pair<int,int>> bounds = hb.topBottomPairs;
    for (int d = 0; d < D; ++d) {
        bounds[d].first  = std::min(bounds[d].first,  insertIdx[d]);
        bounds[d].second = std::max(bounds[d].second, insertIdx[d]);
    }

    // 2.  Choose the attribute with the SMALLEST interval → fewest candidates.
    int pivot = 0;
    std::size_t span = std::numeric_limits<std::size_t>::max();
    for (int d = 0; d < D; ++d) {
        std::size_t cur = bounds[d].second - bounds[d].first + 1;
        if (cur < span) { span = cur; pivot = d; }
    }

    std::size_t wrong = 0;

    // 3.  Scan candidate rows in the pivot column only.
    const auto& col = columns[pivot];
    for (int idx = bounds[pivot].first; idx <= bounds[pivot].second; ++idx) {
        const DataATTR& attr = col[idx];

        // Skip rows that already belong to the block’s class.
        if (attr.classNum == hb.classNum) continue;

        // Check every other dimension quickly; bail on first failure.
        bool inside = true;
        int row = attr.classIndex;
        for (int d = 0; d < D; ++d) {

            if (d == pivot)
                continue;

            float v = columns[d][row].value;
            bool ok = false;
            const auto& mins = hb.minimums[d];
            const auto& maxs = hb.maximums[d];

            for (std::size_t c = 0; c < mins.size(); ++c) {
                if (v >= mins[c] && v <= maxs[c]) {
                    ok = true;
                    break;
                }
            }

            // new point might have extended the bound
            if (!ok && v >= columns[d][insertIdx[d]].value && v <= columns[d][insertIdx[d]].value) ok = true;

            if (!ok) {
                inside = false;
                break;
            }
        }

        if (inside) ++wrong;
    }

    // 4.  Accuracy if we accept the point (+1 correct) and wrong extras.
    return float(hb.size + 1) / float(hb.size + 1 + wrong);
}


int Knn::bruteMergable(const std::vector<float>& point, const std::vector<std::vector<std::vector<float>>>& classifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES) {
    constexpr float EPSILON = 1e-6f;
    constexpr int NUM_SAMPLES = 300; // Sampling limit for each class

    int FIELD_LENGTH = hyperBlocks[0].maximums.size();

    // Pre-process hyperblocks by class for faster access
    std::vector<std::vector<size_t>> blocksByClass(NUM_CLASSES);
    for (size_t i = 0; i < hyperBlocks.size(); i++) {
        blocksByClass[hyperBlocks[i].classNum].push_back(i);
    }

    // Stores min and max average impurity for each class of blocks
    std::vector<std::pair<float, float>> classImpurityRange(
        NUM_CLASSES,
        { std::numeric_limits<float>::max(), std::numeric_limits<float>::min() }
    );

    // Store average impurity metrics for each class combination
    std::vector<std::vector<float>> classMetrics(
        NUM_CLASSES,
        std::vector<float>(NUM_CLASSES, 0.0f)
    );

    // Create a single vector of all classified points with their class labels
    struct ClassifiedPoint {
        std::vector<float> point;
        int classLabel;
    };
    std::vector<ClassifiedPoint> allClassifiedPoints;
    allClassifiedPoints.reserve(std::accumulate(
        classifiedData.begin(),
        classifiedData.end(),
        0,
        [](size_t sum, const std::vector<std::vector<float>>& vec) { return sum + vec.size(); }
    ));
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        for (const auto& p : classifiedData[cls]) {
            allClassifiedPoints.push_back({ p, cls });
        }
    }

    // First phase: Gather metrics using sampling
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::vector<std::vector<float>> classSamples;
        int sampleCount = std::min(NUM_SAMPLES, static_cast<int>(classifiedData[i].size()));
        std::vector<int> indices(classifiedData[i].size());
        std::iota(indices.begin(), indices.end(), 0);

        unsigned int seed = std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
        for (int j = 0; j < sampleCount; ++j) {
            classSamples.push_back(classifiedData[i][indices[j]]);
        }

        std::vector<std::pair<float, int>> threadMetrics(NUM_CLASSES, { 0.0f, 0 });

        for (int targetClass = 0; targetClass < NUM_CLASSES; ++targetClass) {
            if (blocksByClass[targetClass].empty()) continue;

            float totalImpurity = 0.0f;
            int validMerges = 0;

            for (const auto& sample : classSamples) {
                for (size_t blockIdx : blocksByClass[targetClass]) {
                    const auto& block = hyperBlocks[blockIdx];

                    bool anyChanges = false;
                    std::vector<std::pair<float, float>> expansionRanges(FIELD_LENGTH);
                    std::vector<bool> hasExpansion(FIELD_LENGTH, false);

                    for (int attr = 0; attr < FIELD_LENGTH; ++attr) {
                        if (sample[attr] < block.minimums[attr][0]) {
                            expansionRanges[attr] = { sample[attr], block.minimums[attr][0] };
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        } else if (sample[attr] > block.maximums[attr][0]) {
                            expansionRanges[attr] = { block.maximums[attr][0], sample[attr] };
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        }
                    }
                    if (!anyChanges) continue;

                    int rightCls = 0, wrongCls = 0;
                    for (const auto& cp : allClassifiedPoints) {
                        bool insideBlock = true;
                        for (int attr = 0; attr < FIELD_LENGTH; ++attr) {
                            if (cp.point[attr] < block.minimums[attr][0] - EPSILON ||
                                cp.point[attr] > block.maximums[attr][0] + EPSILON) {
                                insideBlock = false;
                                break;
                            }
                        }
                        if (insideBlock) continue;

                        bool inExpansion = false;
                        for (int attr = 0; attr < FIELD_LENGTH; ++attr) {
                            if (hasExpansion[attr] &&
                                cp.point[attr] >= expansionRanges[attr].first - EPSILON &&
                                cp.point[attr] <= expansionRanges[attr].second + EPSILON) {
                                inExpansion = true;
                                break;
                            }
                        }
                        if (!inExpansion) continue;

                        if (cp.classLabel == targetClass) ++rightCls;
                        else ++wrongCls;
                    }

                    if (rightCls + wrongCls > 0) {
                        float impurity = static_cast<float>(wrongCls) / (rightCls + wrongCls);
                        totalImpurity += impurity;
                        validMerges++;
                    }
                }
            }

            if (validMerges > 0) {
                threadMetrics[targetClass] = { totalImpurity, validMerges };
            }
        }

        #pragma omp critical
        {
            for (int targetClass = 0; targetClass < NUM_CLASSES; ++targetClass) {
                // Instead of structured bindings, split manually into variables
                float sumImp = threadMetrics[targetClass].first;
                int count = threadMetrics[targetClass].second;
                if (count > 0) {
                    float avgImp = sumImp / count;
                    classMetrics[targetClass][i] = avgImp;
                    classImpurityRange[targetClass].first  = std::min(classImpurityRange[targetClass].first,  avgImp);
                    classImpurityRange[targetClass].second = std::max(classImpurityRange[targetClass].second, avgImp);
                }
            }
        }
    }

    // Second phase: classify the single point
    std::vector<float> bestImpurities(NUM_CLASSES, std::numeric_limits<float>::max());
    for (int targetClass = 0; targetClass < NUM_CLASSES; ++targetClass) {
        if (blocksByClass[targetClass].empty())
            continue;

        for (size_t blockIdx : blocksByClass[targetClass]) {
            const auto& block = hyperBlocks[blockIdx];

            bool inside = true;
            for (int a = 0; a < FIELD_LENGTH; ++a) {
                if (point[a] < block.minimums[a][0] - EPSILON ||
                    point[a] > block.maximums[a][0] + EPSILON) {
                    inside = false;
                    break;
                }
            }
            if (inside) {
                bestImpurities[targetClass] = 0.0f;
                break;
            }

            bool anyChange = false;
            std::vector<std::pair<float, float>> expansion(FIELD_LENGTH);
            std::vector<bool> useExp(FIELD_LENGTH,false);
            for (int a = 0; a < FIELD_LENGTH; ++a) {
                if (point[a] < block.minimums[a][0]) {
                    expansion[a] = { point[a], block.minimums[a][0] };
                    useExp[a]    = true;
                    anyChange    = true;
                } else if (point[a] > block.maximums[a][0]) {
                    expansion[a] = { block.maximums[a][0], point[a] };
                    useExp[a]    = true;
                    anyChange    = true;
                }
            }
            if (!anyChange) continue;

            int rightCls = 0, wrongCls = 0;
            for (auto& cp : allClassifiedPoints) {
                bool inBlk = true;
                for (int a = 0; a < FIELD_LENGTH; ++a) {
                    if (cp.point[a] < block.minimums[a][0] - EPSILON ||
                        cp.point[a] > block.maximums[a][0] + EPSILON) {
                        inBlk = false;
                        break;
                    }
                }
                if (inBlk) continue;

                bool inExp = false;
                for (int a = 0; a < FIELD_LENGTH; ++a) {
                    if (useExp[a] &&
                        cp.point[a] >= expansion[a].first  - EPSILON &&
                        cp.point[a] <= expansion[a].second + EPSILON) {
                        inExp = true;
                        break;
                    }
                }
                if (!inExp) continue;

                if (cp.classLabel == targetClass) ++rightCls;
                else ++wrongCls;
            }

            float imp = 1.0f;
            if (rightCls + wrongCls > 0)
                imp = static_cast<float>(wrongCls) / (rightCls + wrongCls);
            else if (anyChange)
                imp = 0.5f;

            bestImpurities[targetClass] = std::min(bestImpurities[targetClass], imp);
        }
    }

    // Normalize scores
    std::vector<float> normScores(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float lo = classImpurityRange[c].first;
        float hi = classImpurityRange[c].second;
        if (hi - lo < EPSILON)
            normScores[c] = (bestImpurities[c] < lo + EPSILON) ? 0.0f : 1.0f;
        else
            normScores[c] = (bestImpurities[c] - lo) / (hi - lo);
    }

    // Return class with lowest score
    auto it = std::min_element(normScores.begin(), normScores.end());
    return static_cast<int>(std::distance(normScores.begin(), it));
}


bool Knn::isInside(const std::vector<float>& point, const std::vector<std::vector<float>>& fMins, const std::vector<std::vector<float>>& fMaxes){
    for (int i = 0; i < point.size(); ++i) {
        bool insideDim = false;

        // Go through all possible disjunctives
        for (int r = 0; r < fMins[i].size(); ++r) {
            if ((point[i] + EPSILON >= fMins[i][r]) && (point[i] - EPSILON <= fMaxes[i][r])) {
                insideDim = true;
                break;
            }
        }

        if (!insideDim) return false;
    }

    return true;
}

// returns the distance between two points, by attribute. This is helpful when we are not able to compare the distances by aggregating in any way. Useful for heterogenous data.
std::vector<float> Knn::losslessDistance(const std::vector<float> &seedPoint, const std::vector<float> &trainPoint) {
    std::vector<float> distances(seedPoint.size(), 0);
    for (int attribute = 0; attribute < seedPoint.size(); attribute++)
        distances[attribute] = std::fabs(seedPoint[attribute] - trainPoint[attribute]);

    return distances;
}

std::vector<float> Knn::computeStdDeviations(const std::vector<std::vector<std::vector<float>>>& trainData) {
    // total number of points
    int numPoints = 0;
    for (const auto& classPoints : trainData)
        numPoints += static_cast<int>(classPoints.size());
    if (numPoints == 0) return {};

    int FIELD_LENGTH = static_cast<int>(trainData[0][0].size());

    // compute means
    std::vector<float> means(FIELD_LENGTH, 0.0f);
    for (const auto& classPoints : trainData) {
        for (const auto& point : classPoints) {
            for (int j = 0; j < FIELD_LENGTH; ++j) {
                means[j] += point[j];
            }
        }
    }
    for (int j = 0; j < FIELD_LENGTH; ++j) {
        means[j] /= numPoints;
    }

    // compute sum of squared deviations
    std::vector<float> sqDiffs(FIELD_LENGTH, 0.0f);
    for (const auto& classPoints : trainData) {
        for (const auto& point : classPoints) {
            for (int j = 0; j < FIELD_LENGTH; ++j) {
                float d = point[j] - means[j];
                sqDiffs[j] += d * d;
            }
        }
    }

    // finalize to standard deviations
    std::vector<float> stddevs(FIELD_LENGTH);
    for (int j = 0; j < FIELD_LENGTH; ++j) {
        stddevs[j] = std::sqrt(sqDiffs[j] / numPoints);

        // if you want sample std-dev instead, use:
        // stddevs[j] = std::sqrt(sqDiffs[j] / (numPoints - 1));
    }
    return stddevs;
}

// this needs to get reset to false every time we bring in a new dataset
bool Knn::deviationsComputed = false;
int Knn::thresholdKNN(const std::vector<float> &point, const std::vector<std::vector<std::vector<float>>>& classifiedData, int NUM_CLASSES, int k, float threshold) {

    // Number of attributes in the point
    int FIELD_LENGTH = point.size();


    // Only compute once, on the first call
    static std::vector<float> deviations;
    if (!deviationsComputed) {
        deviations = computeStdDeviations(classifiedData);
        deviationsComputed = true;
    }

    // Count total number of training points
    int totalTrainingPoints = 0;
    for (const auto& classPoints : classifiedData) {
        totalTrainingPoints += classPoints.size();
    }

    // Build a list of (similarity, classLabel) pairs
    std::vector<std::pair<int,int>> sims;
    sims.reserve(totalTrainingPoints);
    for (int trainClass = 0; trainClass < NUM_CLASSES; ++trainClass) {
        for (const auto &trainPoint : classifiedData[trainClass]) {
            int sim = 0;
            std::vector<float> diffs = losslessDistance(point, trainPoint);
            for (int att = 0; att < FIELD_LENGTH; ++att) {
                if (diffs[att] < deviations[att] * threshold) {
                    ++sim;
                }
            }
            sims.emplace_back(sim, trainClass);
        }
    }

    // Sort descending by similarity
    std::sort(sims.begin(), sims.end(),[](const std::pair<int,int>& a, const std::pair<int,int>& b) { return a.first > b.first;});

    // Find cutoff similarity at rank k. this way if we have a tie, we include a few extras
    int cutoffIdx = std::min(static_cast<int>(sims.size()) - 1, k - 1);
    int cutoffSim = sims[cutoffIdx].first;

    // Vote on all entries with sim >= cutoffSim
    std::vector<int> votes(NUM_CLASSES, 0);
    for (const std::pair<int,int>& pr : sims) {
        if (pr.first < cutoffSim)
            break;
        votes[pr.second]++;
    }

    // Return the index of the class with the most votes
    return std::distance(votes.begin(),std::max_element(votes.begin(), votes.end()));
}