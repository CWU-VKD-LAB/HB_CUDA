#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include <future>
#include <thread>
#define M_PI 3.14159265358979323846
// our flag for if we want to apply arc cos to each vector or not. In DV it is applied,
// and this gets us slightly better blocks.
#define USE_TRIG
using namespace std;

// ------------------------------------------------------------
// 1) A helper to compute the partial sum of a data set's means
// ------------------------------------------------------------
vector<float> partialMean(const vector<vector<float>>& data,
                               int start, int end)
{
    if (start >= end) {
        return {};
    }
    int numFeatures = data[0].size();
    vector<float> localSum(numFeatures, 0.0f);

    for (int idx = start; idx < end; idx++) {
        const auto& row = data[idx];
        for (int j = 0; j < numFeatures; j++) {
            localSum[j] += row[j];
        }
    }
    return localSum;
}

// --------------------------------------------------------------
// 2) A helper to compute a partial scatter matrix for [start, end)
// --------------------------------------------------------------
vector<vector<float>> partialScatter(const vector<vector<float>>& data, const vector<float>& mean, int start, int end)
{
    int numFeatures = static_cast<int>(mean.size());
    // Create a local scatter matrix
    vector<vector<float>> localSw(numFeatures, vector<float>(numFeatures, 0.0f));

    for (int idx = start; idx < end; idx++) {
        const auto& x = data[idx];
        for (int i = 0; i < numFeatures; i++) {
            float diff_i = x[i] - mean[i];
            for (int j = 0; j < numFeatures; j++) {
                localSw[i][j] += diff_i * (x[j] - mean[j]);
            }
        }
    }
    return localSw;
}

// --------------------------------------------------------------
// Matrix inverse using naive Gauss-Jordan elimination
// (Consider a library for large matrices.)
// --------------------------------------------------------------
vector<vector<float>> inverse(const vector<vector<float>>& matrix) {
    int n = matrix.size();
    vector<vector<float>> augmented(n, vector<float>(2 * n, 0.0f));

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0f;  // identity portion
    }

    // Forward elimination
    for (int i = 0; i < n; i++) {
        float pivot = augmented[i][i];
        if (fabs(pivot) < 1e-12f) {
            cerr << "Matrix is singular or near-singular.\n";
            exit(EXIT_FAILURE);
        }
        // Normalize pivot row
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        // Eliminate in other rows
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse
    vector<vector<float>> inv(n, vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i][j] = augmented[i][j + n];
        }
    }
    return inv;
}

// --------------------------------------------------------------
// Simple matrix-vector multiply
// --------------------------------------------------------------
vector<float> matrixVectorMultiply(const vector<vector<float>>& A, const vector<float>& v) {
    int rows = A.size();
    int cols = A[0].size();
    vector<float> result(rows, 0.0f);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

// --------------------------------------------------------------
// Helper: dot product of two vectors
// --------------------------------------------------------------
inline float dotProduct(const vector<float>& a, const vector<float>& b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Define a struct to hold both the weight vector and its bias.
struct LDAClassifier {
    vector<float> w;
    float bias;
};

// --------------------------------------------------------------
// 3) Compute a single LDA vector w (and bias) for classA vs. classB
//    with partial parallelization via async
// --------------------------------------------------------------
LDAClassifier computeBinaryLDA(const vector<vector<float>>& classA,
                               const vector<vector<float>>& classB)
{
    // We assume both classA and classB have at least one row
    int numFeatures = classA[0].size();

    // Decide how many threads to launch
    unsigned concurrency = thread::hardware_concurrency();
    if (concurrency == 0) {
        concurrency = 4; // fallback if hardware_concurrency is not available
    }

    // ---------------------------
    // 3.1) Compute mean of classA
    // ---------------------------
    auto meanA_future = async(launch::async,
                                   [&, concurrency]() {
        int totalSamples = static_cast<int>(classA.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        // Launch partial sums
        vector<future<vector<float>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = min(start + chunkSize, totalSamples);
            futures.push_back(async(launch::async,
                partialMean, cref(classA), start, end));
            start = end;
        }
        // Combine partial sums
        vector<float> sumA(numFeatures, 0.0f);
        for (auto &f : futures) {
            auto localSum = f.get();
            for (int i = 0; i < numFeatures; i++) {
                sumA[i] += localSum[i];
            }
        }
        // Divide by totalSamples to get mean
        for (int i = 0; i < numFeatures; i++) {
            sumA[i] /= static_cast<float>(totalSamples);
        }
        return sumA;
    });

    // ---------------------------
    // 3.2) Compute mean of classB
    // ---------------------------
    auto meanB_future = async(launch::async,
                                   [&, concurrency]() {
        int totalSamples = static_cast<int>(classB.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        // Launch partial sums
        vector<future<vector<float>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = min(start + chunkSize, totalSamples);
            futures.push_back(async(launch::async,
                partialMean, cref(classB), start, end));
            start = end;
        }
        // Combine partial sums
        vector<float> sumB(numFeatures, 0.0f);
        for (auto &f : futures) {
            auto localSum = f.get();
            for (int i = 0; i < numFeatures; i++) {
                sumB[i] += localSum[i];
            }
        }
        // Divide by totalSamples to get mean
        for (int i = 0; i < numFeatures; i++) {
            sumB[i] /= static_cast<float>(totalSamples);
        }
        return sumB;
    });

    // Wait for means
    vector<float> meanA = meanA_future.get();
    vector<float> meanB = meanB_future.get();

    // --------------------------------
    // 3.3) Compute scatter for classA
    // --------------------------------
    vector<vector<float>> Sw(numFeatures, vector<float>(numFeatures, 0.0f));

    auto scatterA_future = async(launch::async,
                                      [&, concurrency]() {
        vector<vector<float>> localSw(numFeatures, vector<float>(numFeatures, 0.0f));
        int totalSamples = static_cast<int>(classA.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        vector<future<vector<vector<float>>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = min(start + chunkSize, totalSamples);
            futures.push_back(async(launch::async,
                partialScatter, cref(classA), cref(meanA), start, end));
            start = end;
        }
        // Combine partial results
        for (auto &f : futures) {
            auto localPart = f.get();
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    localSw[i][j] += localPart[i][j];
                }
            }
        }
        return localSw;
    });

    // --------------------------------
    // 3.4) Compute scatter for classB
    // --------------------------------
    auto scatterB_future = async(launch::async,
                                      [&, concurrency]() {
        vector<vector<float>> localSw(numFeatures, vector<float>(numFeatures, 0.0f));
        int totalSamples = static_cast<int>(classB.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        vector<future<vector<vector<float>>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = min(start + chunkSize, totalSamples);
            futures.push_back(async(launch::async,
                partialScatter, cref(classB), cref(meanB), start, end));
            start = end;
        }
        // Combine partial results
        for (auto &f : futures) {
            auto localPart = f.get();
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    localSw[i][j] += localPart[i][j];
                }
            }
        }
        return localSw;
    });

    // Merge scatter from classA and classB
    auto SwA = scatterA_future.get();
    auto SwB = scatterB_future.get();
    for (int i = 0; i < numFeatures; i++) {
        for (int j = 0; j < numFeatures; j++) {
            Sw[i][j] = SwA[i][j] + SwB[i][j];
        }
    }

    // REGULARIZE TO AVOID THE MATRIX BEING SINGULAR.
    float lambda = 1e-4f;
    for (int i = 0; i < numFeatures; i++) {
        Sw[i][i] += lambda;
    }

    // 3.5) Compute w = Sw^-1 * (meanB - meanA)
    vector<vector<float>> SwInv = inverse(Sw);
    vector<float> meanDiff(numFeatures, 0.0f);
    for (int j = 0; j < numFeatures; j++) {
        meanDiff[j] = meanB[j] - meanA[j];
    }
    vector<float> w = matrixVectorMultiply(SwInv, meanDiff);

    // 3.6) Normalize w
    float normVal = 0.0f;
    for (float val : w) {
        normVal += val * val;
    }
    normVal = sqrt(normVal);
    if (normVal > 1e-12f) {
        for (float &val : w) {
            val /= normVal;
        }
    }

    // 3.7) Compute bias: midpoint between classA and classB means
    float projMeanA = dotProduct(w, meanA);
    float projMeanB = dotProduct(w, meanB);
    float bias = (projMeanA + projMeanB) / 2.0f;

    // Flip w and bias if classA's projection is lower than classB's
    if (projMeanA < projMeanB) {
        for (float &val : w) {
            val = -val;
        }
        bias = -bias;
    }

    LDAClassifier classifier;
    classifier.w = move(w);
    classifier.bias = bias;
    return classifier;
}

// predictClass: for a given sample, compute (w^T x - bias) and return the class with the highest score.
int predictClass(const vector<LDAClassifier>& classifiers, const vector<float>& sample) {
    int bestClass = -1;
    float bestScore = -1e9f; // a very low starting score
    for (size_t i = 0; i < classifiers.size(); i++) {
        float score = dotProduct(classifiers[i].w, sample) - classifiers[i].bias;
        if (score > bestScore) {
            bestScore = score;
            bestClass = static_cast<int>(i);
        }
    }
    return bestClass;
}

// testMultiClassAccuracy: loops over each sample in inputData (labeled by class),
// predicts its class using predictClass, and computes overall accuracy.
vector<int> testMultiClassAccuracy(const vector<LDAClassifier>& classifiers,
                                        const vector<vector<vector<float>>>& inputData) {
    int numClasses = inputData.size();
    vector<int> correctPerClass(numClasses, 0);
    vector<int> totalPerClass(numClasses, 0);
    int totalCorrect = 0;
    int totalSamples = 0;

    for (int i = 0; i < numClasses; ++i) {
        for (const auto& sample : inputData[i]) {
            int predicted = predictClass(classifiers, sample);
            if (predicted == i) {
                ++correctPerClass[i];
                ++totalCorrect;
            }
            ++totalPerClass[i];
            ++totalSamples;
        }
    }

    // Compute per-class accuracy and associate with class index
    vector<pair<int, float>> classAcc;
    for (int i = 0; i < numClasses; ++i) {
        float acc = totalPerClass[i] > 0 ? static_cast<float>(correctPerClass[i]) / totalPerClass[i] : 0.0f;
        classAcc.push_back(make_pair(i, acc));
    }

    // Sort by descending accuracy
    sort(classAcc.begin(), classAcc.end(),
              [](const pair<int, float>& a, const pair<int, float>& b) {
                  return a.second > b.second;
              });

    // Extract sorted class indices
    vector<int> ordering;
    for (const auto& pair : classAcc) {
        ordering.push_back(pair.first);
    }

    for(int i = 0; i < ordering.size(); i++) {
      cout << ordering[i] << " " << endl;
    }
    return ordering;
}

// --------------------------------------------------------------
// “One-vs-Rest” LDA for multi-class
// --------------------------------------------------------------
/**
 * Given inputData where inputData[i] = all samples of class i,
 * run a 2-class LDA for each class i vs. all other classes combined.
 *
 * Returns a vector of separation vectors (with optional arccos normalization),
 * one per class.
 */
pair<vector<vector<float>>, vector<int>> linearDiscriminantAnalysis(const vector<vector<vector<float>>>& inputData) {
    int numClasses = inputData.size();
    vector<LDAClassifier> classifiers(numClasses); // one classifier per class

    // For each class i, gather its samples in classA and gather all other samples in classB
    for (int i = 0; i < numClasses; i++) {
        // classA = the samples of class i
        const auto& classA = inputData[i];

        // classB = union of samples of all other classes
        vector<vector<float>> classB;
        for (int j = 0; j < numClasses; j++) {
            if (j == i) continue; // skip class i
            classB.insert(classB.end(), inputData[j].begin(), inputData[j].end());
        }
        // Compute the 2-class LDA classifier for i vs. rest (parallel inside)
        classifiers[i] = computeBinaryLDA(classA, classB);
    }

    // Optionally apply arc-cos transformation to the final w's
    vector<vector<float>> separationVectors;
    for (LDAClassifier &classifier : classifiers) {
#ifdef USE_TRIG
        // Copy the classifier weights
        vector<float> normalizedVector = classifier.w;

        float maxVal = *max_element(normalizedVector.begin(), normalizedVector.end());
        float minVal = *min_element(normalizedVector.begin(), normalizedVector.end());

        // Avoid div-by-zero if all values are the same
        float range = maxVal - minVal;
        if (range == 0.0f) {
            range = 1.0f;
        }

        constexpr float radToDeg = 180.0f / M_PI;
        for (auto& val : normalizedVector) {
            float norm = (val - minVal) / range;
            // arccos(1) -> 0°, arccos(0) -> 90°
            val = acos(norm) * radToDeg;
        }
        separationVectors.push_back(move(normalizedVector));
#else
        separationVectors.push_back(classifier.w);
#endif
    }

    // Evaluate accuracy
    vector<int> classOrder = testMultiClassAccuracy(classifiers, inputData);

    return make_pair(separationVectors, classOrder);
}
