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

// ------------------------------------------------------------
// 1) A helper to compute the partial sum of a data set's means
// ------------------------------------------------------------
std::vector<float> partialMean(const std::vector<std::vector<float>>& data,
                               int start, int end)
{
    if (start >= end) {
        return {};
    }
    int numFeatures = data[0].size();
    std::vector<float> localSum(numFeatures, 0.0f);

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
std::vector<std::vector<float>> partialScatter(const std::vector<std::vector<float>>& data, const std::vector<float>& mean, int start, int end)
{
    int numFeatures = static_cast<int>(mean.size());
    // Create a local scatter matrix
    std::vector<std::vector<float>> localSw(numFeatures, std::vector<float>(numFeatures, 0.0f));

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
std::vector<std::vector<float>> inverse(const std::vector<std::vector<float>>& matrix) {
    int n = matrix.size();
    std::vector<std::vector<float>> augmented(n, std::vector<float>(2 * n, 0.0f));

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
        if (std::fabs(pivot) < 1e-12f) {
            std::cerr << "Matrix is singular or near-singular.\n";
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
    std::vector<std::vector<float>> inv(n, std::vector<float>(n, 0.0f));
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
std::vector<float> matrixVectorMultiply(const std::vector<std::vector<float>>& A, const std::vector<float>& v) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<float> result(rows, 0.0f);
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
inline float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Define a struct to hold both the weight vector and its bias.
struct LDAClassifier {
    std::vector<float> w;
    float bias;
};

// --------------------------------------------------------------
// 3) Compute a single LDA vector w (and bias) for classA vs. classB
//    with partial parallelization via std::async
// --------------------------------------------------------------
LDAClassifier computeBinaryLDA(const std::vector<std::vector<float>>& classA,
                               const std::vector<std::vector<float>>& classB)
{
    // We assume both classA and classB have at least one row
    int numFeatures = classA[0].size();

    // Decide how many threads to launch
    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) {
        concurrency = 4; // fallback if hardware_concurrency is not available
    }

    // ---------------------------
    // 3.1) Compute mean of classA
    // ---------------------------
    auto meanA_future = std::async(std::launch::async,
                                   [&, concurrency]() {
        int totalSamples = static_cast<int>(classA.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        // Launch partial sums
        std::vector<std::future<std::vector<float>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = std::min(start + chunkSize, totalSamples);
            futures.push_back(std::async(std::launch::async,
                partialMean, std::cref(classA), start, end));
            start = end;
        }
        // Combine partial sums
        std::vector<float> sumA(numFeatures, 0.0f);
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
    auto meanB_future = std::async(std::launch::async,
                                   [&, concurrency]() {
        int totalSamples = static_cast<int>(classB.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        // Launch partial sums
        std::vector<std::future<std::vector<float>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = std::min(start + chunkSize, totalSamples);
            futures.push_back(std::async(std::launch::async,
                partialMean, std::cref(classB), start, end));
            start = end;
        }
        // Combine partial sums
        std::vector<float> sumB(numFeatures, 0.0f);
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
    std::vector<float> meanA = meanA_future.get();
    std::vector<float> meanB = meanB_future.get();

    // --------------------------------
    // 3.3) Compute scatter for classA
    // --------------------------------
    std::vector<std::vector<float>> Sw(numFeatures, std::vector<float>(numFeatures, 0.0f));

    auto scatterA_future = std::async(std::launch::async,
                                      [&, concurrency]() {
        std::vector<std::vector<float>> localSw(numFeatures, std::vector<float>(numFeatures, 0.0f));
        int totalSamples = static_cast<int>(classA.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        std::vector<std::future<std::vector<std::vector<float>>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = std::min(start + chunkSize, totalSamples);
            futures.push_back(std::async(std::launch::async,
                partialScatter, std::cref(classA), std::cref(meanA), start, end));
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
    auto scatterB_future = std::async(std::launch::async,
                                      [&, concurrency]() {
        std::vector<std::vector<float>> localSw(numFeatures, std::vector<float>(numFeatures, 0.0f));
        int totalSamples = static_cast<int>(classB.size());
        int chunkSize = (totalSamples + concurrency - 1) / concurrency;

        std::vector<std::future<std::vector<std::vector<float>>>> futures;
        int start = 0;
        while (start < totalSamples) {
            int end = std::min(start + chunkSize, totalSamples);
            futures.push_back(std::async(std::launch::async,
                partialScatter, std::cref(classB), std::cref(meanB), start, end));
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
    std::vector<std::vector<float>> SwInv = inverse(Sw);
    std::vector<float> meanDiff(numFeatures, 0.0f);
    for (int j = 0; j < numFeatures; j++) {
        meanDiff[j] = meanB[j] - meanA[j];
    }
    std::vector<float> w = matrixVectorMultiply(SwInv, meanDiff);

    // 3.6) Normalize w
    float normVal = 0.0f;
    for (float val : w) {
        normVal += val * val;
    }
    normVal = std::sqrt(normVal);
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
    classifier.w = std::move(w);
    classifier.bias = bias;
    return classifier;
}

// predictClass: for a given sample, compute (w^T x - bias) and return the class with the highest score.
int predictClass(const std::vector<LDAClassifier>& classifiers, const std::vector<float>& sample) {
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
void testMultiClassAccuracy(const std::vector<LDAClassifier>& classifiers,
                            const std::vector<std::vector<std::vector<float>>>& inputData) {
    int numClasses = inputData.size();
    int totalSamples = 0;
    int correct = 0;

    for (int i = 0; i < numClasses; i++) {
        for (const auto &sample: inputData[i]) {
            int predicted = predictClass(classifiers, sample);
            if (predicted == i) {
                correct++;
            }
            totalSamples++;
        }
    }

    float accuracy = static_cast<float>(correct) / totalSamples;
    std::cout << "Overall multi-class accuracy: " << accuracy * 100.0f << "%" << std::endl;
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
std::vector<std::vector<float>> linearDiscriminantAnalysis(const std::vector<std::vector<std::vector<float>>>& inputData) {
    int numClasses = inputData.size();
    std::vector<LDAClassifier> classifiers(numClasses); // one classifier per class

    // For each class i, gather its samples in classA and gather all other samples in classB
    for (int i = 0; i < numClasses; i++) {
        // classA = the samples of class i
        const auto& classA = inputData[i];

        // classB = union of samples of all other classes
        std::vector<std::vector<float>> classB;
        for (int j = 0; j < numClasses; j++) {
            if (j == i) continue; // skip class i
            classB.insert(classB.end(), inputData[j].begin(), inputData[j].end());
        }
        // Compute the 2-class LDA classifier for i vs. rest (parallel inside)
        classifiers[i] = computeBinaryLDA(classA, classB);
    }

    // Optionally apply arc-cos transformation to the final w's
    std::vector<std::vector<float>> separationVectors;
    for (LDAClassifier &classifier : classifiers) {
#ifdef USE_TRIG
        // Copy the classifier weights
        std::vector<float> normalizedVector = classifier.w;

        float maxVal = *std::max_element(normalizedVector.begin(), normalizedVector.end());
        float minVal = *std::min_element(normalizedVector.begin(), normalizedVector.end());

        // Avoid div-by-zero if all values are the same
        float range = maxVal - minVal;
        if (range == 0.0f) {
            range = 1.0f;
        }

        constexpr float radToDeg = 180.0f / M_PI;
        for (auto& val : normalizedVector) {
            float norm = (val - minVal) / range;
            // arccos(1) -> 0°, arccos(0) -> 90°
            val = std::acos(norm) * radToDeg;
        }
        separationVectors.push_back(std::move(normalizedVector));
#else
        separationVectors.push_back(classifier.w);
#endif
    }

    // Evaluate accuracy
    testMultiClassAccuracy(classifiers, inputData);

    return separationVectors;
}
