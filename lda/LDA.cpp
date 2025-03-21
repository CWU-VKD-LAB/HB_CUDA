#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>

// Matrix inverse using naive Gauss-Jordan elimination
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
        if (fabs(pivot) < 1e-12) {
            std::cerr << "Matrix is singular or near-singular.\n";
            exit(EXIT_FAILURE);
        }
        // Normalize pivot row
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        // Eliminate down
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

// Simple matrix-std::vector multiply
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

// Helper: dot product of two std::vectors
inline float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// --------------------------------------
// 2. Core two-class LDA function (with bias computation)
// --------------------------------------

// Define a struct to hold both the weight std::vector and its bias.
struct LDAClassifier {
    std::vector<float> w;
    float bias;
};

/**
 * Computes a single LDA std::vector w (and bias) that separates two classes:
 *   classA vs. classB
 * Returns an LDAClassifier containing the weight std::vector and bias.
 */
LDAClassifier computeBinaryLDA(const std::vector<std::vector<float>>& classA,
                               const std::vector<std::vector<float>>& classB)
{
    int numFeatures = classA[0].size();

    // 1. Compute mean std::vectors of class A and class B
    std::vector<float> meanA(numFeatures, 0.0f);
    for (auto& row : classA) {
        for (int j = 0; j < numFeatures; j++) {
            meanA[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanA[j] /= (float)classA.size();
    }

    std::vector<float> meanB(numFeatures, 0.0f);
    for (auto& row : classB) {
        for (int j = 0; j < numFeatures; j++) {
            meanB[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanB[j] /= (float)classB.size();
    }

    // 2. Compute within-class scatter Sw = S_A + S_B
    std::vector<std::vector<float>> Sw(numFeatures, std::vector<float>(numFeatures, 0.0f));
    auto addScatter = [&](const std::vector<std::vector<float>>& data, const std::vector<float>& mean) {
        for (auto& x : data) {
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    Sw[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
                }
            }
        }
    };

    addScatter(classA, meanA);
    addScatter(classB, meanB);

    // REGULARIZE TO AVOID THE MATRIX BEING SINGULAR.
    float lambda = 1e-3f; // adjust as needed.
    for (int i = 0; i < numFeatures; i++) {
        Sw[i][i] += lambda;
    }

    // 3. Compute w = Sw^-1 * (meanB - meanA)
    std::vector<std::vector<float>> SwInv = inverse(Sw);
    std::vector<float> meanDiff(numFeatures, 0.0f);
    for (int j = 0; j < numFeatures; j++) {
        meanDiff[j] = meanB[j] - meanA[j];
    }
    std::vector<float> w = matrixVectorMultiply(SwInv, meanDiff);

    // 4. Normalize w
    float normVal = 0.0f;
    for (float val : w) {
        normVal += val * val;
    }
    normVal = sqrt(normVal);
    if (normVal > 1e-12) {
        for (float &val : w) {
            val /= normVal;
        }
    }

    // 5. Compute bias: use the midpoint between the projections of the class means
    float projMeanA = dotProduct(w, meanA);
    float projMeanB = dotProduct(w, meanB);
    float bias = (projMeanA + projMeanB) / 2.0f;

    // Flip w and bias if the target class's projection is lower than the other
    if (projMeanA < projMeanB) {
        for (float &val : w) {
            val = -val;
        }
        bias = -bias;
    }

    LDAClassifier classifier;
    classifier.w = w;
    classifier.bias = bias;
    return classifier;
}

// predictClass: for a given sample, compute the adjusted score for each classifier
// (w^T x - bias) and return the class with the highest score.
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
        for (const auto &sample : inputData[i]) {
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


// --------------------------------------
// “One-vs-Rest” LDA for multi-class
// --------------------------------------
/**
 * Given inputData where inputData[i] = all samples of class i,
 * run a 2-class LDA for each class i vs. all other classes combined.
 *
 * Returns a std::vector of LDAClassifier, one per class.
 */
std::vector<std::vector<float>> linearDiscriminantAnalysis(const std::vector<std::vector<std::vector<float>>>& inputData) {
    int numClasses = inputData.size();
    std::vector<LDAClassifier> classifiers(numClasses); // one classifier per class

    // For each class i, gather its samples in classA
    // and gather all other samples in classB
    for (int i = 0; i < numClasses; i++) {
        // classA = the samples of class i
        const auto& classA = inputData[i];
        // classB = union of samples of all other classes
        std::vector<std::vector<float>> classB;
        for (int j = 0; j < numClasses; j++) {
            if (j == i) continue; // skip class i
            classB.insert(classB.end(), inputData[j].begin(), inputData[j].end());
        }
        // Compute the 2-class LDA classifier for i vs. rest
        classifiers[i] = computeBinaryLDA(classA, classB);
    }
    std::vector<std::vector<float>> separationVectors;
    for (LDAClassifier &classifier : classifiers) {
        separationVectors.push_back(classifier.w);
    }

    testMultiClassAccuracy(classifiers, inputData);
    return separationVectors;
}
