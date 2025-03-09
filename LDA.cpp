#include <vector>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

using namespace std;

// --------------------------------------
// 1. Some utility functions
// --------------------------------------

// Matrix inverse using naive Gauss-Jordan elimination
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
        if (fabs(pivot) < 1e-12) {
            cerr << "Matrix is singular or near-singular.\n";
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
    vector<vector<float>> inv(n, vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i][j] = augmented[i][j + n];
        }
    }
    return inv;
}

// Simple matrix-vector multiply
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

// --------------------------------------
// 2. Core two-class LDA function
// --------------------------------------
/**
 * Computes a single LDA vector w that separates two classes:
 *   classA vs. classB
 * Returns a vector of dimension [numFeatures].
 */
vector<float> computeBinaryLDA(const vector<vector<float>>& classA,
                                const vector<vector<float>>& classB)
{

    int numFeatures = classA[0].size();

    // 1. Compute mean vectors of class A and class B
    vector<float> meanA(numFeatures, 0.0f);
    for (auto& row : classA) {
        for (int j = 0; j < numFeatures; j++) {
            meanA[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanA[j] /= (float)classA.size();
    }

    vector<float> meanB(numFeatures, 0.0f);
    for (auto& row : classB) {
        for (int j = 0; j < numFeatures; j++) {
            meanB[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanB[j] /= (float)classB.size();
    }

    // 2. Compute within-class scatter Sw = S_A + S_B
    vector<vector<float>> Sw(numFeatures, vector<float>(numFeatures, 0.0f));
    auto addScatter = [&](const vector<vector<float>>& data, const vector<float>& mean) {
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

    // 3. Compute w = Sw^-1 * (meanB - meanA)
    vector<vector<float>> SwInv = inverse(Sw);

    // meanDiff = (meanB - meanA)
    vector<float> meanDiff(numFeatures, 0.0f);
    for (int j = 0; j < numFeatures; j++) {
        meanDiff[j] = meanB[j] - meanA[j];
    }

    // Multiply: w = SwInv * meanDiff
    vector<float> w = matrixVectorMultiply(SwInv, meanDiff);

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

    return w;
}

// --------------------------------------
// 3. “One-vs-Rest” LDA for multi-class
// --------------------------------------
/**
 * Given inputData where inputData[i] = all samples of class i,
 * run a 2-class LDA for each class i vs. all other classes combined.
 *
 * Returns a vector of LDA vectors, one per class.
 */

vector<vector<float>> linearDiscriminantAnalysis(const vector<vector<vector<float>>>& inputData) {
    int numClasses = inputData.size();
    vector<vector<float>> ldaVectors(numClasses); // one discriminant vector per class

    // For each class i, gather its samples in classA
    // and gather all other samples in classB
    for (int i = 0; i < numClasses; i++) {
        // classA = the samples of class i
        const auto& classA = inputData[i];

        // classB = union of samples of all other classes
        vector<vector<float>> classB;
        // put in all our classes that are not classA.
        for (int j = 0; j < numClasses; j++) {
            if (j == i) continue; // skip class i
            classB.insert(classB.end(), inputData[j].begin(), inputData[j].end());
        }

        // Compute the 2-class LDA vector for i vs rest
        vector<float> w = computeBinaryLDA(classA, classB);

        // store our vector which best separates the class.
        ldaVectors[i] = w;
    }
    
    // Normalize each resulting vector between 0 and 1, then apply arccos to each value,
    // and convert the result from radians to degrees.
    for (auto &vec : ldaVectors) {
        // Find min and max of the vector.
        float minVal = *std::min_element(vec.begin(), vec.end());
        float maxVal = *std::max_element(vec.begin(), vec.end());
        
        // Prevent division by zero if all values are identical.
        float range = (maxVal - minVal);
        if (range < 1e-12) {
            range = 1;  // Alternatively, you could set all normalized values to 0.5.
        }
        
        // Normalize each element to [0, 1], apply arccos, and then convert to degrees.
        for (float &val : vec) {
            float normalized = (val - minVal) / range;
            // Apply arccos (result is in radians) and convert to degrees.
            val = std::acos(normalized) * (180.0f / M_PI);
        }
    }

    return ldaVectors;
}

