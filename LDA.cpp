#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <numeric>

// ASSUMPTIONS COMPARED TO JTABVIS VERSION:
//      - data is already set up and normalized.
//      - we have the class label stripped off already.
//      - Data is already grouped by class in input data, the 3D array.

using namespace std;

vector<vector<float>> allData; // our flattened data all in one big list
vector<int> classBreaks; // since we don't have labels, this is how we keep track of which points are for which class.
vector<vector<float>> eigenVectors;
vector<float> eigenValues;
float threshold = 1e-10;
int maxIterations = 100;

// prepare our data
void prepareData(const vector<vector<vector<float>>>& inputData, int numAttributes){

    int cols = numAttributes;
    int rows = 0;

    // Flatten inputData into allData
    for (auto& classGroup : inputData) {
        rows += classGroup.size();
        allData.insert(allData.end(), classGroup.begin(), classGroup.end());
        classBreaks.push_back(classGroup.size());
    }

    vector<float> means(cols, 0.0);
    // Calculate means using accumulate for each column
    for (int j = 0; j < cols; j++) {
        float sum = accumulate(allData.begin(), allData.end(), 0.0f,
            [j](float acc, const vector<float>& row) {
                return acc + row[j];
            });
        means[j] = sum / float(rows);
    }

    // center the data
    for(int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            allData[i][j] -= means[j];
        }
    }
}

// lda helpers.
vector<vector<float>> inverse(const vector<vector<float>>& matrix) {
    int n = matrix.size();
    vector<vector<float>> augmented(n, vector<float>(2 * n, 0.0));

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0;  // Identity matrix
    }

    // Forward elimination
    for (int i = 0; i < n; i++) {
        float pivot = augmented[i][i];
        if (pivot == 0) {
            cerr << "Matrix is singular and cannot be inverted.\n";
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    vector<vector<float>> inverse(n, vector<float>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = augmented[i][j + n];
        }
    }

    return inverse;
}

vector<vector<float>> matrixMultiply(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    int m = a.size();
    int n = b[0].size();
    int p = a[0].size();

    vector<vector<float>> result(m, vector<float>(n, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

// returns our SUPERVECTOR!!!
// the vector corresponding to the sum of each attribute in our list of eigenvectors.
// has very little real mathematical meaning, but allows us to have a way of sorting attributes which would suggest who is doing the most work in our dataset.
// the eigenvectors themselves point to the directions which give us greatest class separation, therefore, adding up all the attributes across vectors can kind of tell us
// which guys are actually working around here.
vector<float> computeLDA(const vector<vector<vector<float>>>& inputData) {

    int numClasses = inputData.size();
    int numFeatures = allData[0].size();
    int numSamples = allData.size();

    // Initialize within-class scatter matrix Sw and between-class scatter matrix Sb
    vector<vector<float>> Sw(numFeatures, vector<float>(numFeatures, 0.0f));

    // Compute class means and within-class scatter matrix Sw
    unordered_map<int, vector<float>> classMeans;
    for (int classId = 0; classId < numClasses; classId++) {
        const auto& classPoints = inputData[classId];

        vector<float> classMean(numFeatures, 0.0f);

        // Compute class mean
        for (const auto& point : classPoints) {
            for (int j = 0; j < numFeatures; j++) {
                classMean[j] += point[j];
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            classMean[j] /= classPoints.size();
        }
        classMeans[classId] = classMean;

        // Compute within-class scatter matrix Sw
        for (const auto& point : classPoints) {
            for (int j = 0; j < numFeatures; j++) {
                for (int k = 0; k < numFeatures; k++) {
                    Sw[j][k] += (point[j] - classMean[j]) * (point[k] - classMean[k]);
                }
            }
        }
    }

    // Compute global mean
    vector<float> globalMean(numFeatures, 0.0f);
    for (const auto& row : allData) {
        for (int j = 0; j < numFeatures; j++) {
            globalMean[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        globalMean[j] /= numSamples;
    }

    // Compute between-class scatter matrix Sb
    vector<vector<float>> Sb(numFeatures, vector<float>(numFeatures, 0.0f));
    for (int classId = 0; classId < numClasses; classId++){

        // grab each class one by one
        const auto& classPoints = inputData[classId];
        const auto& classMean = classMeans[classId];

        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < numFeatures; k++) {
                Sb[j][k] += classPoints.size() * (classMean[j] - globalMean[j]) * (classMean[k] - globalMean[k]);
            }
        }
    }

    // Solve generalized eigenvalue problem: Sb * v = λ * Sw * v
    // We use Sw^-1 * Sb to approximate the solution (inverse computation not included here)

    // Placeholder for solving the eigenvalue problem
    vector<vector<float>> SwInvSb = matrixMultiply(inverse(Sw),Sb);

    // Eigen decomposition using power iteration (simplified)
    int maxComponents = min(numClasses - 1, numFeatures);
    vector<vector<float>> eigenvectors(numFeatures, vector<float>(maxComponents, 0.0f));
    vector<float> eigenvalues(maxComponents, 0.0f);

    for (int i = 0; i < maxComponents; i++) {
        vector<float> v(numFeatures, 0.0f);
        v[i] = 1.0f;

        for (int iter = 0; iter < maxIterations; iter++) { // Assume maxIterations = 100
            vector<float> newVector(numFeatures, 0.0f); // = MATRIXX VECTOR MULTIPLY HERE <<<---------------------                 double[] newVector = matrixVectorMultiply(SwInvSb, vector);

            for (int j = 0; j < numFeatures; j++) {
                for (int k = 0; k < numFeatures; k++) {

                    // v = vector in java code.
                    // make sure that we are doing that normalizing step properly.
                    // make sure that we are checking convergence the right way.
                    newVector[j] += SwInvSb[j][k] * v[k];
                }
            }

            // Normalize
            float norm = sqrt(inner_product(newVector.begin(), newVector.end(), newVector.begin(), 0.0f));
            if (norm > 0) {
                transform(newVector.begin(), newVector.end(), newVector.begin(), [norm](float val) { return val / norm; });
            }

            if (equal(v.begin(), v.end(), newVector.begin(), [](float a, float b) { return fabs(a - b) < threshold; })) {
                break;
            }
            v = newVector;
        }

        // Store eigenvector
        for (int j = 0; j < numFeatures; j++) {
            eigenvectors[j][i] = v[j];
        }

        // Compute eigenvalue
        // make sure this is equivalent to the matrixvector multiply.
        // make sure we also get the dot product correct at the bottom.
        vector<float> Av(numFeatures, 0.0f);
        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < numFeatures; k++) {
                Av[j] += SwInvSb[j][k] * v[k];
            }
        }
        // check this as compared to dot product.
        eigenvalues[i] = inner_product(v.begin(), v.end(), Av.begin(), 0.0f);

        // Deflate matrix
        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < numFeatures; k++) {
                SwInvSb[j][k] -= eigenvalues[i] * v[j] * v[k];
            }
        }
    }
    // Output for verification (can be removed)
    cout << "Computed " << maxComponents << " eigenvectors for LDA." << endl;

    // print all our eigenvectors.
    for (const auto& eigenvector : eigenvectors) {
        for (float i : eigenvector) {
            cout << i << "\t";
        }
        cout << endl;
    }

    cout << "FINAL COMBINED SUPER EIGENVECTOR" << endl;
    vector<float> superVector(numFeatures, 0.0f);
    for (int i = 0; i < eigenvectors.size(); i++) {
        for (int j = 0; j < maxComponents; j++) {
            // add the value squared of this particular feature in this particular eigenvector. this gives us a sense of who is doing the heavy lifting
            // to seperate the classes in each eigenvector. not perfect, but a way of determining which attributes aren't doing anything. which is what we care about.
            superVector[i] += eigenvectors[i][j] * eigenvectors[i][j];
        }
        cout << "SUPERVECTOR ATTRIBUTE " << i << " " << superVector[i] << endl;
    }
    cout << endl;
    return superVector;
}