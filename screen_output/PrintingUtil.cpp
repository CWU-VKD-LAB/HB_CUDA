//
// Created by asnyd on 3/20/2025.
//
#include <iomanip>
#include "PrintingUtil.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <numeric>

using namespace std;

// Function to clear the console screen (cross-platform)
void PrintingUtil::clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

/**
 * Function to wait for user input before continuing.
 * Used in the command line interface.
 */
void PrintingUtil::waitForEnter() {
    cout << "\nPress Enter to continue...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

/**
 * This is a function to display the command line interface menu.
 *
 * Update me if you add new options to the switch case in Host.cu
 */
void PrintingUtil::displayMainMenu() {
    clearScreen();
    cout << "=== HyperBlock Classification System ===\n\n";
    cout << "1. Import training data.\n";
    cout << "2. Import testing data.\n";
    cout << "3. Save normalized training data.\n";
    cout << endl;
    cout << "4. Import regular Hyperblocks.\n";
    cout << "5. Export regular Hyperblocks.\n";
    cout << "6. Generate Hyperblocks.\n";
    cout << "7. Simplify Hyperblocks.\n";
    cout << "8. Test Hyperblocks.\n";
    cout << "9. Test 1-1 HyperBlocks.\n";
    cout << "10. K Fold Cross Validation.\n";
    cout << endl;
    cout << "11. Generate 1-1 Hyperblocks.\n";
    cout << "12. Import 1-1 Hyperblocks.\n";
    cout << "13. Export 1-1 Hyperblocks.\n";
    cout << "14. K-Fold 1-1\n";
    cout << "15. Generate One To Rest Blocks and Save.\n";
    cout << endl;

    cout << "16. Find Best Parameters (Grid Search).\n";
    cout << "17. Generate Next Level HBs.\n";
    cout << "18. K fold validation with Level N HBs.\n\n" << endl;
    cout << "19. Quit\n\n";


}

/**
 * This struct holds the performance metrics
 * for a set of hyperblocks.
 * 
 * Typically, you would have the overall set of 
 * metrics be stored in the floats. Then each individual
 * class should be stored in the vectors. 
 * 
 * The entry [i] in the vector should be for class i. 
 * 
 */
struct PerformanceMetrics{
    float accuracy;
    vector<float> byClassAccuracy;
    
    // Precision = TP / (TP + FP) 
    float precision;
    vector<float> byClassPrecision;

    // Recall = TP / (TP + FN)
    float recall;
    vector<float> byClassRecall;

    // (2 * Precision  
    float f1;
    vector<float> byClassF1;

    // Constructor
    PerformanceMetrics(float a, vector<float> cA, float p, vector<float> cP, float r, vector<float> cR, float f, vector<float> cF) : 
    accuracy(a), byClassAccuracy(cA), precision(p), byClassPrecision(cP), recall(r), byClassRecall(cR), f1(f), byClassF1(cF) {}
};


/**
 * Takes in the confusion matrix for the classifier,
 * and computes performance metrics of the model.
 * 
 * This is returned in a struct!
 */
PerformanceMetrics computePerformanceMetrics(vector<vector<long>>& confusionMatrix) {
    int numClasses = confusionMatrix.size();
    vector<float> byClassAccuracy(numClasses, 0.0f);
    vector<float> byClassPrecision(numClasses, 0.0f);
    vector<float> byClassRecall(numClasses, 0.0f);
    vector<float> byClassF1(numClasses, 0.0f);

    long total = 0;
    long correct = 0;

    // Compute precision, recall, F1, accuracy for each class
    for (int i = 0; i < numClasses; ++i) {
        long tp = confusionMatrix[i][i];
        
        // sum of actual class i (used for recall)
        long rowSum = 0;

        // sum of predicted class i (used for precision)
        long colSum = 0;

        for (int j = 0; j < numClasses; ++j) {
            rowSum += confusionMatrix[i][j]; // all predicted labels for actual i
            colSum += confusionMatrix[j][i]; // all actual labels we predicted as i
            total += confusionMatrix[i][j];

            if (i == j) correct += confusionMatrix[i][j]; // Diagonal on the matrix
        }

        // Avoid divide-by-zero
        float precision = (colSum == 0) ? 0.0f : static_cast<float>(tp) / colSum;
        float recall    = (rowSum == 0) ? 0.0f : static_cast<float>(tp) / rowSum;
        float f1        = (precision + recall == 0) ? 0.0f : 2 * precision * recall / (precision + recall);
        float acc       = (rowSum == 0) ? 0.0f : static_cast<float>(tp) / rowSum;

        byClassPrecision[i] = precision;
        byClassRecall[i] = recall;
        byClassF1[i] = f1;
        byClassAccuracy[i] = acc;
    }

    float accuracy = (total == 0) ? 0.0f : static_cast<float>(correct) / total;

    float avgPrecision = accumulate(byClassPrecision.begin(), byClassPrecision.end(), 0.0f) / numClasses;
    float avgRecall    = accumulate(byClassRecall.begin(), byClassRecall.end(), 0.0f) / numClasses;
    float avgF1        = accumulate(byClassF1.begin(), byClassF1.end(), 0.0f) / numClasses;

    return PerformanceMetrics(accuracy, byClassAccuracy, avgPrecision, byClassPrecision,avgRecall, byClassRecall, avgF1, byClassF1);
}

float PrintingUtil::printConfusionMatrix(vector<vector<long>>& confusionMatrix, const int NUM_CLASSES, map<int, string>& CLASS_MAP_INT) {
    PerformanceMetrics metrics = computePerformanceMetrics(confusionMatrix);

    // Calculate column width based on the longest class name and largest number
    size_t maxWidth = 8; // Minimum width

    // Find the spacing we need for the class labels.
    for (int i = 0; i < confusionMatrix.size(); i++) {
        maxWidth = max(maxWidth, CLASS_MAP_INT[i].length() + 2);
    }

    for (const auto& row : confusionMatrix) {
        for (const auto& cell : row) {
            string numStr = to_string(cell);
            maxWidth = max(maxWidth, numStr.length() + 2);
        }
    }

    // Print header row with "Actual\Predicted" in the corner
    cout << setw(maxWidth) << "Act\\Pred" << " |";
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << setw(maxWidth) << CLASS_MAP_INT[i] << " |";
    }
    cout << endl;

    // Print separator line
    cout << string(maxWidth, '-') << "-+";
    for (size_t i = 0; i < CLASS_MAP_INT.size(); i++) {
        cout << string(maxWidth, '-') << "-+";
    }
    cout << endl;

    // Go through and print the "by class" metrics.
    cout << "=== Class Accuracies ===" << endl;
    for(int i = 0; i < confusionMatrix.size(); i++) cout << "Class " << CLASS_MAP_INT[i] << ": " << metrics.byClassAccuracy[i] << endl;

    cout << "=== Class Precisions ===" << endl;
    for(int i = 0; i < confusionMatrix.size(); i++) cout << "Class " << CLASS_MAP_INT[i] << ": " << metrics.byClassPrecision[i] << endl;
    
    cout << "===== Class Recall =====" << endl;
    for(int i = 0; i < confusionMatrix.size(); i++) cout << "Class " << CLASS_MAP_INT[i] << ": " << metrics.byClassRecall[i] << endl;

    cout << "====== Class F1 ========" << endl;
    for(int i = 0; i < confusionMatrix.size(); i++) cout << "Class " << CLASS_MAP_INT[i] << ": " << metrics.byClassF1[i] << endl;
    
    // Print the overall results!
    cout << "Overall accuracy: " << metrics.accuracy << endl;
    cout << "Overall precision: " << metrics.precision << endl;
    cout << "Overall recall is: " << metrics.recall << endl;
    cout << "Overall F1 score is: " << metrics.f1 << endl;

    //TODO: Could change this to return whole struct. 
    return metrics.accuracy;
}


/**
 * This just prints out the dataset for debugging purposes.
 *
 * @param vec The dataset, [class][pointIdx][attrIdx]
 */
void PrintingUtil::printDataset(const vector<vector<vector<float>>>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << "Class " << i << ":" << endl;

        for (const auto& row : vec[i]) {
            cout << "  [";

            for (int j = 0; j < row.size(); j++) {
                cout << row[j];
                if (j < row.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        cout << endl;
    }
}