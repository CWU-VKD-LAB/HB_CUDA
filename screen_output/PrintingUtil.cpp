//
// Created by Austin Snyder on 3/20/2025.
//

#include "PrintingUtil.h"
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
    cout << "=== Hyperblock (HB) Classification System ===\n\n";
    cout << "1. Import training data.\n";
    cout << "2. Import testing data.\n";
    cout << "3. Save normalized training data.\n";
    cout << endl;
    cout << "4. Import regular HBs.\n";
    cout << "5. Export regular HBs.\n";
    cout << "6. Generate HBs.\n";
    cout << "7. Simplify HBs.\n";
    cout << "8. Test HBs.\n";
    cout << "9. Test 1-1 HBs.\n";
    cout << "10. K-Fold Cross Validation.\n";
    cout << endl;
    cout << "11. Generate 1-1 HBs.\n";
    cout << "12. Import 1-1 HBs.\n";
    cout << "13. Export 1-1 HBs.\n";
    cout << "14. K-Fold on 1-1 HBs\n";
    cout << "15. Generate One To Rest Blocks and Save.\n";
    cout << endl;

    cout << "16. Find Best Parameters (Grid Search).\n";
    cout << "17. Generate Next Level HBs.\n";
    cout << "18. K-Fold validation with Level N HBs.\n\n" << endl;
    cout << "19. Generate and Test Precision Weighted HBs. (Experimental)";
    cout << "20. Quit\n\n";
}


float PrintingUtil::printConfusionMatrix(vector<vector<long>>& data, const int NUM_CLASSES, map<int, string>& CLASS_MAP_INT) {
    vector<string> classLabels(NUM_CLASSES);

    vector<float> accuracies(NUM_CLASSES, 0.0);

    // Calculate the accuracies of each of the rows.
    // Only the diagonal values are correct predictions
    long overallCorrect = 0;
    long overallIncorrect = 0;
    long overallTotalClassifications = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        long correct = 0;
        long incorrect = 0;
        long totalClassifications = 0;

        for (int j = 0; j < NUM_CLASSES; ++j) {
            totalClassifications += data[i][j];
            if (i == j) {
                correct += data[i][j];  // Diagonal value indicates correct predictions
            } else {
                incorrect += data[i][j];  // Off-diagonal values are incorrect predictions
            }
        }

        if (totalClassifications > 0) {
            accuracies[i] = (float)correct / totalClassifications;
        }

        overallCorrect += correct;
        overallIncorrect += incorrect;
        overallTotalClassifications += totalClassifications;
    }

    // Overall Accuracy, prevent divide by 0 with the ternary
    float overallAccuracy = (overallTotalClassifications != 0) ? ((float)overallCorrect / overallTotalClassifications) : 0;

    // Calculate column width based on the longest class name and largest number
    size_t maxWidth = 8; // Minimum width

    for (const auto& name : classLabels) {
        maxWidth = max(maxWidth, name.length() + 2);
    }

    for (const auto& row : data) {
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

    // Print each row with row label
    for (size_t i = 0; i < data.size(); i++) {
        cout << setw(maxWidth) << CLASS_MAP_INT[i] << " |";

        for (size_t j = 0; j < data[i].size(); j++) {
            cout << setw(maxWidth) << data[i][j] << " |";
        }

        cout << accuracies[i] << endl;
    }

    cout << "The overall accuracy is " << overallAccuracy << endl;
    return overallAccuracy;
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