//
// Created by asnyd on 3/20/2025.
//

#include "IntervalHyperBlock.h"
#include <vector>
#include <future>
#include <algorithm>
#include <../hyperblock/HyperBlock.h>
#include <ostream>

// Interval struct to make interval thing more understandable
struct Interval{
    int size;
    int start;
    int end;
    int attribute;

    Interval(int s, int st, int e, int a) : size(s), start(st), end(e), attribute(a) {}
};

// Struct version of DataATTR xrecord
struct DataATTR {
    float value; // Value of one attribute of a point
    int classNum; // The class number of the point
    int classIndex; // The index of point within the class

    DataATTR(float val, int cls, int index) : value(val), classNum(cls), classIndex(index) {}
};

/**
     * Finds largest interval across all dimensions of a set of data.
     * @param dataByAttribute all data split by attribute
     * @param accThreshold accuracy threshold for interval
     * @param existingHB existing hyperblocks to check for overlap
     * @return largest interval
     */
std::vector<DataATTR> intervalHyper(std::vector<std::vector<DataATTR>>& dataByAttribute, float accThreshold, std::vector<HyperBlock>& existingHB){

    std::vector<std::future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -1);

    // Search each attribute
    for (int i = 0; i < dataByAttribute.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(std::launch::async, IntervalHyperBlock::longestInterval, ref(dataByAttribute[i]), accThreshold, ref(existingHB), i));
    }

    // Wait for results then find largest interval
    for(auto& future1 : intervals){
        Interval intr = future1.get();
        if(intr.size > 1 && intr.size > best.size){
            best.size = intr.size;
            best.start = intr.start;
            best.end = intr.end;
            best.attribute = intr.attribute;

            attr = intr.attribute;
        }
    }

    // Construct ArrayList of data
    std::vector<DataATTR> longest;
    if(best.size != -1){
        for(int i = best.start; i <= best.end; i++){
            longest.push_back(dataByAttribute[attr][i]);
        }
    }

    return longest;
}




/**
 * Seperates data into seperate vecs by attribute
 */
std::vector<std::vector<DataATTR>> separateByAttribute(std::vector<std::vector<std::vector<float>>>& data){
    std::vector<std::vector<DataATTR>> attributes;

    // Go through the attribute columns
    for(int k = 0; k < FIELD_LENGTH; k++){
        std::vector<DataATTR> tmpField;

        // Go through the classes
        for(int i = 0; i < data.size(); i++){
            // Go through the points
            for(int j = 0; j < data[i].size(); j++){
                tmpField.push_back(DataATTR(data[i][j][k], i, j));
            }
        }

        // Sort data by value then add
        sort(tmpField.begin(), tmpField.end(), [](const DataATTR& a, const DataATTR& b) {
            return a.value < b.value;
        });
        attributes.push_back(tmpField);
    }

    return attributes;
}



/***
 * This will sort the array based on the "best" columns values
 *
 * The columns themselves aren't moving, we are moving the points
 * based on the one columns values;
 */
void sortByColumn(std::vector<std::vector<float>>& classData, int colIndex) {
    sort(classData.begin(), classData.end(), [colIndex](const std::vector<float>& a, const std::vector<float>& b) {
        return a[colIndex] < b[colIndex];
    });
}


/***
 * Finds the longest interval in a sorted list of data by attribute.
 * @param dataByAttribute sorted data by attribute
 * @param accThreshold accuracy threshold for interval
 * @param existingHB existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval longestInterval(std::vector<DataATTR>& dataByAttribute, float accThreshold, std::vector<HyperBlock>& existingHB, int attr){
    //cout << "Started longest interval \n" << endl;

    Interval intr(1, 0, 0, attr);
    Interval max_intr(-1, -1, -1, attr);

    int n = dataByAttribute.size();
    float misclassified = 0;

    for(int i = 1; i < n; i++){
        // If current class matches with next
        if(dataByAttribute[intr.start].classNum == dataByAttribute[i].classNum){
            intr.size++;
        }
        else if( (misclassified+1) / intr.size > accThreshold){
            // ^ i think this is a poor way to check. but not changing rn for the translation from java
            misclassified++;
            intr.size++;
        }
        else{
            // Remove value from interval if accuracy is below threshold.
            if(dataByAttribute[i-1].value == dataByAttribute[i].value){
                // remove then skip overlapped values
                IntervalHyperBlock::removeValueFromInterval(dataByAttribute, intr, dataByAttribute[i].value);
                i = IntervalHyperBlock::skipValueInInterval(dataByAttribute, i, dataByAttribute[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && IntervalHyperBlock::checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
                max_intr.start = intr.start;
                max_intr.end = intr.end;
                max_intr.size = intr.size;
                max_intr.attribute = attr;
            }

            // Reset curr interval
            intr.size = 1;
            intr.start = i;
            misclassified = 0;
        }
        intr.end = i;
    }

    // final check update longest interval if it doesn't overlap
    if(intr.size > max_intr.size && IntervalHyperBlock::checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    //cout << "Finished longest interval \n" << endl;

    return max_intr;
}


/*
*  Check if interval range overlaps with any existing hyperblocks
*  to not overlap the interval maximum must be below all existing hyperblock minimums
*  or the interval minimum must be above all existing hyperblock maximums
*/
bool checkIntervalOverlap(std::vector<DataATTR>& dataByAttribute, Interval& intr, int attr, std::vector<HyperBlock>& existingHB){
    // interval range of vals
    float intv_min = dataByAttribute[intr.start].value;
    float intv_max = dataByAttribute[intr.end].value;

    for(const HyperBlock& hb : existingHB){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    // If unique return true
    return true;
}



int skipValueInInterval(std::vector<DataATTR>& dataByAttribute, int i, float value){
    while(dataByAttribute[i].value == value){
        if(i < dataByAttribute.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    return i;
}


void removeValueFromInterval(std::vector<DataATTR>& dataByAttribute, Interval& intr, float value){
    while(dataByAttribute[intr.end].value == value){
        if(intr.end > intr.start){
            intr.size--;
            intr.end--;
        }
        else{
            intr.size = -1;
            break;
        }
    }
}

void generateHBs(std::vector<std::vector<std::vector<float>>>& data, std::vector<HyperBlock>& hyperBlocks, std::vector<int> &bestAttributes){
    // Hyperblocks generated with this algorithm
    std::vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    std::vector<std::vector<DataATTR>> dataByAttribute = separateByAttribute(data);
    std::vector<std::vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    std::vector<std::vector<std::vector<float>>> datum;
    std::vector<std::vector<std::vector<float>>> seed_data;
    std::vector<std::vector<int>> skips;
	// "Initialized datum, seed_data, skips\n" << endl;

    // Initially generate blocks
        while(dataByAttribute[0].size() > 0){

            std::vector<DataATTR> intv = intervalHyper(dataByAttribute, 100, gen_hb);
  			all_intv.push_back(intv);

        // if hyperblock is unique then add
        if(intv.size() > 1){
            std::vector<std::vector<std::vector<float>>> hb_data;
            std::vector<std::vector<float>> intv_data;

            // Add the points from real data that are in the intervals
            for(DataATTR& dataAttr : intv){
                intv_data.push_back(data[dataAttr.classNum][dataAttr.classIndex]);
            }

            // add data and hyperblock
            hb_data.push_back(intv_data);

            HyperBlock hb(hb_data, intv[0].classNum);

            gen_hb.push_back(hb);
        }else{
            break;
        }
    }

        // Add all hbs from gen_hb to hyperBlocks
        hyperBlocks.insert(hyperBlocks.end(), gen_hb.begin(), gen_hb.end());

    // All data: go through each class and add points from data
    for(const std::vector<std::vector<float>>& classData : data){
        datum.push_back(classData);
        seed_data.push_back(std::vector<std::vector<float>>());
        skips.push_back(std::vector<int>());
    }

    // find which data to skip
    for(const std::vector<DataATTR>& dataAttrs : all_intv){
        for(const DataATTR& dataAttr : dataAttrs){
            skips[dataAttr.classNum].push_back(dataAttr.classIndex);
        }
    }
    // Sort the skips
    for(std::vector<int>& skip : skips){
        sort(skip.begin(), skip.end());
    }

    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(skips[i].size() > 0){
                if(j != skips[i][0]){
                    seed_data[i].push_back(data[i][j]);
                }
                else{
                    // remove first element from skips[i]
                    skips[i].erase(skips[i].begin());
                }
            }
            else{
                seed_data[i].push_back(data[i][j]);
            }
        }
    }

    // Sort data by most important attribute
    for(int i = 0; i < datum.size(); i++){
        sortByColumn(datum[i], bestAttributes[i]);
        sortByColumn(seed_data[i], bestAttributes[i]);
    }

    try{
        merger_cuda(seed_data, datum, hyperBlocks);
    }catch (std::exception e){
        //std::cout << "Error in generateHBs: merger_cuda" << std::endl;
    }
}
