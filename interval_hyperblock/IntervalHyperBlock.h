//
// Created by asnyd on 3/20/2025.
//

#ifndef INTERVALHYPERBLOCK_H
#define INTERVALHYPERBLOCK_H

class IntervalHyperBlock {
  public:
    static Interval longestInterval(std::vector<DataATTR>& dataByAttribute, float accThreshold, std::vector<HyperBlock>& existingHB, int attr);
    static void removeValueFromInterval(std::vector<DataATTR>& dataByAttribute, Interval& intr, float value);
    static int skipValueInInterval(std::vector<DataATTR>& dataByAttribute, int i, float value);
    static bool checkIntervalOverlap(std::vector<DataATTR>& dataByAttribute, Interval& intr, int attr, std::vector<HyperBlock>& existingHB);
    static std::vector<std::vector<DataATTR>> separateByAttribute(std::vector<std::vector<std::vector<float>>>& data);
    static void sortByColumn(std::vector<std::vector<float>>& classData, int colIndex);
    static void generateHBs(std::vector<std::vector<std::vector<float>>>& data, std::vector<HyperBlock>& hyperBlocks, std::vector<int> &bestAttributes);

};



#endif //INTERVALHYPERBLOCK_H
