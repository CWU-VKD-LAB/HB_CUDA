//
// Created by asnyd on 3/20/2025.
//

#ifndef KNN_H
#define KNN_H



class Knn {
    public:
       static float euclideanDistance(const std::vector<float>& hbCenter, const std::vector<float>& point);
       static std::vector<std::vector<long>> kNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k);


};



#endif //KNN_H
