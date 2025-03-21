//
// Created by asnyd on 3/21/2025.
//

#ifndef DATAATTR_H
#define DATAATTR_H

// Struct version of DataATTR xrecord
struct DataATTR {
    float value; // Value of one attribute of a point
    int classNum; // The class number of the point
    int classIndex; // The index of point within the class

    DataATTR(float val, int cls, int index) : value(val), classNum(cls), classIndex(index) {}
};

#endif //DATAATTR_H
