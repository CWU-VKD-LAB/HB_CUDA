//
// Created by asnyd on 3/21/2025.
//

#ifndef INTERVAL_H
#define INTERVAL_H


// Interval struct to make interval thing more understandable
struct Interval{
    int size;
    int start;
    int end;
    int attribute;

    Interval(int s, int st, int e, int a) : size(s), start(st), end(e), attribute(a) {}
};



#endif //INTERVAL_H
