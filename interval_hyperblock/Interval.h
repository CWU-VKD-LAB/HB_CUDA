//
// Created by Austin Snyder on 3/21/2025.
//

#ifndef INTERVAL_H
#define INTERVAL_H


// Interval struct to make interval thing more understandable
struct Interval{
    int size;
    int start; // the index which we are starting our interval from
    int end;   // end index, obviously
    int attribute;
    int dominantClass; // the class which is the most of the interval. should be like 95% usually at least. depends on accuracy.

    Interval(int s, int st, int e, int a, int d) : size(s), start(st), end(e), attribute(a), dominantClass(d) {}
};


#endif //INTERVAL_H
