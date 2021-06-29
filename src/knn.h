#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);
    void fit(Matrix X, Vector Y);
    Vector predict(Matrix X);
    
private:
    int kVecinos;
    Matrix D;
    Vector values;
};
