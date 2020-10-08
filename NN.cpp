#include <iostream>
#include <math.h>
#include "NN.hpp"


NN::NN(std::vector<uint> topology) {
    int i;

    input_size = topology[0];
    output_size = topology.back();
    n_layers = topology.size();
    for (i = 1; i < n_layers; i++) {

        weights.push_back(new Matrix(topology[i], topology[i-1]));
        weights.back()->setRandom();
        bias.push_back(new ColVector(topology[i]));
        bias.back()->setRandom();
        z.push_back(new ColVector(topology[i]));
        a.push_back(new ColVector(topology[i]));
        deltas.push_back(new ColVector(topology[i]));
    }
}

NN::~NN() {
    int i;
    for (i = 0; i < n_layers-1; i++) {
        delete weights[i];
        delete bias[i];
        delete z[i];
        delete a[i];
        delete deltas[i];    
    }
}

void NN::feedfoward(ColVector input) { 
    ColVector cur = input;
    Matrix w;
    ColVector b;
    int i;

    for (i = 0; i < n_layers - 1; i++) {
        w = *weights[i];
        b = *bias[i];

        *z[i] = w*cur + b;
        // Apply activation function to each elemento of the vector
        cur = z[i]->unaryExpr(&activation);
        *a[i] = cur;
    }
}

float NN::cost(ColVector* x, ColVector* y) {
    float* c = new float;
    ColVector dif;

    feedfoward(*x);
    dif = *(a.back()) - *y;
    *c += std::pow( dif.norm(), 2);

    costs_over_time.push_back( (*c)/2 );
    return (*c)/2;
}

void NN::calc_deltas(ColVector output) {
    int i;
    *(deltas.back()) = (*(a.back()) - output).array() * z.back()->unaryExpr(&activation_derivate).array();

    for (i = deltas.size() - 2; i >= 0; i--) {
        *(deltas[i]) = (weights[i+1]->transpose() * *(deltas[i+1])).array() * z[i]->unaryExpr(&activation_derivate).array();
    }
}

void NN::update_weitghs(float lr, ColVector x) {
    int i;
    ColVector curr_a;
    curr_a = x;

    for (i = 0; i < n_layers - 1; i++){
        *(weights[i]) = *(weights[i]) - lr * ( *(deltas[i]) * curr_a.transpose());
        *(bias[i]) = *(bias[i]) - lr * *(deltas[i]);
        curr_a = *(a[i]);
    }
}

void NN::backpropagate(ColVector* x, ColVector* y, float lr) {
    calc_deltas(*y);
    update_weitghs(lr, *x);
}

void NN::train(std::vector<ColVector*> x, std::vector<ColVector*> y, float lr) {
    int i;
    int n = x.size();

    for (i = 0; i < n; i++) {
        feedfoward(*(x[i]));
        backpropagate(x[i], y[i], lr);
    }
}

ColVector* NN::predict(ColVector* x) {
    feedfoward(*x);
    return a.back();
}

float NN::accuracy(std::vector<ColVector*> X, std::vector<ColVector*> y) {
    float acc = 0;
    int i;
    Index max_pred;
    Index max_real;
    ColVector* y_real;
    ColVector* y_pred;
    ColVector* x;

    for (i = 0; i < X.size(); i++) {
        x = X[i];
        y_real = y[i];
        y_pred = predict(y_real);

        y_real->maxCoeff(&max_real);
        y_pred->maxCoeff(&max_pred);

        if (max_real == max_pred)
            acc++;
    }
    
    return acc/X.size();
}