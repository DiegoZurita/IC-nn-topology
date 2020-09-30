#include <iostream>
#include <math.h>
#include "NN.hpp"


NN::NN(std::vector<uint> topology) {
    int i;

    input_size = topology[0];
    output_size = topology.back();
    n_layers = topology.size();
    for (i = 1; i < n_layers; i++) {
        std::cout << "Layer: " << i << std::endl;
        std::cout << "(" << topology[i] << "," << topology[i-1] << ")\n";

        weights.push_back(new Matrix(topology[i], topology[i-1]));
        weights.back()->setRandom();
        bias.push_back(new ColVector(topology[i]));
        bias.back()->setRandom();
        z.push_back(new ColVector(topology[i]));
        a.push_back(new ColVector(topology[i]));
        deltas.push_back(new ColVector(topology[i]));
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

    return (*c)/2;
}

void NN::calc_deltas(ColVector output) {
    int i;
    std::cout << "calc deltas \n";
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

        std::cout << std::endl;
        std::cout << *(weights[i]) << std::endl;
        std::cout << std::endl;
        curr_a = *(a[i]);
    }
    std::cout << "----------------" << std::endl;
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
