#include <iostream>
#include "NN.hpp"
#include <math.h>


float activation(float x) {
    return tanhf(x);
}

float activation_derivate(float x) {
    return 1 - tanhf(x) * tanhf(x); 
}

float fx(float x) {
    return std::pow(x, 2);
}

int main() 
{

    int samples = 50;
    float a = -1;
    float b = 1;
    float h = (b-a)/(samples-1);
    int i;

    std::vector<ColVector*> X;
    std::vector<ColVector*> y;
    ColVector* red;
    ColVector* blue;
    ColVector* y_red;
    ColVector* y_blue;

    NN n({2, 2});


    // Creating samples
    for (i = 0; i < samples; i++) {
        red = new ColVector(2);
        y_red = new ColVector(2);
        (*red)[0] = a + i*h;
        (*red)[1] = fx((*red)[0]) - 0.5;
        (*y_red)[0] = 1;
        (*y_red)[1] = 0;

        blue = new ColVector(2);
        y_blue = new ColVector(2);
        (*blue)[0] = a + i*h;
        (*blue)[1] = fx((*blue)[0]) + 0.5;
        (*y_blue)[0] = 0;
        (*y_blue)[1] = 1;

        X.push_back(red);
        X.push_back(blue);

        y.push_back(y_red);
        y.push_back(y_blue);
    }



    // training
    n.train(X, y, 0.4);
    
    std::cout << "Acuracia: " << n.accuracy(X, y) << std::endl;


    // Generating output
    int pixels = 100;
    h = (b - a) / (pixels - 1);
    float h_i = 4.0/ (pixels - 1);
    ColVector* x = new ColVector(2);
    ColVector* pred;
    Index max_i;
    for (i = 0; i < pixels; i++) {
        for (int j = 0; j < pixels; j++) {
            (*x)[0] = a + j*h;
            (*x)[1] = -2 + i*h_i;

            pred = n.predict(x);
            pred->maxCoeff(&max_i);

            std::cout << max_i << " ";
        }
        std::cout << std::endl;
    }
}