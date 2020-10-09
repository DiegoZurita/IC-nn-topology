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
        (*blue)[1] = fx((*blue)[0]) + 0.8;
        (*y_blue)[0] = 0;
        (*y_blue)[1] = 1;


        X.push_back(blue);
        y.push_back(y_blue);

        X.push_back(red);
        y.push_back(y_red);
    }



    // training
    NN n({2, 2});
    n.train(X, y, 0.03872, 20);



    int pixels = 100;
    h = (b - a) / (pixels - 1);
    float h_i = 2.0/ (pixels - 1);
    int j = 0;
    ColVector* x = new ColVector(2);
    ColVector* pred;
    Index max_i;
    for (i = 0; i < pixels; i++) {
        std::cout << 1.5 - i*h_i << " ";
        for (j = 0; j < pixels; j++) {
            (*x)[0] = a + j*h;
            (*x)[1] = 1.5 - i*h_i;

            pred = n.predict(x);
            pred->maxCoeff(&max_i);

            std::cout << max_i << " ";
        }
        std::cout << std::endl;
    }
}