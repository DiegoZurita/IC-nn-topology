#ifndef NN_H
#define NN_H

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef Matrix::Index Index;
typedef unsigned int uint;

float activation(float x);
float activation_derivate(float x);

class NN {
public:
    NN(std::vector<uint> topology);
    ~NN();
    void feedfoward(ColVector input);
    void calc_deltas(ColVector ouput);
    void update_weitghs(float lr, ColVector x);
    void backpropagate(ColVector* x, ColVector* y, float lr);
    float cost(ColVector* x, ColVector* y);
    float accuracy(std::vector<ColVector*> X, std::vector<ColVector*> y);
    void train(std::vector<ColVector*> x, std::vector<ColVector*> y, float lr);
    ColVector* predict(ColVector* x);

    uint input_size;
    uint output_size;
    uint n_layers; 
    std::vector<float> costs_over_time;
    std::vector<Matrix*> weights;
    std::vector<ColVector*> bias;
    std::vector<ColVector*> z;
    std::vector<ColVector*> a; // z after apply the activation function
    std::vector<ColVector*> deltas;
};

#endif // NN_H