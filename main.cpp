#include <iostream>
#include "NN.hpp"


float activation(float x) {
    return tanhf(x);
}

float activation_derivate(float x) {
    return 1 - tanhf(x) * tanhf(x); 
}

int main() 
{
	NN n({2, 3, 4});
	ColVector* input = new ColVector(2);
	ColVector* output = new ColVector(4);
	
	(*input)[0] = 1.0;
	(*input)[1] = 2.;

	(*output)[0] = 1;
	(*output)[1] = 0;
	(*output)[2] = 2;
	(*output)[3] = 1;
	
	std::cout << "Treinando..\n"; 
	for (int i = 0; i < 10; i++) {
		std::cout << "Epoch:" << i <<"\n";
		n.train({input}, {output}, 2.0);

		std::cout << "Custo:\n";
		std::cout << n.cost(input, output)<< std::endl;

		std::cout << std::endl;
	}
}
