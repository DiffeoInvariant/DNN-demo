#include "include/Layer.hpp"
#include <Eigen/Core>
#include <utility>
#include <iostream>

int main(){
	using Mat = Eigen::MatrixXd;
	using Vec = Eigen::VectorXd;
	
	Vec input(4);
	input[0] = -0.1;
	input[1] = 0.1;
	input[2] = 0.86;
	input[3] = -0.31;


	//layer taking 4 inputs, giving 4 outputs
	NN::Layer<NN::UpdateRule::NesterovAccGrad, double, double> testLayer(std::make_pair(4, 1), 3, "relu");

	testLayer.setInputs(input);
	//set learning rate, momentum
	testLayer.setUpdateParams(1.0e-2, 0.1);

	testLayer.forwardPass();

	auto outputs = testLayer.getOutputs();

	std::cout << "Inputs: \n" << input << '\n' << "Outputs: \n" << outputs <<'\n';

	auto grad = testLayer.computeJacobian();

	std::cout << "Jacobian: \n" << grad << '\n';
	return 0;
}

