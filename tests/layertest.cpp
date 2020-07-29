#include "../include/Layer.hpp"
#include <Eigen/Core>
#include <utility>
#include <iostream>

int main(){
	using Mat = Eigen::MatrixXd;
	using Vec = Eigen::VectorXd;
	
	Vec input(4);
	input[0] = 0.1;
	input[1] = -0.1;
	input[2] = -0.86;
	input[3] = -0.31;


	//layer taking 4 inputs, giving 4 outputs
	NN::Layer testLayer(std::make_pair(4, 1), 3, "tanh");

	testLayer.setInputs(input);
	//set learning rate, momentum
	testLayer.setUpdateParams(1.0, 0.1);

	testLayer.forwardPass();

	auto outputs = testLayer.getOutputs();

	std::cout << "Inputs: \n" << input << '\n' << "Prediction: \n" << outputs <<'\n';

	auto grad = testLayer.computeJacobian();

	std::cout << "Jacobian: \n" << grad << '\n';

	auto target = 2*Mat::Ones(4,3);

	std::cout << "Target:\n" << target << '\n';

	auto resid = target - outputs;

	std::cout << "resid:\n" << resid << '\n';
	//resid is also the negative of the derivative of the L2 loss function w.r.t. the output
	testLayer.backwardPass(-resid);

	std::cout << "Gradient:\n" << testLayer.getGradient() << '\n';

	std::cout << "Weights:\n" << testLayer.getWeights() << '\n';

	testLayer.updateWeights();

	std::cout << "New weights:\n" << testLayer.getWeights() << '\n';

	testLayer.forwardPass();

	outputs = testLayer.getOutputs();

	std::cout << "New prediction:\n" << outputs << '\n';


	
	return 0;
}

