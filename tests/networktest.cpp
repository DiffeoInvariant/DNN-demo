#include "../include/Network.hpp"
#include "../include/Layer.hpp"
#include <Eigen/Core>


using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

using TestLayer = NN::Layer;
using TestNetwork = NN::Network;

int main(){
	// 3-layer network, taking a 10 x 1 input and producing one output, 
	// with a hidden layer containing 8 neurons, an output layer with 5,
	// and sigmoid activation
	std::cout << "Making layers\n";
	TestLayer l1(std::make_pair(2, 10), 8, "sigmoid");
	TestLayer l2(std::make_pair(2,8), 5, "sigmoid");
	TestLayer l3(std::make_pair(2,5), 1, "sigmoid");

	std::cout << "Making network\n";

	//construct network from initializer list of layers
	TestNetwork net("sigmoid", "L2", {l1, l2, l3});

	std::cout << "Making input\n";

	//network input
	Mat input(10,2);
	//input << 0.63, 0.12, 1.35, -0.02, 0.18, 0.43, 0.79, 0.36, 0.63, 0.82;
	input(0,0) = 0.63;
	input(1,0) = 0.12;
	input(2,0) = 1.35;
	input(3,0) = -0.02;
	input(4,0) = 0.18;
	input(5,0) = 0.43;
	input(6,0) = 0.79;
	input(7,0) = 0.36;
	input(8,0) = 0.63;
	input(9,0) = 0.82;
	input(0,1) = 1.31;
	input(1,1) = 0.08;
	input(2,1) = 0.42;
	input(3,1) = 0.91;
	input(4,1) = 0.13;
	input(5,1) = 0.01;
	input(6,1) = -2.1;
	input(7,1) = -1.2;
	input(8,1) = 0.47;
	input(9,1) = 0.52;




	std::cout << "Making target\n";

	//goal is to produce a 0.15 in each output slot
	auto targ = 0.15 * Vec::Ones(2);

	net.setInputs(input.transpose());
	net.setTarget(targ, true);

	std::cout << "Setting update params\n";

	//set learning rate, momentum
	net.setUpdateParams(1.0e-3, 0.2);

	net.setNumThreads(4);

	std::cout << "Making initial prediction \n";

	auto initialPred = net.predictVal();

	std::cout << "Inputs :\n" << input.transpose() << "\n Target:\n" << targ <<
		"\n Initial Prediction: \n" << initialPred << '\n';
	
	net.summary();

	std::cout << "Training for up to 1,000,000 iterations:\n";
	net.train(1.0e-5, 1.0e6);

	std::cout << "Target :\n" << targ << "\n Trained Prediction: \n" << net.getOutputs() << '\n';
	auto trainLoss = net.getLossHistory();

	std::cout << "Final loss: \n" << trainLoss.back();

	std::cout << "\nTrained in " << trainLoss.size() << " iterations.\n";

	net.visualizeNetwork();
	
	return 0;
}

