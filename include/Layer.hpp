#ifndef LAYER_HPP
#define LAYER_HPP
//#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_map>
#include <functional>
#include <utility>
#include <string>
#include <cmath>
#include <random>
#include <memory>
#include <omp.h>
#include <iostream>
//#include <mkl.h>

namespace NN
{
	using Mat = Eigen::MatrixXd;
	using Vec = Eigen::VectorXd;
	using int_t = int_fast64_t;


	//namespace-level variable for activations
	std::unordered_map<std::string, std::function<double(double)>> ACTIVATIONS = {
		{"linear", [](double x){ return x;}},
		{"sigmoid", [](double x){ return 1.0/(1.0 + exp(-x));}},
		{"tanh", [](double x){ return tanh(x);}},
		{"relu", [](double x){ return x > 0 ? x : 0.0;}},
		{"softplus", [](double x){ return log(exp(x) + 1);}}
	};
	
	/*
	 * takes pairs of (input, output)
	 * */
	std::unordered_map<std::string, std::function<Mat(std::pair<Mat, Mat>)>> ACTIVATION_DERIVATIVES = {
		{"linear", [](const std::pair<Mat, Mat>& x) -> Mat { 
									return Mat::Ones(x.second.rows(), x.second.cols());
															  } },
		{"sigmoid", [](const std::pair<Mat, Mat>& x) -> Mat {
									return x.second.cwiseProduct((Mat::Ones(x.second.rows(), x.second.cols())- x.second));
															   } },
		{"tanh",   [](const std::pair<Mat, Mat>& x) -> Mat {
									return Mat::Ones(x.second.rows(), x.second.cols()) - x.second.cwiseProduct(x.second);
													} },
		{"relu", [](const std::pair<Mat, Mat>& x) -> Mat { 
									return x.first.unaryExpr([](double y){
											return y > 0.0 ? 1.0 : 0.0; }); } },
		{"softplus", [](const std::pair<Mat, Mat>& x) -> Mat { 
									return x.first.unaryExpr([](double y){return 1/(exp(-y)+1);});
															 } }

	};

	enum class UpdateRule
	{
		NesterovAccGrad//simple momentum update
	};


	template<UpdateRule update, typename... updateArgs>
	class Layer
	{

	protected:

		std::pair<int_t, int_t> input_shape;

		int_t output_size;

		std::function<double(double)> activation;

		std::function<Mat(std::pair<Mat, Mat>)> activation_grad;

		Mat actVals;

		Mat weights;

		Mat weightUpdate = Mat::Zero(input_shape.second+1, output_size);

		Mat gradient;

		Mat outputs;

		Mat inputs;

		Mat inputMat;

		Mat err;

		Mat Jacobian;

		std::tuple<updateArgs...> updateParams;

		std::string name="Layer";


	public:

		static Mat makeInputMat(const Mat& input)
		{
			Mat ipm(input.rows(), input.cols() + 1);
			ipm << input, Mat::Ones(input.rows(),1);
			return ipm;
		}
		
		Layer(std::pair<int_t, int_t> _input_shape,
			  int_t _output_size,
			  const std::function<double(double)>& _activation=NN::ACTIVATIONS["relu"],
			  const std::function<Vec(std::pair<Vec, Vec>)>
			  _activation_grad=NN::ACTIVATION_DERIVATIVES["relu"]) :
															input_shape(_input_shape),
															output_size(_output_size),
															activation(_activation),
															activation_grad(_activation_grad)
		{
			weights = Mat::Random(input_shape.second + 1, output_size);
		};

		Layer(std::pair<int_t, int_t> _input_shape,
					 int_t _output_size,
					 std::string _activation, bool initWeights=true) : 
							input_shape(_input_shape),
							output_size(_output_size)
		{
			activation = NN::ACTIVATIONS[_activation];
			activation_grad = NN::ACTIVATION_DERIVATIVES[_activation];
			if(initWeights){
				weights = Mat::Random(input_shape.second +1, output_size);
			}
		};

		Layer(const Mat& _inputs,
				     int_t _output_size,
					 const Mat& _weights,
					 const std::function<double(double)>& _activation=NN::ACTIVATIONS["relu"],
					 const std::function<Vec(std::pair<Vec, Vec>)>
					_activation_grad=NN::ACTIVATION_DERIVATIVES["relu"])  :
													input_shape(std::make_pair(_inputs.rows(), _inputs.cols())),
													output_size(_output_size),
													activation(_activation),
													activation_grad(_activation_grad),
													weights(_weights),
													inputs(_inputs)
		{
			inputMat = makeInputMat(inputs);
		};


		auto getWeights() const noexcept
		{
			return weights;
		}

		auto getOutputs() const noexcept
		{
			return outputs;
		}

		auto getInputShape() const noexcept
		{
			return input_shape;
		}

		auto getOutputSize() const noexcept
		{
			return output_size;
		}

		auto getName()
		{
			return name;
		}

		void setName(std::string newName)
		{
			name = newName;
		}

		void setName(int newName)
		{
			name = "Layer " + std::to_string(newName);
		}


		void setWeights(const Mat& _weights) noexcept
		{
			weights = _weights;
		}

		void setInputShape(std::pair<int_t, int_t> _input_shape, bool reinitWeights=true) 
		{
			if(_input_shape.first <= 0 or _input_shape.second <= 0){
				throw  "Error: both elements of input_shape must be positive.";
			}
			input_shape = _input_shape;
			//if input shape is changed, reinitialize the weights as random
			if(reinitWeights){
				weights = Mat::Random(input_shape.second +1, output_size);
			}
		}

		void setOutputSize(int_t _num_outputs) noexcept
		{
			output_size = _num_outputs;
		}

		void setInputs(const Mat& _inputs, bool usemakeInputMat=true)
		{
			inputs = _inputs;
			if(usemakeInputMat){
				inputMat = makeInputMat(inputs);
			}
			setInputShape(std::make_pair(inputs.rows(), inputs.cols()));
		}


		Mat getJacobian() const
		{
			return Jacobian;
		}

		Mat getGradient() const
		{
			return gradient;
		}

		Mat getErr() const
		{
			return err;
		}

		void setUpdateParams(updateArgs... args) noexcept {
			updateParams = std::tuple<updateArgs...>(args...);
		}

		std::tuple<updateArgs...> getUpdateParams() const noexcept
		{
			return updateParams;
		}

		void setActivation(std::string actName)
		{
			activation = ACTIVATIONS[actName];
			activation_grad = ACTIVATION_DERIVATIVES[actName];
		}

		void forwardPass(const Mat& inputData)
		{
			setInputs(inputData);
			if(weights.rows() != input_shape.second + 1){
				throw "Input size error";
			} else if(weights.cols() != output_size){
				throw "Output size error";
			}
			//make activation values for each neuron
			//#pragma omp parallel
			
			actVals = inputMat * weights;
			outputs = actVals.unaryExpr(activation);
			//end omp parallel
		}

		void forwardPass()
		{
			forwardPass(inputs);
		}

		Mat makeActDerivs() const noexcept
		{
			auto actPair = std::make_pair(inputMat, outputs);
			return activation_grad(actPair);
		}


		Mat computeJacobian() noexcept
		{
			auto actDerivs = makeActDerivs();
			
			Jacobian = actDerivs * weights.transpose();
			return Jacobian;
		}

		void backwardPass(const Layer& next) noexcept
		{
			Mat loss_g = next.getErr() * next.getWeights().transpose();

			loss_g.conservativeResize(loss_g.rows(), loss_g.cols()-1);

			auto actDerivs = makeActDerivs();
			err = loss_g.cwiseProduct(actDerivs);
				
			gradient = inputMat.transpose() * err;
	
		}

		void backwardPass(const Mat& loss_grad) noexcept
		{
			//if loss_grad is supplied
			auto actDerivs = makeActDerivs();
			err = loss_grad.cwiseProduct(actDerivs);
			
			gradient = inputMat.transpose() * err;
		}

		void updateWeights()
		{
			if constexpr(update == UpdateRule::NesterovAccGrad){
				//if we're using Nesterov accelerated grad, params are learning rate, momentum
				double learningRate, momentum;
				std::tie(learningRate, momentum) = updateParams;
	
				weightUpdate = momentum * weightUpdate - learningRate * gradient;

				weights += weightUpdate;
			} else {
				throw "Error: only NesterovAccGrad is implemented now.";
			}
		}

		void updateWeights(const std::tuple<updateArgs...>& params)
		{
			updateParams = params;
			updateWeights();
		}

		void visualizeLayer() 
		{
			std::cout << "\n================  " << name << "  ================\n\n";
			std::cout << " ([inputs,1] * [weights]) -> activation -> outputs   \n";
			std::cout << " (             [  bias ])                          \n\n";
			std::cout << " \nInputs:\n" << inputs << '\n';
			std::cout << " \nWeights (last row is bias):\n" << weights << '\n';
			std::cout << " \n[inputs,1] * [weights, bias]^T:\n" << actVals << '\n';
			std::cout << " \nOutputs:\n" << outputs << '\n';
			std::cout << " ===================================================\n";
		}
		
	};//end class Layer



}//end namespace NN
#endif
