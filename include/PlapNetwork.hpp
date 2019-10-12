#ifndef PLAP_NETWORK_HPP
#define PLAP_NETWORK_HPP

#include "Layer.hpp"
#include "Network.hpp"
#include <utility>
#include <string>
#include <cmath>

namespace NN
{

	template<UpdateRule update, typename... updateArgs>
	class PlapFinalLayer : protected Layer<update, updateArgs...>
	{
	protected:

		double p = 2.0;

		double a = 1.0;

		Vec actVals;
		
		std::function<double(double)> activation;

		std::function<Mat(std::pair<Mat, Mat>)> activation_grad;

		Vec output;

		Mat inputs;

		Mat Jacobian;

	public:
		PlapFinalLayer(std::pair<int_t, int_t> _input_shape,
					   int_t num_outputs=1,
					   std::string _activation="sigmoid",
					   double _p=2.0,
					   double _a=1.0) :
								Layer<update, updateArgs...>(_input_shape, num_outputs, _activation, false),
								p(_p),
								a(_a),
								activation(ACTIVATIONS[_activation]),
								activation_grad(ACTIVATION_DERIVATIVES[_activation])
		{

		};

		void setInputs(const Mat& _inputs)
		{
			Layer<update, updateArgs...>::setInputs(_inputs, false);
			inputs = _inputs;
		}

		auto getOutput()
		{
			return output;
		}

		void forwardPass()
		{
			actVals = Vec::Zero(inputs.cols());
			int_t count = 0;
			for(auto& col : inputs.colwise()){
				double aval = col.squaredNorm();
				actVals[count] = aval;
				count++;
			}
			//make activation
			output = actVals;
			count = 0;
			for(auto& it : actVals){
				output[count] = a * pow(it, (p-2)*0.5);
				count++;
			}
		}


		void forwardPass(const Mat& inputData)
		{
			setInputs(inputData);
			forwardPass();
		}

		void computeJacobian()
		{
			





	};


}
#endif //PLAP_NETWORK_HPP
