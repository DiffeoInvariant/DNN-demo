#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Layer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_map>
#include <functional>
#include <optional>
#include <utility>
#include <string>
#include <cmath>
#include <random>
#include <memory>
#include <list>
#include <vector>
#include <initializer_list>
#include <iterator>
#include <iostream>
#include <omp.h>


namespace NN
{
	std::unordered_map<std::string, std::function<double(Vec, Vec)>> VECTOR_LOSS = {
		{"L2", [](Vec pred, Vec obs) -> double {
					Vec resid = pred - obs;
					return 0.5 * resid.dot(resid);
				} }
	};

	std::unordered_map<std::string, std::function<Vec(Vec, Vec)>> VECTOR_LOSS_DERIVATIVE = {
		{"L2", [](Vec pred, Vec obs) -> Vec { return pred - obs; } }
	};

	template<UpdateRule update, typename... updateArgs>
	class Network
	{
	protected:
		
		std::pair<int_t, int_t> input_shape;

		Mat inputs;

		int_t num_outputs;

		Vec outputs;

		Vec target;

		std::list<Layer<update, updateArgs...>> layers;

		std::list<std::pair<int_t, int_t>> layer_input_shapes;

		std::function<double(Vec,Vec)> vector_loss_func;

		std::function<Vec(Vec,Vec)> vector_loss_derivative;

		Vec loss_deriv;

		double scalar_loss;

		std::vector<double> trainingLoss;

		Vec resid;

		Mat gradient;

	public:

		Network(std::pair<int_t, int_t> _input_shape,
				int_t _num_outputs,
				std::string activation,
				std::string loss="L2") : input_shape(_input_shape),
										num_outputs(_num_outputs),
										vector_loss_func(VECTOR_LOSS[loss]),
										vector_loss_derivative(VECTOR_LOSS_DERIVATIVE[loss])

		{
			Layer<update, updateArgs...> finalLayer(input_shape, num_outputs, activation);
			layers({finalLayer});

			layer_input_shapes({input_shape});
		};

		Network(std::initializer_list<Layer<update, updateArgs...>> _layers) : layers(_layers) 
		{
			for(const auto& it : layers){
				layer_input_shapes.push_back(it.getInputShape());
			}
			input_shape = layer_input_shapes.front();

			num_outputs = layers.back().getOutputSize();
		};

		Network(std::string activation,
				std::string loss,
				std::initializer_list<Layer<update, updateArgs...>> _layers) : layers(_layers),
														vector_loss_func(VECTOR_LOSS[loss]),
														vector_loss_derivative(VECTOR_LOSS_DERIVATIVE[loss])

		{
			for(auto& it : layers){
				layer_input_shapes.push_back(it.getInputShape());
				it.setActivation(activation);
			}
			input_shape = layer_input_shapes.front();

			num_outputs = layers.back().getOutputSize();
		};

		Network(std::string activation,
				std::string loss,
				const std::list<std::pair<int_t, int_t>> _layer_input_shapes) :
														layer_input_shapes(_layer_input_shapes),
														vector_loss_func(VECTOR_LOSS[loss]),
														vector_loss_derivative(VECTOR_LOSS_DERIVATIVE[loss])
		{
			std::list<Layer<update, updateArgs...>> layerList;
			for(const auto& lis : layer_input_shapes){
				layerList.push_back(Layer<update, updateArgs...>(lis, lis.first, activation));
			}
			layers = layerList;
			num_outputs = layers.back().getOutputSize();

			input_shape = layer_input_shapes.front();
		};


		auto getLayers() const noexcept
		{
			return layers;
		}

		auto getLayerInputShapes() const noexcept
		{
			return layer_input_shapes;
		}

		auto getInputShape() const noexcept
		{
			return input_shape;
		}

		auto getNumOutputs() const noexcept
		{
			return num_outputs;
		}

		auto getOutputs() const
		{
			return outputs;
		}

		auto getTarget() const
		{
			return target;
		}

		auto getLossDeriv() const
		{
			return loss_deriv;
		}

		auto getScalarLoss() const
		{
			return scalar_loss;
		}

		auto getGradient() const
		{
			return gradient;
		}

		auto getLossHistory() const 
		{
			return trainingLoss;
		}

		//gets weights for first layer
		auto getFirstWeights() const
		{
			return layers.front().getWeights();
		}

		void setInputs(const Mat& _inputs, bool overrideInputShape=false)
		{
			if(not overrideInputShape)
			{
				if(_inputs.rows() != input_shape.first){
					throw "Error: new input matrix must have number of rows of input_shape.first";
				} else if(_inputs.cols() != input_shape.second){
					throw "Error: new input matrix must have number of cols of input_shape.second";
				}
				inputs = _inputs;
			} else {
				inputs = _inputs;
				input_shape = std::make_pair(inputs.rows(), inputs.cols());
				layers.front().setInputs(inputs);
				layer_input_shapes.front() = input_shape;
			}
		}

		void setTarget(const Vec& _target, bool overrideTargetSize=false)
		{
			if(_target.size() != num_outputs){
				if(overrideTargetSize){
					num_outputs = target.cols();
					layers.back().setOutputSize(num_outputs);
					layers.back().setInputShape(layer_input_shapes.back());
					target = _target;
				} else {
					throw "Error: _target must have length equal to num_outputs";
				}
			} else {
				target = _target;
			}
		}

		void setLayers(const std::list<Layer<update, updateArgs...>>& newLayers)
		{

			layers = newLayers;

			std::list<Layer<update, updateArgs...>> new_layer_input_shapes;

			for(const auto& it : layers){
				new_layer_input_shapes.push_back(it.getInputShape());
			}
			input_shape = new_layer_input_shapes.front();

			num_outputs = new_layer_input_shapes.back().second;

			layer_input_shapes = new_layer_input_shapes;
		}

		//moves layers onto end of list
		void appendLayers(std::list<Layer<update, updateArgs...>>& newLayers)
		{

			for(const auto& it : newLayers){
				layer_input_shapes.push_back(it.getInputShape());
			}
			
			num_outputs = layer_input_shapes.back().second;

			layers.splice(layers.end(), newLayers);

		}

		void insertLayer(typename std::list<Layer<update, updateArgs...>>::iterator& location,
						const Layer<update, updateArgs...>& newLayer)
		{
			if(location == layers.end()){
				//if location is the end
				appendLayers({newLayer});
			} else {
				layers.insert(location, newLayer);

				std::list<Layer<update, updateArgs...>> new_layer_input_shapes;

				//update shape list
				for(const auto& it : layers){
					new_layer_input_shapes.push_back(it.getInputShape());
				}
				input_shape = new_layer_input_shapes[0];

				num_outputs = new_layer_input_shapes.back().second;

				layer_input_shapes = new_layer_input_shapes;
			}
		}

		std::list<Mat> getWeights() const noexcept
		{
			std::list<Mat> weightList;
			for(const auto& l : layers){
				weightList.push_back(l.getWeights());
			}
			return weightList;
		}
		
		//for each layer, gets pair of error, gradient. Network must have gone through
		//at least one backwards pass
		std::list<std::pair<Mat, Mat>> getErrGradientList() const
		{
			std::list<std::pair<Mat, Mat>> eglist;
			for(const auto& l : layers){
				eglist.push_back(std::make_pair(l.getErr(), l.getGradient()));
			}
			return eglist;
		}

		void setWeights(const std::list<Mat>& weights)
		{
			if(weights.size() != layers.size()){
				throw "Error: must provide exactly one weight matrix for each layer.";
			}
			auto wit = weights.begin();
			for(auto& l : layers){
				l.setWeights(*wit);
				std::advance(wit,1);
			}
		}
		//gives all layers the same update params
		void setUpdateParams(updateArgs... args) noexcept
		{
			for(auto& l : layers){
				l.setUpdateParams(args...);
			}
		}
		//different args for each layer
		void setUpdateParams(const std::list<std::tuple<updateArgs...>>& argsList)
		{
			if(argsList.size() != layers.size()){
				throw "Error: must provide exactly one args tuple for each layer.";
			}
			auto alit = argsList.begin();
			for(auto& l : layers){
				std::apply(l.setUpdateArgs, *alit);
				std::advance(alit, 1);
			}
		}

		//same activation for each layer
		void setActivations(std::string activations) noexcept
		{
			for(auto& l : layers){
				l.setActivation(activations);
			}
		}

		void setActivations(const std::list<std::string>& activations)
		{
			if(activations.size() != layers.size()){
				throw "Error: must provide exactly one activation for each layer";
			}
			auto ait = activations.begin();
			for(auto& l : layers){
				l.setActivation(*ait);
				std::advance(ait,1);
			}
		}

		void setLossFunc(std::string loss)
		{
			vector_loss_func = VECTOR_LOSS[loss];
			vector_loss_derivative = VECTOR_LOSS_DERIVATIVE[loss];
		}
		
		void setLossFunc(const std::function<double(Vec,Vec)>& _vector_loss_func,
						 const std::function<Vec(Vec,Vec)>& _vector_loss_derivative)
		{
			vector_loss_func = _vector_loss_func;
			vector_loss_derivative = _vector_loss_derivative;
		}

		//runs a forward pass through the layers, returning the output if desired
		void predict(std::optional<Mat> inputData=std::nullopt,
								   std::optional<Vec> _target=std::nullopt) 
		{
			if(inputData){
				setInputs(*inputData);
			}
			if(_target){
				setTarget(*_target);
			}
			
			Mat layerOut = inputs;
			for(auto& l : layers)
			{
				l.setInputs(layerOut);

				l.forwardPass();
				//output of this layer is input to the next layer, then eventually the output
				layerOut = l.getOutputs();
			}
			outputs = layerOut;

			resid = outputs - target;

			scalar_loss = vector_loss_func(outputs, target);

			loss_deriv = vector_loss_derivative(outputs, target);

		}
		
		//same as predict(), but returns the output
		Vec predictVal(std::optional<Mat> inputData=std::nullopt,
								   std::optional<Vec> _target=std::nullopt) 
		{
			if(inputData){
				setInputs(*inputData);
			}
			if(_target){
				setTarget(*_target);
			}
			
			Mat layerOut = inputs;
			for(auto& l : layers)
			{
				l.setInputs(layerOut);

				l.forwardPass();
				//output of this layer is input to the next layer, then eventually the output
				layerOut = l.getOutputs();
			}
			outputs = layerOut;

			resid = outputs - target;

			scalar_loss = vector_loss_func(outputs, target);

			loss_deriv = vector_loss_derivative(outputs, target);

			return outputs;
		}

		//computes gradient of network
		void backwardPass()
		{
			//from the second-to-last layer, iterate to the beginning
			bool isFirst = true;
			auto prevLayer = layers.back();
			for(auto l=layers.rbegin(); l != layers.rend(); l++){
				if(isFirst){
					(*l).backwardPass(loss_deriv);
					isFirst = false;
				} else {
					(*l).backwardPass(prevLayer);
				}
				prevLayer = *l;
			}
			gradient = layers.front().getGradient();
		}

		void updateWeights()
		{
			for(auto& l : layers){
				l.updateWeights();
			}
		}

		void updateWeights(const std::list<std::tuple<updateArgs...>>& argsList)
		{
			setUpdateParams(argsList);
			updateWeights();
		}

		void updateWeights(updateArgs... args)
		{
			setUpdateParams(args...);
			updateWeights();
		}

		void train(double stopTol=1.0e-5, 
				   size_t maxIter=1.0e3,
				   std::optional<Mat> inputData=std::nullopt,
				   std::optional<Vec> _target=std::nullopt,
				   bool noprint=false)
		{
			//run first training round
			predict(inputData=inputData, _target=_target);
			backwardPass();
			trainingLoss.push_back(scalar_loss);
			size_t num_iter = 1;

			updateWeights();
			//run until stopping criteria are hit
			while(num_iter < maxIter and gradient.norm() > stopTol)
			{
				predict();
				backwardPass();
				trainingLoss.push_back(scalar_loss);
				updateWeights();
				num_iter++;
			}
			if(num_iter >= maxIter and not noprint){
				std::cout << "WARNING: NETWORK HIT MAX ITERATIONS IN TRAINING. SCALAR LOSS IS "
					<< scalar_loss << ". \n";
			}
		}
			

		void summary()
		{
			std::cout << "===============================\n";
			std::cout << "      Network Summary:\n\n";
			std::cout << " (input size) -> (output size)\n\n";
			size_t count = 1;
			for(const auto& ls : layers){
				std::cout << "Layer " << count << ": (" << ls.getInputShape().first
					<< " x " << ls.getInputShape().second << ") -> (" << ls.getOutputSize() << ") \n";
			}
			std::cout << "===============================\n";
		}

	};






}//end namespace NN
#endif
