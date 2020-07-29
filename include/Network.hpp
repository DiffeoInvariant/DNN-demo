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
  std::unordered_map<std::string, std::function<double(Vec, Vec)>>
  VECTOR_LOSS = {
		 {"L2", [](Vec pred, Vec obs) -> double {
			  Vec resid = pred - obs;
			  return 0.5 * resid.dot(resid);
			}
		 }
  };

  std::unordered_map<std::string, std::function<Vec(Vec, Vec)>>
  VECTOR_LOSS_DERIVATIVE = {
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
	    std::string loss="L2")
      : input_shape(_input_shape),
	num_outputs(_num_outputs),
	vector_loss_func(VECTOR_LOSS[loss]),
	vector_loss_derivative(VECTOR_LOSS_DERIVATIVE[loss])

    {
      Layer<update, updateArgs...> finalLayer(input_shape, num_outputs, activation);
      layers.push_back({finalLayer});

      layer_input_shapes.push_back({input_shape});

      Eigen::setNbThreads(0);
    };

    Network(std::initializer_list<Layer<update, updateArgs...>> _layers)
      : layers(_layers) 
    {
      for(const auto& it : layers){
	layer_input_shapes.push_back(it.getInputShape());
      }
      input_shape = layer_input_shapes.front();

      num_outputs = layers.back().getOutputSize();

      Eigen::setNbThreads(0);

    };

    Network(std::string activation,
	    std::string loss,
	    std::initializer_list<Layer<update, updateArgs...>> _layers)
      : layers(_layers),
	vector_loss_func(VECTOR_LOSS[loss]),
	vector_loss_derivative(VECTOR_LOSS_DERIVATIVE[loss])

    {
      for(auto& it : layers){
	layer_input_shapes.push_back(it.getInputShape());
	it.setActivation(activation);
      }
      input_shape = layer_input_shapes.front();

      num_outputs = layers.back().getOutputSize();

      Eigen::setNbThreads(0);

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

      Eigen::setNbThreads(0);

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

    void setNumThreads(int n){
      Eigen::setNbThreads(n);
    }

    void setInputs(const Mat& _inputs, bool overrideInputShape=false);

    void setTarget(const Vec& _target, bool overrideTargetSize=false);

    void setLayers(const std::list<Layer<update, updateArgs...>>& newLayers);

    //moves layers onto end of list
    void appendLayers(std::list<Layer<update, updateArgs...>>& newLayers);
    

    void insertLayer(typename std::list<Layer<update, updateArgs...>>::iterator& location,
		     const Layer<update, updateArgs...>& newLayer);
    

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

    void setWeights(const std::list<Mat>& weights);
    
    //gives all layers the same update params
    void setUpdateParams(updateArgs... args) noexcept
    {
      for(auto& l : layers){
	l.setUpdateParams(args...);
      }
    }
    //different args for each layer
    void setUpdateParams(const std::list<std::tuple<updateArgs...>>& argsList);

    //same activation for each layer
    void setActivations(std::string activations) noexcept
    {
      for(auto& l : layers){
	l.setActivation(activations);
      }
    }

    void setActivations(const std::list<std::string>& activations);

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
		 std::optional<Vec> _target=std::nullopt);
		
    //same as predict(), but returns the output
    Vec predictVal(std::optional<Mat> inputData=std::nullopt,
		   std::optional<Vec> _target=std::nullopt);

    //computes gradient of network
    void backwardPass();

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
	       std::optional<Vec> _newtarget=std::nullopt,
	       bool noprint=false);		

    void summary();

    void visualizeNetwork();

  };


}//end namespace NN
#endif
