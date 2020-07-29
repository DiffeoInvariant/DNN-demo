#include "Network.hpp"

namespace NN
{


  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setInputs(const Mat& _inputs, bool overrideInputShape)
  {
    if(not overrideInputShape) {
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


  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setTarget(const Vec& _target, bool overrideTargetSize)
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setLayers(const std::list<Layer<update, updateArgs...>>& newLayers)
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::appendLayers(std::list<Layer<update, updateArgs...>>& newLayers)
  {
    for(const auto& it : newLayers){
      layer_input_shapes.push_back(it.getInputShape());
    }
			
    num_outputs = layer_input_shapes.back().second;

    layers.splice(layers.end(), newLayers);
  }


  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::insertLayer(typename std::list<Layer<update, updateArgs...>>::iterator& location,
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


  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setWeights(const std::list<Mat>& weights)
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setUpdateParams(const std::list<std::tuple<updateArgs...>>& argsList)
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::setActivations(const std::list<std::string>& activations)
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::predict(std::optional<Mat> inputData,
					      std::optional<Vec> _target) 
  {
    if(inputData){
      setInputs(*inputData);
    }
    if(_target){
      setTarget(*_target);
    }
			
    Mat layerOut = inputs;
    for(auto& l : layers) {
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

  template<UpdateRule update, typename... updateArgs>
  Vec Network<update,updateArgs...>::predictVal(std::optional<Mat> inputData,
						std::optional<Vec> _target) 
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

  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::backwardPass()
  {
    //from the second-to-last layer, iterate to the beginning
    bool isFirst = true;
    auto prevLayer = layers.back();
    for(auto l=layers.rbegin(); l != layers.rend(); l++){
      if(isFirst){
	l->backwardPass(loss_deriv);
	isFirst = false;
      } else {
	l->backwardPass(prevLayer);
      }
      prevLayer = *l;
    }
    gradient = layers.front().getGradient();  
  }


  template<UpdateRule update, typename... updateArgs>
  void Network<update,updateArgs...>::train(double stopTol, 
					    size_t maxIter,
					    std::optional<Mat> inputData,
					    std::optional<Vec> _newtarget,
					    bool noprint)
  {
    //run first training round
    predict(inputData, _newtarget);
    backwardPass();
    trainingLoss.push_back(scalar_loss);
    size_t num_iter = 1;
    updateWeights();
    //run until stopping criteria are hit
    while(num_iter < maxIter and gradient.norm() > stopTol) {
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
  



}//namespace NN
