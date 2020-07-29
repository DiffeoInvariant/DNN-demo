#include <Layer.hpp>


namespace NN
{ 
  template<UpdateRule update, typename... updateArgs>
  Mat Layer<update,updateArgs...>::makeInputMat(const Mat& input)
  {
    Mat ipm(input.rows(), input.cols() + 1);
    ipm << input, Mat::Ones(input.rows(),1);
    return ipm;
  }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::setInputShape(std::pair<int_t, int_t> _input_shape,
						  bool reinitWeights)
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

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::setInputs(const Mat& _inputs, bool usemakeInputMat)
  {
    inputs = _inputs;
    if(usemakeInputMat){
      inputMat = makeInputMat(inputs);
    }
    setInputShape(std::make_pair(inputs.rows(), inputs.cols()));
  }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::setActivation(std::string actName)
  {
    activation = ACTIVATIONS[actName];
    activation_grad = ACTIVATION_DERIVATIVES[actName];
  }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::forwardPass(const Mat& inputData)
  {
    setInputs(inputData);
    if(weights.rows() != input_shape.second + 1){
      throw "Input size error";
    } else if(weights.cols() != output_size){
      throw "Output size error";
    }
    //make activation values for each neuron
    #pragma omp parallel
    {
      actVals = inputMat * weights;
    }

    #pragma omp parallel
    {
      outputs = actVals.unaryExpr(activation);
    }
    //end omp parallel
  }



  template<UpdateRule update, typename... updateArgs>
  Mat Layer<update,updateArgs...>::computeJacobian() noexcept
  {
    auto actDerivs = makeActDerivs();
    
    Jacobian = actDerivs * weights.transpose();
    return Jacobian;
  }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::backwardPass(const Layer& next) noexcept
  {
    Mat loss_g;
    #pragma omp parallel
    {
     loss_g = next.getErr() * next.getWeights().transpose();
    }

    loss_g.conservativeResize(loss_g.rows(), loss_g.cols()-1);

    auto actDerivs = makeActDerivs();
    err = loss_g.cwiseProduct(actDerivs);
				
    gradient = inputMat.transpose() * err;
	
    }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::backwardPass(const Mat& loss_grad) noexcept
  {
    //if loss_grad is supplied
    auto actDerivs = makeActDerivs();
    err = loss_grad.cwiseProduct(actDerivs);
			
    gradient = inputMat.transpose() * err;
  }


  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::updateWeights()
  {
    if constexpr(update == UpdateRule::NesterovAccGrad){
	//if we're using Nesterov accelerated grad, params are learning rate, momentum
	auto [learningRate, momentum] = updateParams;
	
        #pragma omp parallel
	{
	  weightUpdate = momentum * weightUpdate - learningRate * gradient;

	  weights += weightUpdate;
	}
      } else {
      throw "Error: only NesterovAccGrad is implemented now.";
    }
  }


  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::updateWeights(double mult)
  {
    updateWeights();
    #pragma omp parallel
    weights *= mult;
  }

  template<UpdateRule update, typename... updateArgs>
  void Layer<update,updateArgs...>::visualizeLayer(std::ostream& ostr) 
  {
    ostr << "\n================  " << name << "  ================\n\n";
    ostr << " ([inputs,1] * [weights]) -> activation -> outputs   \n";
    ostr << " (             [  bias ])                          \n\n";
    ostr << " \nInputs:\n" << inputs << '\n';
    ostr << " \nWeights (last row is bias):\n" << weights << '\n';
    ostr << " \n[inputs,1] * [weights, bias]^T:\n" << actVals << '\n';
    ostr << " \nOutputs:\n" << outputs << '\n';
    ostr << " ===================================================\n";
  }


  

}
