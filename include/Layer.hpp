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
  static std::unordered_map<std::string, std::function<double(double)>>
  ACTIVATIONS = {
		 {"linear", [](double x){ return x;}},
		 {"sigmoid", [](double x){ return 1.0/(1.0 + exp(-x));}},
		 {"tanh", [](double x){ return tanh(x);}},
		 {"relu", [](double x){ return x > 0 ? x : 0.0;}},
		 {"softplus", [](double x){ return log(exp(x) + 1);}}
  };
	
  /*
   * takes pairs of (input, output)
   * */
  static std::unordered_map<std::string, std::function<Mat(std::pair<Mat, Mat>)>>
  ACTIVATION_DERIVATIVES = {
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

    std::tuple<double,double> updateParams;

    std::string name="Layer";

    UpdateRule update=UpdateRule::NesterovAccGrad;


  public:

    static Mat makeInputMat(const Mat& input);
    
		
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

    void setInputShape(std::pair<int_t, int_t> _input_shape, bool reinitWeights=true); 

    void setOutputSize(int_t _num_outputs) noexcept
    {
      output_size = _num_outputs;
    }

    void setInputs(const Mat& _inputs, bool usemakeInputMat=true);


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

    void setUpdateParams(double learningrate, double momentum) noexcept {
      updateParams = std::tuple<double,double>(learningrate,momentum);
    }

    std::tuple<double,double> getUpdateParams() const noexcept
    {
      return updateParams;
    }

    void setActivation(std::string actName);

    void forwardPass(const Mat& inputData);

    void forwardPass();

    Mat makeActDerivs() const noexcept
    {
      auto actPair = std::make_pair(inputMat, outputs);
      return activation_grad(actPair);
    }


    Mat computeJacobian() noexcept;

    void backwardPass(const Layer& next) noexcept;
    

    void backwardPass(const Mat& loss_grad) noexcept;

    void updateWeights();

    void updateWeights(double mult);


    void updateWeights(const std::tuple<double,double>& params)
    {
      updateParams = params;
      updateWeights();
    }

    void visualizeLayer(std::ostream& ostr = std::cout);
		
  };//end class Layer


}//end namespace NN
#endif
