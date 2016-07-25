#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include "blob.hpp"
#include "common.hpp"
#include "caffe.pb.h"
#include "math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)//显式构造函数
    : layer_param_(param){
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>(layer_param_.shape(i)));
		  blobs_[i]->fromProto(&layer_param_.weightAndBias[i][0]);
        }
      }
    }
  virtual ~Layer() {}

  void SetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	  LayerSetUp(bottom, top);
	  Reshape(bottom, top);
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  //从layer_param_中读取相关参数，并做处理，然后设定top blobs的shape和中间buffer
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  //纯虚函数，根据bottom blob 设定top blob，应该会用到一些参数，比如pooling层，top blob的大小就减半
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline void Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  std::vector<std::shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;

  /** The vector that stores the learnable parameters as a set of blobs. */
  std::vector<std::shared_ptr<Blob<Dtype> > > blobs_;

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Lock during forward to ensure sequential forward
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
    break;
  default:
    std::cout << "Unknown caffe mode.";
  }
}
}  // namespace caffe

#endif  // CAFFE_LAYER_H_
