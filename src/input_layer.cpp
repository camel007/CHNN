#include <vector>

#include "input_layer.hpp"

namespace caffe {

template <typename Dtype>
void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param;
  const int num_shape = 1;
  CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
		const int shape_index = (num_shape == 1) ? 0 : i;
      top[i]->Reshape(param.shape_);
    }
  }
}

INSTANTIATE_CLASS(InputLayer);
}  // namespace caffe
