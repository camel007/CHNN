#include <algorithm>
#include <vector>

#include "relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i) {
		top_data[i] = std::max(bottom_data[i], Dtype(0));
	}
}
#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
