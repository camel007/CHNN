#include <algorithm>
#include <cfloat>
#include <vector>

#include "pooling_layer.hpp"
#include "math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.poolParam;
  kernel_h_ = kernel_w_ = pool_param.kernel_size();

  pad_h_ = pad_w_ = pool_param.pad();
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  stride_h_ = stride_w_ = pool_param.stride();
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	//if (global_pooling_) {
	//	kernel_h_ = bottom[0]->height();
	//	kernel_w_ = bottom[0]->width();
	//}
	pooled_height_ = static_cast<int>(ceil(static_cast<float>(
		height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
	pooled_width_ = static_cast<int>(ceil(static_cast<float>(
		width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
	if (pad_h_ || pad_w_) {
		// If we have padding, ensure that the last pooling starts strictly
		// inside the image (instead of at the padding); otherwise clip the last.
		if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
			--pooled_height_;
		}
		if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
			--pooled_width_;
		}
		CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
		CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
	}
	top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
		pooled_width_);
	if (top.size() > 1) {
		top[1]->ReshapeLike(*top[0]);
	}
	// If max pooling, we will initialize the vector index part.
	if (top.size() == 1) {
		max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
			pooled_width_);
	}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
	//std::cout << "------------------------" << std::endl;
	//std::copy(top[0]->mutable_cpu_data(), top[0]->mutable_cpu_data() + int(0.5f*top_count), std::ostream_iterator<Dtype>(std::cout, "\t"));
	//std::copy(top[0]->mutable_cpu_data() + int(0.5f*top_count), top[0]->mutable_cpu_data() + top_count, std::ostream_iterator<Dtype>(std::cout, "\t"));
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
