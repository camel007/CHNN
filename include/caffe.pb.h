#pragma once
#include <iostream>
#include <vector>
#include "math_functions.hpp"

namespace caffe{
	class ConvolutionParameter
	{
	public:
		ConvolutionParameter()
		{
			kernel_ = 1;
			stride_ = 1;
			pad_ = 0;
			dilation_ = 1;
			output_ = 0;
		}
		~ConvolutionParameter()	{}

	public:
		int kernel_size() const
		{
			return kernel_;
		}
		int stride() const
		{
			return stride_;
		}
		int pad() const
		{
			return pad_;
		}
		int dilation() const
		{
			return dilation_;
		}
		int output() const
		{
			return output_;
		}

		int kernel_;
		int stride_;
		int pad_;
		int dilation_;
		int output_;
	};
	class PoolingParameter
	{
	public:
		PoolingParameter()
		{
			kernel_size_ = 2;
			pad_ = 0;
			stride_ = 2;
		}
		~PoolingParameter()	{}

		int kernel_size() const
		{ return kernel_size_; }
		int pad() const
		{ return pad_; }
		int stride() const
		{ return stride_; }

	public:
		int kernel_size_;
		int pad_;
		int stride_;
	};
	class InnerproductParameter
	{
	public:
		InnerproductParameter()
		{
			num_output = 0;
		}
		~InnerproductParameter()	{}

		int num_output;
	};
	class SoftmaxParameter{
	public:
		SoftmaxParameter(){
			num_output = 0;
		}
		~SoftmaxParameter(){}

		int num_output;
	};
	class InputParameter
	{
	public:
		InputParameter()	{}
		~InputParameter()	{}

		std::vector<int> shape_;
	};

	class LayerParameter
	{
	public:
		LayerParameter()
		{
			size = 0;
		}
		~LayerParameter()	{}

	public:
		int blobs_size() const
		{
			return size;
		}

		std::vector<int> shape(int idx) const
		{
			if (idx >= size) std::cout << "Wrong idx" << std::endl;
			
			return shapes_[idx];
		}

		std::vector<std::vector<int>> shapes_;
		int size;//可学习的参数个数;eg, convolution层的weight算一个，bias算一个
		
		std::vector<std::vector<float>> weightAndBias;

		InputParameter input_param;
		ConvolutionParameter convParam;
		PoolingParameter poolParam;
		InnerproductParameter InnerParam;
		SoftmaxParameter softParam;
	};
}
