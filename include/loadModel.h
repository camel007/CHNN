#ifndef CAFFE_LOAD_MODEL_HPP_
#define CAFFE_LOAD_MODEL_HPP_
#include <cstdio>
#include <vector>
#include "caffe.pb.h"
class loadModel
{
	std::FILE *fin;
public:
	loadModel()			{ fin = nullptr; }
	~loadModel()		{ if(fin!= nullptr) std::fclose(fin); }

public:
	void load(const char *filename)
	{
		fin = std::fopen(filename, "rb");
		if (!fin)
		{
			std::cout << "Loading trained model wrong!\n";
			return;
		}
	}

	void readConvParam(caffe::LayerParameter &layer_param_)
	{
		layer_param_.size = 2;
		layer_param_.shapes_.resize(2);
		layer_param_.weightAndBias.resize(2);
		caffe::ConvolutionParameter& conv_param = layer_param_.convParam;

		std::fread(&conv_param.pad_, sizeof(int), 1, fin);
		std::fread(&conv_param.stride_, sizeof(int), 1, fin);
		//std::fread(&conv_param.kernel_, sizeof(int), 1, fin);
		//std::fread(&conv_param.output_, sizeof(int), 1, fin);

		layer_param_.shapes_[0].resize(4);//weights parameters
		
		std::fread(&layer_param_.shapes_[0][0], sizeof(int), 4, fin);
		int *p = &layer_param_.shapes_[0][0];
		conv_param.kernel_ = p[2];
		conv_param.output_ = p[0];
		int weightLength = p[0] * p[1] * p[2] * p[3];
		layer_param_.weightAndBias[0].resize(weightLength);
		std::fread(&layer_param_.weightAndBias[0][0], sizeof(float), weightLength, fin);

		layer_param_.shapes_[1].resize(1);//bias parameters
		std::fread(&layer_param_.shapes_[1][0], sizeof(int), 1, fin);
		int biasLength = layer_param_.shapes_[1][0];
		layer_param_.weightAndBias[1].resize(biasLength);
		std::fread(&layer_param_.weightAndBias[1][0], sizeof(float), biasLength, fin);
	}
	void readPoolParam(caffe::LayerParameter &layer_param_)
	{
		layer_param_.size = 0;
		caffe::PoolingParameter& pool_param = layer_param_.poolParam;

		std::fread(&pool_param.pad_, sizeof(int), 1, fin);
		std::fread(&pool_param.stride_, sizeof(int), 1, fin);
		std::fread(&pool_param.kernel_size_, sizeof(int), 1, fin);
	}
	void readInnerProductParam(caffe::LayerParameter &layer_param_)
	{
		layer_param_.size = 2;
		layer_param_.shapes_.resize(2);
		layer_param_.weightAndBias.resize(2);
		caffe::InnerproductParameter& inner_param = layer_param_.InnerParam;

		layer_param_.shapes_[0].resize(2);//weight parameters
		std::fread(&layer_param_.shapes_[0][0], sizeof(int), 2, fin);
		inner_param.num_output = layer_param_.shapes_[0][0];
		int weightLength = layer_param_.shapes_[0][0] * layer_param_.shapes_[0][1];
		layer_param_.weightAndBias[0].resize(weightLength);
		std::fread(&layer_param_.weightAndBias[0][0], sizeof(float), weightLength, fin);

		layer_param_.shapes_[1].resize(1);//bias parameters
		std::fread(&layer_param_.shapes_[1][0], sizeof(int), 1, fin);
		int biasLength = layer_param_.shapes_[1][0];
		layer_param_.weightAndBias[1].resize(biasLength);
		std::fread(&layer_param_.weightAndBias[1][0], sizeof(float), biasLength, fin);
	}
};

#endif
