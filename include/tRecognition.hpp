#pragma once
#ifndef __FACE_RECOGNITOIN_H
#define __FACE_RECOGNITION_H
#include <opencv2/opencv.hpp>
#include "caffe.hpp"
#include "loadModel.h"

static std::vector<std::string> rec_layer_names = { "INPUT", \
												"CONVOLUTION", "RELU", "CONVOLUTION", "RELU","POOLING",\
												"CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "POOLING",\
												"CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "POOLING",\
												"CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "POOLING",\
												"CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "CONVOLUTION", "RELU", "POOLING",\
												"INNERPRODUCT", "RELU",\
												"INNERPRODUCT", "RELU"\
											  };


class faceRecognition{
	loadModel lm;
	int layer_sz;

	std::vector<std::shared_ptr<caffe::Layer<float> > > layers_;
	std::vector<caffe::LayerParameter> param;
	std::vector<std::vector<caffe::Blob<float>*>> bottom_vecs_;
	std::vector<std::vector<caffe::Blob<float>*>> top_vecs_;
	/// @brief the blobs storing intermediate results between the layer.
	std::vector<std::shared_ptr<caffe::Blob<float> > > blobs_;
	std::vector<int> input_shape;
public:
	faceRecognition()		{ layer_sz = 0; }
	~faceRecognition()		{}

	//face detection
	void init(const char *filename, bool is_gpu = true);

	void forward();

	void setData(std::vector<cv::Mat> &candidateFaces);

	std::vector<std::vector<float> > getResult();

protected:
	/// @brief Append a new top blob to the net.
	void AppendTop(int layer_id);
	/// @brief Append a new bottom blob to the net.
	void AppendBottom(int layer_id, int bottom_id);
	/// @brief wrap image to blob
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > &input_batch, float *input_data, int width, int height, int channels, int num);
	/// @preprocess image to input layer
	void PreprocessBatch(const std::vector<cv::Mat> &imgs, std::vector< std::vector<cv::Mat> > &input_batch, int height, int width);
};
#endif
