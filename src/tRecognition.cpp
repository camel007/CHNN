#include "tRecognition.hpp"

//face detection
void faceRecognition::init(const char *filename, bool is_gpu){
	if (is_gpu)
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	else
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	lm.load(filename);

	layer_sz = rec_layer_names.size();
	layers_.resize(layer_sz);
	param.resize(layer_sz);
	bottom_vecs_.resize(layer_sz);
	top_vecs_.resize(layer_sz);

	input_shape.push_back(1);
	input_shape.push_back(3);
	input_shape.push_back(224);
	input_shape.push_back(224);

	for (int i = 0; i < layer_sz; i++)
	{
		std::string name = rec_layer_names[i];
		if (name == "INPUT")
		{
			param[i].size = 0;
			param[i].input_param.shape_ = input_shape;
			layers_[i].reset(new caffe::InputLayer<float>(param[i]));
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "CONVOLUTION")
		{
			lm.readConvParam(param[i]);
			layers_[i].reset(new caffe::ConvolutionLayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "RELU")
		{
			param[i].size = 0;
			layers_[i].reset(new caffe::ReLULayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "POOLING")
		{
			lm.readPoolParam(param[i]);
			layers_[i].reset(new caffe::PoolingLayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "INNERPRODUCT")
		{
			lm.readInnerProductParam(param[i]);
			layers_[i].reset(new caffe::InnerProductLayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "DROPOUT")
		{
			param[i].size = 0;
			layers_[i].reset(new caffe::DropoutLayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else if (name == "SOFTMAX")
		{
			param[i].size = 0;
			layers_[i].reset(new caffe::SoftmaxLayer<float>(param[i]));
			AppendBottom(i, i - 1);
			AppendTop(i);
			layers_[i]->SetUp(bottom_vecs_[i], top_vecs_[i]);
		}
		else
			std::cout << "Wrong laeyer name\n";
	}
}

void faceRecognition::forward()
{
	for (int i = 0; i < layer_sz; i++)
	{
		layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
		//if (i > 31){
		//	std::cout << "\n--------------------------------\n";
		//	std::vector<caffe::Blob<float>*> lay = top_vecs_[i];
		//	int size = lay[0]->count();
		//	std::cout.precision(3);
		//	std::copy(lay[0]->cpu_data(), lay[0]->cpu_data() + size, std::ostream_iterator<float>(std::cout, "\t"));
		//}
	}
}

void faceRecognition::setData(std::vector<cv::Mat> &candidateFaces)
{
	if (candidateFaces.empty()) printf("Invalid input for face recognitoin!\n");
	int num = candidateFaces.size(), channels = input_shape[1], height = input_shape[2], width = input_shape[3];
	std::vector<caffe::Blob<float>*>  input_layer = top_vecs_[0];
	input_layer[0]->Reshape(num, channels, height, width);
	float *data = input_layer[0]->mutable_cpu_data();
	std::vector<std::vector<cv::Mat>> input_channels;
	WrapBatchInputLayer(input_channels, data, width, height, channels, num);
	PreprocessBatch(candidateFaces, input_channels, width, height);
}

std::vector<std::vector<float> > faceRecognition::getResult()
{
	std::vector<caffe::Blob<float>*> recognition_layer = top_vecs_.back();
	int featureLen = recognition_layer[0]->channels();
	int sampleNo = recognition_layer[0]->num();
	const float *rec_start = recognition_layer[0]->cpu_data();
	const float *rec_end = recognition_layer[0]->cpu_data() + sampleNo * featureLen;

	std::vector<std::vector<float> > predictions;
	std::vector<float> recRes(rec_start, rec_end);
	float *start = &recRes[0], *end;
	for (int k = 0; k < sampleNo;  ++k)
	{
		end = start + featureLen;
		std::vector<float> feature(start, end);
		predictions.push_back(feature);
		start = end;
	}

	return predictions;
}
/// @brief Append a new top blob to the net.
void faceRecognition::AppendTop(int layer_id)
{
	std::shared_ptr<caffe::Blob<float> > blob_pointer(new caffe::Blob<float>());
	blobs_.push_back(blob_pointer);
	top_vecs_[layer_id].push_back(blob_pointer.get());
}
/// @brief Append a new bottom blob to the net.
void faceRecognition::AppendBottom(int layer_id, int bottom_id)
{
	bottom_vecs_[layer_id].push_back(blobs_[bottom_id].get());
}
/// @brief wrap image to blob
void faceRecognition::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > &input_batch, float *input_data, int width, int height, int channels, int num){
	for (int j = 0; j < num; j++){
		std::vector<cv::Mat> input_channels;
		for (int i = 0; i < channels; ++i){
			cv::Mat channel(height, width, CV_32FC1, input_data, width*sizeof(float));
			input_channels.push_back(channel);
			input_data += width * height;
		}
		input_batch.push_back(std::vector<cv::Mat>(input_channels));
	}
}
/// @preprocess image to input layer
void faceRecognition::PreprocessBatch(const std::vector<cv::Mat> &imgs, std::vector< std::vector<cv::Mat> > &input_batch, int height, int width){
	for (int i = 0; i < imgs.size(); i++){
		cv::Mat img = imgs[i];
		std::vector<cv::Mat> *input_channels = &(input_batch[i]);


		cv::Mat sample_resized;
		if (img.cols != width || img.rows != height)
			cv::resize(img, sample_resized, cv::Size(width, height));
		else
			sample_resized = img;

		cv::Mat float_mat;
		sample_resized.convertTo(float_mat, CV_32FC3);
		float_mat -= cv::Scalar(93.5940f, 104.7624f, 129.1863f);
		cv::Mat trans_Mat = float_mat.t();
		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(float_mat, *input_channels);
	}
}