#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

  void saveCaffeModel();

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};


void Classifier::saveCaffeModel()
{
    vector<shared_ptr<Layer<float> > > layers = net_->layers();
    int layer_sz = layers.size();

    std::vector<std::string> layer_names = net_->layer_names();

    std::FILE *fout = std::fopen("vgg_16_face.bin", "w");
    if(!fout) return;
    for(int i = 0; i < layer_sz - 4; ++i)
    {
        boost::shared_ptr<caffe::Layer<float> > lay = layers[i];
        std::string name = lay->type();
        std::vector<boost::shared_ptr<caffe::Blob<float> > >& blobs = lay->blobs();

        std::string layer_name = layer_names[i];

        if(name == "Convolution")
        {
            boost::shared_ptr<caffe::Blob<float> > weight = blobs[0];
            boost::shared_ptr<caffe::Blob<float> > bias = blobs[1];

            int pad = 1;
            //if(layer_name == "face-conv6")
            //   pad = 0;
            int stride = 1;
            std::fwrite(&pad, sizeof(int), 1, fout);
            std::fwrite(&stride, sizeof(int), 1, fout);

            //weight
            std::vector<int> shape = weight->shape();
            std::fwrite(&shape[0], sizeof(int), 4, fout);
            std::fwrite(weight->cpu_data(), sizeof(float), weight->count(), fout);

            //bias
            shape = bias->shape();
            std::fwrite(&shape[0], sizeof(float), 1, fout);
            std::fwrite(bias->cpu_data(), sizeof(float), bias->count(), fout);
        }
        else if(name == "Pooling")
        {
            int pad = 0;
            int stride = 2;
            int kernel_size = 2;
            std::fwrite(&pad, sizeof(int), 1, fout);
            std::fwrite(&stride, sizeof(int), 1, fout);
            std::fwrite(&kernel_size, sizeof(int), 1, fout);
        }
        else if(name == "InnerProduct")
        {
            boost::shared_ptr<caffe::Blob<float> > weight = blobs[0];
            boost::shared_ptr<caffe::Blob<float> > bias = blobs[1];

            //weight
            std::vector<int> shape = weight->shape();
            std::fwrite(&shape[0], sizeof(int), 2, fout);
            std::fwrite(weight->cpu_data(), sizeof(float), weight->count(), fout);

            //bias
            shape = bias->shape();
            std::fwrite(&shape[0], sizeof(int), 1, fout);
            std::fwrite(bias->cpu_data(), sizeof(float), bias->count(), fout);
        }
    }

    std::fclose(fout);
}

int main() {

  ::google::InitGoogleLogging("");

  string model_file   = "/home/rpk/caffe-master/models/vgg_face/VGG_FACE_deploy.prototxt";
  string trained_file = "/home/rpk/caffe-master/models/vgg_face/VGG_FACE.caffemodel";
  Classifier classifier(model_file, trained_file);
  classifier.saveCaffeModel();

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
