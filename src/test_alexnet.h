#include "caffe.pb.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <glog/logging.h> 
#include <vector>
#include <string>

#include "Halide.h"

#include "bromide.h"

#include "CycleTimer.h"



using namespace std;
using namespace caffe;
using namespace Bromide;

int perf_alexnet(Halide::ImageParam input_data, Halide::ImageParam labels, int count=1) {

	NetParameter param;
  	ReadNetParamsFromBinaryFile("/path/to/bromide/resources/bvlc_alexnet.caffemodel", &param);

  	// timer
  	// double startTime = CycleTimer::currentSeconds();

  	Layer input_layer = ::Bromide::Input_layer(input_data, 227, 227, 3, count); //1
  	Layer label_layer = ::Bromide::Input_layer(labels, 1, 1, 1, count); //2

  	int conv1_filters = 96, conv1_size = 11, conv1_stride = 4;
	Halide::ImageParam kernel1(Halide::type_of<float>(), 4);
	Halide::ImageParam bias1(Halide::type_of<float>(), 1);
	Halide::Image<float> kernel1_ = blob_to_image(param.layers(1).blobs(0));
	//printf("kernel1_ has width %d, height %d, channel %d, num %d\n", kernel1_.width(), kernel1_.height(), kernel1_.channels(), conv1_filters);
	Halide::Image<float> bias1_ = blob_to_image(param.layers(1).blobs(1));
	kernel1.set(kernel1_);
	bias1.set(bias1_);
	Layer conv1 = ::Bromide::Conv_layer(input_layer, kernel1, bias1, conv1_size, conv1_size, conv1_filters, 0, 0, conv1_stride, conv1_stride); //3
	//printf("conv1 outputs width %d, height %d, channel %d, num %d\n", conv1.size[0], conv1.size[1], conv1.size[2], conv1.size[3]);

	Layer relu1 = ::Bromide::ReLU_layer(conv1); //4

	int norm1_size = 5;
	float norm1_alpha = 0.0001f, norm1_beta = 0.75f;
	Layer norm1 = ::Bromide::Norm_layer(relu1, 1, 1, norm1_size, norm1_alpha, norm1_beta); //6

  	int pool1_size = 3, pool1_stride = 2;
	Layer pool1 = ::Bromide::Pool_layer(norm1, "max", pool1_size, pool1_size, 0, 0, pool1_stride, pool1_stride); //5
	//printf("pool1 outputs width %d, height %d, channel %d, num %d\n", pool1.size[0], pool1.size[1], pool1.size[2], pool1.size[3]);

	int conv2_filters = 256, conv2_pad = 2, conv2_size = 5;
	Halide::ImageParam kernel2(Halide::type_of<float>(), 4);
	Halide::ImageParam bias2(Halide::type_of<float>(), 1);
	Halide::Image<float> kernel2_ = blob_to_image(param.layers(5).blobs(0));
	//printf("kernel2_ has width %d, height %d, channel %d, num %d\n", kernel2_.width(), kernel2_.height(), kernel2_.channels(), conv2_filters);
	Halide::Image<float> bias2_ = blob_to_image(param.layers(5).blobs(1));
	kernel2.set(kernel2_);
	bias2.set(bias2_);
	Layer conv2 = ::Bromide::Conv_layer(pool1, kernel2, bias2, conv2_size, conv2_size, conv2_filters, conv2_pad, conv2_pad, 1, 1, 2); //7
	//printf("conv2 outputs width %d, height %d, channel %d, num %d\n", conv2.size[0], conv2.size[1], conv2.size[2], conv2.size[3]);

	Layer relu2 = ::Bromide::ReLU_layer(conv2); //8

	int norm2_size = 5;
	float norm2_alpha = 0.0001f, norm2_beta = 0.75f;
	Layer norm2 = ::Bromide::Norm_layer(relu2, 1, 1, norm2_size, norm2_alpha, norm2_beta); //10

	int pool2_size = 3, pool2_stride = 2;
	Layer pool2 = ::Bromide::Pool_layer(norm2, "max", pool2_size, pool2_size, 0, 0, pool2_stride, pool2_stride); //9
	//printf("pool2 outputs width %d, height %d, channel %d, num %d\n", pool2.size[0], pool2.size[1], pool2.size[2], pool2.size[3]);

	int conv3_filters = 384, conv3_pad = 1, conv3_size = 3;
	Halide::ImageParam kernel3(Halide::type_of<float>(), 4);
	Halide::ImageParam bias3(Halide::type_of<float>(), 1);
	Halide::Image<float> kernel3_ = blob_to_image(param.layers(9).blobs(0));
	//printf("kernel3_ has width %d, height %d, channel %d, num %d\n", kernel3_.width(), kernel3_.height(), kernel3_.channels(), conv3_filters);
	Halide::Image<float> bias3_ = blob_to_image(param.layers(9).blobs(1));
	kernel3.set(kernel3_);
	bias3.set(bias3_);
	Layer conv3 = ::Bromide::Conv_layer(pool2, kernel3, bias3, conv3_size, conv3_size, conv3_filters, conv3_pad, conv3_pad, 1, 1); //11
	//printf("conv3 outputs width %d, height %d, channel %d, num %d\n", conv3.size[0], conv3.size[1], conv3.size[2], conv3.size[3]);

	Layer relu3 = ::Bromide::ReLU_layer(conv3); //12

	int conv4_filters = 384, conv4_pad = 1, conv4_size = 3;
	Halide::ImageParam kernel4(Halide::type_of<float>(), 4);
	Halide::ImageParam bias4(Halide::type_of<float>(), 1);
	Halide::Image<float> kernel4_ = blob_to_image(param.layers(11).blobs(0));
	//printf("kernel4_ has width %d, height %d, channel %d, num %d\n", kernel4_.width(), kernel4_.height(), kernel4_.channels(), conv4_filters);
	Halide::Image<float> bias4_ = blob_to_image(param.layers(11).blobs(1));
	kernel4.set(kernel4_);
	bias4.set(bias4_);
	Layer conv4 = ::Bromide::Conv_layer(relu3, kernel4, bias4, conv4_size, conv4_size, conv4_filters, conv4_pad, conv4_pad, 1, 1, 2); //13
	//printf("conv4 outputs width %d, height %d, channel %d, num %d\n", conv4.size[0], conv4.size[1], conv4.size[2], conv4.size[3]);

	Layer relu4 = ::Bromide::ReLU_layer(conv4); //14

	int conv5_filters = 256, conv5_pad = 1, conv5_size = 3;
	Halide::ImageParam kernel5(Halide::type_of<float>(), 4);
	Halide::ImageParam bias5(Halide::type_of<float>(), 1);
	Halide::Image<float> kernel5_ = blob_to_image(param.layers(13).blobs(0));
	//printf("kernel5_ has width %d, height %d, channel %d, num %d\n", kernel5_.width(), kernel5_.height(), kernel5_.channels(), conv5_filters);
	Halide::Image<float> bias5_ = blob_to_image(param.layers(13).blobs(1));
	kernel5.set(kernel5_);
	bias5.set(bias5_);
	Layer conv5 = ::Bromide::Conv_layer(relu4, kernel5, bias5, conv5_size, conv5_size, conv5_filters, conv5_pad, conv5_pad, 1, 1, 2); //15
	//printf("conv5 outputs width %d, height %d, channel %d, num %d\n", conv5.size[0], conv5.size[1], conv5.size[2], conv5.size[3]);

	Layer relu5 = ::Bromide::ReLU_layer(conv5); //16

	int pool5_size = 3, pool5_stride = 2;
	Layer pool5 = ::Bromide::Pool_layer(relu5, "max", pool5_size, pool5_size, 0, 0, pool5_stride, pool5_stride); //17
	//printf("pool5 outputs width %d, height %d, channel %d, num %d\n", pool5.size[0], pool5.size[1], pool5.size[2], pool5.size[3]);

	// TODO: revisit InnerProduct to implicitly flatten
	Layer flatten5 = ::Bromide::Flatten_layer(pool5); //18
	//printf("flatten5 outputs width %d, height %d, channel %d, num %d\n", flatten5.size[0], flatten5.size[1], flatten5.size[2], flatten5.size[3]);

	int fc6_size = 4096;
	Halide::ImageParam W6(Halide::type_of<float>(), 2);
	Halide::ImageParam bias6(Halide::type_of<float>(), 1);
	//Halide::Image<float> W6_ = transpose(blob_to_image(param.layers(16).blobs(0)));
	Halide::Image<float> W6_ = blob_to_image(param.layers(16).blobs(0));
	//printf("W6_ has width %d, height %d, channel %d, num %d\n", W6_.width(), W6_.height(), W6_.channels(), 1);
	Halide::Image<float> bias6_ = blob_to_image(param.layers(16).blobs(1));
	W6.set(W6_);
	bias6.set(bias6_);
	Layer fc6 = ::Bromide::Full_layer(flatten5, W6, bias6, fc6_size); //19
	//printf("fc6 outputs width %d, height %d, channel %d, num %d\n", fc6.size[0], fc6.size[1], fc6.size[2], fc6.size[3]);

	Layer relu6 = ::Bromide::ReLU_layer(fc6); //20

	Layer drop6 = ::Bromide::Drop_layer(relu6); //21

	int fc7_size = 4096;
	Halide::ImageParam W7(Halide::type_of<float>(), 2);
	Halide::ImageParam bias7(Halide::type_of<float>(), 1);
	Halide::Image<float> W7_ = transpose(blob_to_image(param.layers(19).blobs(0)));
	//printf("W7_ has width %d, height %d, channel %d, num %d\n", W7_.width(), W7_.height(), W7_.channels(), 1);
	Halide::Image<float> bias7_ = blob_to_image(param.layers(19).blobs(1));
	W7.set(W7_);
	bias7.set(bias7_);
	Layer fc7 = ::Bromide::Full_layer(drop6, W7, bias7, fc7_size); //22
	//printf("fc7 outputs width %d, height %d, channel %d, num %d\n", fc7.size[0], fc7.size[1], fc7.size[2], fc7.size[3]);

	Layer relu7 = ::Bromide::ReLU_layer(fc7); //23

	Layer drop7 = ::Bromide::Drop_layer(relu7); //24

	int fc8_size = 1000;
	Halide::ImageParam W8(Halide::type_of<float>(), 2);
	Halide::ImageParam bias8(Halide::type_of<float>(), 1);
	//Halide::Image<float> W8_ = transpose(blob_to_image(param.layers(22).blobs(0)));
	Halide::Image<float> W8_ = blob_to_image(param.layers(22).blobs(0));
	Halide::Image<float> bias8_ = blob_to_image(param.layers(22).blobs(1));
	W8.set(W8_);
	//printf("W8_ has width %d, height %d, channel %d, num %d\n", W8_.width(), W8_.height(), W8_.channels(), 1);
	bias8.set(bias8_);
	Layer fc8 = ::Bromide::Full_layer(drop7, W8, bias8, fc8_size); //25
	//printf("fc8 outputs width %d, height %d, channel %d, num %d\n", fc8.size[0], fc8.size[1], fc8.size[2], fc8.size[3]);

	Layer soft = ::Bromide::Soft_layer(fc8); //27

	Halide::Image<float> my_soft_output(soft.size[0], soft.size[1], soft.size[2], soft.size[3]);

	if(use_gpu) {
		Halide::Target target = Halide::get_host_target();

		target.set_feature(Halide::Target::CUDA);

		soft.cnnff.compile_jit(target);
	}
	
	//double endTime0 = CycleTimer::currentSeconds();
	//cout << endTime0 - startTime <<endl;

	soft.cnnff.realize(my_soft_output);

	// timer
  	double startTime = CycleTimer::currentSeconds();

  	for (int i = 0; i < 1; i++)
		soft.cnnff.realize(my_soft_output);

	// timer
  	double endTime = CycleTimer::currentSeconds();
  	cout << (endTime - startTime) <<endl;

  	return 0;

}

void test_alexnet() {

	std::vector<std::string> input_images;
	input_images.push_back("/path/to/bromide/resources/cat.ppm");

	std::string mean_proto = "/path/to/bromide/resources/imagenet_mean.binaryproto";
	BlobProto mean_blob;
	ReadProtoFromBinaryFile(mean_proto, &mean_blob);
/*
	int count = input_images.size();
	Halide::Image<float> mean_image = blob_to_image(mean_blob);
	Halide::ImageParam input_data(Halide::type_of<float>(), 4);
	Halide::ImageParam input_labels(Halide::type_of<int>(), 4);
	Halide::Image<float> input_data_(227, 227, 3, count);

	for (int l = 0; l < count; l++) {
	    //Halide::Image<float> this_image;
	    //Halide::Tools::load_png<Halide::Image<float>>(input_images[l], &this_image);
	    Halide::Image<float> this_image = load_ppm<float> (input_images[l]);
	    for (int k = 0; k < input_data_.channels(); k++) {
			for (int j = 0; j < input_data_.height(); j++) {
				for (int i = 0; i < input_data_.width(); i++) {
					//input_data_(i, j, k, l) = this_image(i, j, k) - mean_image(i, j, k) / 256;
					////printf("image %d, channel %d, row %d, column %d: this_image %f, mean_image %f\n", l, k, j, i, this_image(i, j, k), mean_image(i, j, k));
					input_data_(i, j, k, l) = this_image(i, j, k);
				}
			}
	    }
  	}
*/
  	int count = 64;
  	Halide::Image<float> input_data_(227, 227, 3, count);
  	for(int l = 0; l < count; l++) {
  		for(int k = 0; k < input_data_.channels(); ++k) {
  			for(int j = 0; j < input_data_.height(); ++j) {
  				for(int i = 0; i < input_data_.width(); ++i) {
  					input_data_(i, j, k, l) = rand() * 1.0 / RAND_MAX; 
  				}
  			}
  		}
  	}
  	Halide::ImageParam input_data(Halide::type_of<float>(), 4);
  	Halide::ImageParam input_labels(Halide::type_of<int>(), 4);
  	input_data.set(input_data_);
  	//TO DO input_labels]

  	//Layer final_layer = build_network(input_data, input_labels, count);

  	//Halide::Image<float> output = forward_network(final_layer, true, count);

  	Halide::Image<float> test_correct = perf_alexnet(input_data, input_labels, count);
}
