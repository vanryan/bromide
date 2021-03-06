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

int perf_lenet(Halide::ImageParam input_data, Halide::ImageParam labels, int count=1) {

	NetParameter param;

	ReadNetParamsFromBinaryFile("/path/to/bromide/resources/lenet_iter_10000.caffemodel", &param);

	Layer input_layer = ::Bromide::Input_layer(input_data, 28, 28, 1, count);
	Layer label_layer = ::Bromide::Input_layer(labels, 1, 1, 1, count);

	//test_this_layer(input_layer, count, "input");


  	//first conv layer
	int conv1_filters = 20, conv1_size = 5, conv1_stride = 1;
	Halide::ImageParam kernel1(Halide::type_of<float>(), 4);
	Halide::ImageParam bias1(Halide::type_of<float>(), 1);
	caffe::BlobProto kernel1_blob = *(param.mutable_layer(1)->mutable_blobs(0));
	caffe::BlobProto bias1_blob = *(param.mutable_layer(1)->mutable_blobs(1));

	Halide::Image<float> kernel1_ = blob_to_image_2(kernel1_blob, conv1_filters, input_layer.size[2], conv1_size, conv1_size, 4);
	Halide::Image<float> bias1_ = blob_to_image_2(bias1_blob, 1, 1, 1, conv1_filters, 1);

	kernel1.set(kernel1_);
	bias1.set(bias1_);
	Layer conv1 = ::Bromide::Conv_layer(input_layer, kernel1, bias1, conv1_size, conv1_size, conv1_filters, 0, 0, conv1_stride, conv1_stride); 



 	//first pool layer
	int pool1_size = 2, pool1_stride = 2;
	Layer pool1 = ::Bromide::Pool_layer(conv1, "max", pool1_size, pool1_size, 0, 0, pool1_stride, pool1_stride);



	//second conv layer
	int conv2_filters = 50, conv2_size = 5, conv2_stride = 1;
	Halide::ImageParam kernel2(Halide::type_of<float>(), 4);
	Halide::ImageParam bias2(Halide::type_of<float>(), 1);
	caffe::BlobProto kernel2_blob = *(param.mutable_layer(3)->mutable_blobs(0));
	caffe::BlobProto bias2_blob = *(param.mutable_layer(3)->mutable_blobs(1));
	Halide::Image<float> kernel2_ = blob_to_image_2(kernel2_blob, conv2_filters, pool1.size[2], conv2_size, conv2_size, 4);
	Halide::Image<float> bias2_ = blob_to_image_2(bias2_blob, 1, 1, 1, conv2_filters, 1);
	kernel2.set(kernel2_);
	bias2.set(bias2_);
	Layer conv2 = ::Bromide::Conv_layer(pool1, kernel2, bias2, conv2_size, conv2_size, conv2_filters, 0, 0, conv2_stride, conv2_stride); 



	//second pool layer
	int pool2_size = 2, pool2_stride = 2;
	Layer pool2 = ::Bromide::Pool_layer(conv2, "max", pool2_size, pool2_size, 0, 0, pool2_stride, pool2_stride);



	//flat layer
	Layer flat1 = ::Bromide::Flatten_layer(pool2);



	//first fully-connected layer
	int full3_size = 500;
	Halide::ImageParam weight3(Halide::type_of<float>(), 2);
	Halide::ImageParam bias3(Halide::type_of<float>(), 1);
	caffe::BlobProto weight3_blob = *(param.mutable_layer(5)->mutable_blobs(0));
	caffe::BlobProto bias3_blob = *(param.mutable_layer(5)->mutable_blobs(1));
	Halide::Image<float> weight3_ = blob_to_image_2(weight3_blob, 1, 1, full3_size, flat1.size[0], 2);
	Halide::Image<float> bias3_ = blob_to_image_2(bias3_blob, 1, 1, 1, full3_size, 1);
	weight3.set(weight3_);
	bias3.set(bias3_);
	Layer full3 = ::Bromide::Full_layer(flat1, weight3, bias3, full3_size);

	//first relu layer
	Layer relu1 = ::Bromide::ReLU_layer(full3);



	//second fully-connected layer
	int full4_size = 10;
	Halide::ImageParam weight4(Halide::type_of<float>(), 2);
	Halide::ImageParam bias4(Halide::type_of<float>(), 1);
	caffe::BlobProto weight4_blob = *(param.mutable_layer(7)->mutable_blobs(0));
	caffe::BlobProto bias4_blob = *(param.mutable_layer(7)->mutable_blobs(1));
	Halide::Image<float> weight4_ = blob_to_image_2(weight4_blob, 1, 1, full4_size, relu1.size[0], 2);
	Halide::Image<float> bias4_ = blob_to_image_2(bias4_blob, 1, 1, 1, full4_size, 1);
	weight4.set(weight4_);
	bias4.set(bias4_);
	Layer full4 = ::Bromide::Full_layer(relu1, weight4, bias4, full4_size);

	
	//soft layer
	Layer soft = ::Bromide::Soft_layer(full4);
	


	Halide::Image<float> output_soft(soft.size[0], soft.size[1], soft.size[2], soft.size[3]);
  	soft.cnnff.realize(output_soft);

	// timer
  	double startTime = CycleTimer::currentSeconds();

  	for (int i = 0; i < 1; i++)
		soft.cnnff.realize(output_soft);

	// timer
  	double endTime = CycleTimer::currentSeconds();
  	cout << (endTime - startTime) <<endl;

  	return 0;

}

void test_lenet() {

	std::vector<std::string> input_images;
	input_images.push_back("/home/rding/Bromide/bromide/resources/1.png");

  	int count = 4;
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

  	Halide::Image<float> test_correct = perf_lenet(input_data, input_labels, count);
}
