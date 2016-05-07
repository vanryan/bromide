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
	/*
	//show param
	cout << "conv layer name: " << param.mutable_layer(1)->name() << endl;
	cout << "conv layer type: " << param.mutable_layer(1)->type() << endl;
	cout << "conv layer bottom size: " << param.mutable_layer(1)->bottom_size() << endl;
	cout << "conv layer top size: " << param.mutable_layer(1)->top_size() << endl;
	cout << "conv layer blobs size: " << param.mutable_layer(1)->blobs_size() << endl;
	cout << "conv layer param size: " << param.mutable_layer(1)->param_size() << endl;
	cout << "conv layer kernel size: " << param.mutable_layer(1)->mutable_blobs(0)->data_size() << endl;
	cout << "conv layer bias size: " << param.mutable_layer(1)->mutable_blobs(1)->data_size() << endl;

	for(int i = 0; i < param.mutable_layer(1)->mutable_blobs(0)->data_size(); ++i)
		cout << "conv layer kernel data " << i << ": " << param.mutable_layer(1)->mutable_blobs(0)->data(i) << endl;
	for(int i = 0; i < param.mutable_layer(1)->mutable_blobs(1)->data_size(); ++i)
		cout << "conv layer bias data " << i << ": " << param.mutable_layer(1)->mutable_blobs(1)->data(i) << endl;
	*//*
	for(int i = 0; i < param.mutable_layer(5)->mutable_blobs(0)->data_size(); ++i)
		cout << "fully-connected layer kernel data " << i << ": " << param.mutable_layer(5)->mutable_blobs(0)->data(i) << endl;
	for(int i = 0; i < param.mutable_layer(5)->mutable_blobs(1)->data_size(); ++i)
		cout << "fully-connected layer bias data " << i << ": " << param.mutable_layer(5)->mutable_blobs(1)->data(i) << endl;
	*/

	Layer input_layer = ::Bromide::Input_layer(input_data, 28, 28, 1, count);
	Layer label_layer = ::Bromide::Input_layer(labels, 1, 1, 1, count);

	//test_this_layer(input_layer, count, "input");

	//Halide::Image<float> output_buffer(input_layer.size[0], input_layer.size[1], input_layer.size[2], input_layer.size[3]);
  	//input_layer.cnnff.realize(output_buffer);

  	//first conv layer
	int conv1_filters = 20, conv1_size = 5, conv1_stride = 1;
	Halide::ImageParam kernel1(Halide::type_of<float>(), 4);
	Halide::ImageParam bias1(Halide::type_of<float>(), 1);
	//caffe::BlobProto debug_blob1 = param.mutable_layer(1).blobs(0);
	//Halide::Image<float> kernel1_ = blob_to_image(debug_blob1);
	//cout << param.mutable_layer(1)->mutable_blobs(0)->has_num() << param.mutable_layer(1)->mutable_blobs(0)->has_channels() << param.mutable_layer(1)->mutable_blobs(0)->has_height() << param.mutable_layer(1)->mutable_blobs(0)->has_width() << endl;
	//cout << param.mutable_layer(1)->mutable_blobs(0)->num() << param.mutable_layer(1)->mutable_blobs(0)->channels() << param.mutable_layer(1)->mutable_blobs(0)->height() << param.mutable_layer(1)->mutable_blobs(0)->width() << endl;
	caffe::BlobProto kernel1_blob = *(param.mutable_layer(1)->mutable_blobs(0));
	caffe::BlobProto bias1_blob = *(param.mutable_layer(1)->mutable_blobs(1));
	/*
	Halide::Image<float> kernel1_ = blob_to_image(kernel1_blob);
	Halide::Image<float> bias1_ = blob_to_image(bias1_blob);
	*/
	Halide::Image<float> kernel1_ = blob_to_image_2(kernel1_blob, conv1_filters, input_layer.size[2], conv1_size, conv1_size, 4);
	Halide::Image<float> bias1_ = blob_to_image_2(bias1_blob, 1, 1, 1, conv1_filters, 1);
	//cout << kernel1_.extent(0) << kernel1_.extent(1) << kernel1_.extent(2) << kernel1_.extent(3) << endl;
	/*
	for(int i = 0; i < 5; ++i) {
		for(int j = 0; j < 5; ++j) {
			for(int k = 0; k < 1; ++k) {
				for(int l = 0; l < 20; ++l) {
					printf("The %d %d %d %d -th element: ", i, j, k, l);
					cout << *((float*)kernel1_.address_of(i, j, k, l)) << endl;
				}
			}
		}
	}*/
	//Halide::Image<float> bias1_(conv1_filters);
	kernel1.set(kernel1_);
	bias1.set(bias1_);
	Layer conv1 = ::Bromide::Conv_layer(input_layer, kernel1, bias1, conv1_size, conv1_size, conv1_filters, 0, 0, conv1_stride, conv1_stride); 

	//test_this_layer(conv1, count, "conv1");

 	//first pool layer
	int pool1_size = 2, pool1_stride = 2;
	Layer pool1 = ::Bromide::Pool_layer(conv1, "max", pool1_size, pool1_size, 0, 0, pool1_stride, pool1_stride);

	//test_this_layer(pool1, count, "pool1");

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

	//test_this_layer(conv2, count, "conv2");

	//second pool layer
	int pool2_size = 2, pool2_stride = 2;
	Layer pool2 = ::Bromide::Pool_layer(conv2, "max", pool2_size, pool2_size, 0, 0, pool2_stride, pool2_stride);

	//test_this_layer(pool2, count, "pool2");

	//flat layer
	Layer flat1 = ::Bromide::Flatten_layer(pool2);

	//test_this_layer(flat1, count, "flat1");

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

	//test_this_layer(full3, count, "full3");
	/*
	for(int j = 0; j < weight3_.height(); j++) {
		for(int i = 0; i < weight3_.width(); i++) {
			LOG(INFO) << i << "-" << j << "-" << weight3_(i, j) << " ";
		}
		LOG(INFO) << endl;
	}
	*/
	//first relu layer
	Layer relu1 = ::Bromide::ReLU_layer(full3);

	//test_this_layer(relu1, count, "relu1");

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

	//print_this_image(weight4_, 1, "full4_weights");
	//print_this_image(bias4_, 1, "full4_bias");
	//test_this_layer(full4, count, "full4");
	
	//soft layer
	Layer soft = ::Bromide::Soft_layer(full4);
	
	//test_this_layer(soft, count, "soft");

	Halide::Image<float> output_soft(soft.size[0], soft.size[1], soft.size[2], soft.size[3]);
  	soft.cnnff.realize(output_soft);
	/*
	for(int i = 0; i < output_soft.width(); ++i)
		cout << i << ": " << output_soft(i) << endl;

	double max_val = -1e20;
	int max_id = -1;
	for(int i = 0; i < output_soft.width(); ++i) {
		if(max_val < output_soft(i)) {
			max_val = output_soft(i);
			max_id = i;
		}
	}
	*/

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
/*
	std::string mean_proto = "/home/rding/Bromide/bromide/resources/lenet_mean.binaryproto";
	BlobProto mean_blob;
	ReadProtoFromBinaryFile(mean_proto, &mean_blob);

	int count = input_images.size();
	Halide::Image<float> mean_image = blob_to_image(mean_blob);
	Halide::ImageParam input_data(Halide::type_of<float>(), 4);
	Halide::ImageParam input_labels(Halide::type_of<int>(), 4);
	Halide::Image<float> input_data_(28, 28, 1, count);

	for (int l = 0; l < count; l++) {
	    Halide::Image<float> this_image;
	    Halide::Tools::load_png<Halide::Image<float>>(input_images[l], &this_image);
	    //Halide::Image<float> this_image = load_ppm<float> (input_images[l]);
	    for (int k = 0; k < input_data_.channels(); k++) {
			for (int j = 0; j < input_data_.height(); j++) {
				for (int i = 0; i < input_data_.width(); i++) {
					input_data_(i, j, k, l) = this_image(i, j, k) - mean_image(i, j, k) / 256;
					//printf("image %d, channel %d, row %d, column %d: this_image %f, mean_image %f\n", l, k, j, i, this_image(i, j, k), mean_image(i, j, k));
					//input_data_(i, j, k, l) = this_image(i, j, k);
				}
			}
	    }
  	}

  	input_data.set(input_data_);
 */
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

  	//TO DO input_labels]

  	//Layer final_layer = build_network(input_data, input_labels, count);

  	//Halide::Image<float> output = forward_network(final_layer, true, count);

  	Halide::Image<float> test_correct = perf_lenet(input_data, input_labels, count);
}
