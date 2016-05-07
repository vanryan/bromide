#ifndef BROMIDE_LAYERS_HELPER_H
#define BROMIDE_LAYERS_HELPER_H

#include "Halide.h"

namespace Bromide {

template<class T = float>
class huffman_node {
public:
	T val;
	huffman_node* left;
	huffman_node* right;
	huffman_node() {
		val = 0;
		left = NULL;
		right = NULL;
	}
	huffman_node(T val_ = 0, huffman_node* left_ = NULL, huffman_node* right_ = NULL): val(val_), left(left_), right(right_) {}
	bool is_leaf() {return left == NULL && right == NULL;}
};

void inclusive_scan(Halide::Image<int> &im_array) {

	int array_n = im_array.width();
	for(int i = 1; i < array_n; ++i) {
		im_array(i) = im_array(i - 1) + im_array(i);
	}

}

template<class T = float>
huffman_node<T>* generate_tree(int value_levels, T base_val, T step, int cur_index = 0) {
	//suppose the distribution follows an exponential rule
	if(cur_index + 1 == value_levels) {
		huffman_node<T>* root = new huffman_node<T>(base_val + step * cur_index);
		return root;
	}
	huffman_node<T>* root = new huffman_node<T>();
	root->left = new huffman_node<T>(base_val + step * cur_index);
	root->right = generate_tree<T>(value_levels, base_val, cur_index + 1);
	return root;

}

template<class T = float>
int huffman_decoding(Halide::ImageParam &input, int input_size, huffman_node<T>* root, Halide::ImageParam &output) {

	int cur_id = 0, output_id = 0;
	while(cur_id < input_size) {
		huffman_node<T>* cur_node = root;
		T this_val = 0;
		while(!(cur_node->is_leaf())) {
			this_val = cur_node->val;
			int cur_ele = cur_id / 32;
			int cur_offset = cur_id % 32;
			int cur_bit = (input(cur_ele) << cur_offset) >> 31;
			if(cur_bit) 
				cur_node = cur_node->right;
			else
				cur_node = cur_node->left;
		}
		output(output_id) = this_val;
		output_id++;
	}
	return output_id;

}

}

#endif