#ifndef BROMIDE_CONV_LAYER_H
#define BROMIDE_CONV_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Conv_layer : public Layer{
public:
	Conv_layer(Layer last_layer, Halide::ImageParam kernels, Halide::ImageParam bias, int kernel_x, int kernel_y, 
		int kernel_num, int pad_x = 0, int pad_y = 0, int stride_x = 1, int stride_y = 1, int group_num = 1):
		Layer((last_layer.size[0] + 2 * pad_x - kernel_x) / stride_x + 1, (last_layer.size[1] + 2 * pad_y - kernel_y) / stride_y + 1,
            kernel_num, last_layer.size[3], "conv_layer") {

		int group_size = kernel_num / group_num;
		int input_group_size = last_layer.size[2] / group_num;

		Halide::RDom dom(0, kernel_x, 0, kernel_y, 0, last_layer.size[2] / group_num);
		Halide::Expr group_idx = z / group_size, map_idx = z % group_size;
		Halide::Func conv_basic("conv_basic");

		Halide::RDom r(0, kernel_x, 0, kernel_y, 0, last_layer.size[2]);
				
		if (do_schedule == 1) {
			/*
			 * update definition
			 */
			Halide::Func f_in_bound("f_in_bound");
			f_in_bound = Halide::BoundaryConditions::constant_exterior(last_layer.cnnff, 0, 0, last_layer.size[0], 0, last_layer.size[1]);

			cnnff(x, y, z, w) = bias(z);
            cnnff(x, y, z, w) += kernels(r.x, r.y, r.z % input_group_size, z) * f_in_bound(x * stride_x + r.x - pad_x, y * stride_y + r.y - pad_y, r.z, w);
			

			cnnff.compute_root();
			Halide::Var y_outer, y_inner, z_outer, z_inner;
	        int z_s = 1, y_s = 1; // lenet
	        // int z_s = 32, y_s = 32; // alexnet

			cnnff.update().split(y, y_outer, y_inner, y_s).split(z, z_outer, z_inner, z_s);
			cnnff.update().reorder(y_inner, z_inner, r.z, y_outer, z_outer); 
			cnnff.update().vectorize(x, 8);          
			cnnff.update().parallel(w);
			cnnff.update().unroll(r.x).unroll(r.y);
			f_in_bound.compute_at(cnnff, z_inner);

		} else {
			/*
			 * sum
			 */
			conv_basic(x, y, z, w) = Halide::sum(last_layer.cnnff(x + dom.x, y + dom.y, dom.z + group_idx * input_group_size, w) * kernels(dom.x, dom.y, dom.z, map_idx + group_idx * group_size) + bias(z));
			// padding
			cnnff(x, y, z, w) = conv_basic(x * stride_x - pad_x, y * stride_y - pad_y, z, w);

		}
		

		cnnff.compute_root();
		//cnnff.print_loop_nest();

	}

	/* trying using matrix multiplication to realize Convolution */
	/*
	Conv_layer(Layer last_layer Halide::ImageParam kernels, Halide::ImageParam bias, int kernel_x, int kernel_y, 
		int kernel_num, int pad_x = 0, int pad_y = 0, int stride_x = 1, int stride_y = 1):
		Layer(last_layer.size[0], last_layer.size[1], last_layer.size[2], last_layer.size[3]) {

		Halide::RDom dom(0, kernel_x, 0, kernel_y, 0, last_layer.size[3]);
		Halide::Func conv_basic("conv_basic");

		conv_basic(x, y, z, w) = Halide::sum(last_layer.cnnff(x + dom.x, y + dom.y, dom.z, w) * kernels(dom.x, dom.y, dom.z, z) + bias(z));
		cnnff(x, y, z, w) = conv_basic(x * stride_x - pad_x, y * stride_y - pad_y, z, w);
		
	}
	*/
};

}

#endif
