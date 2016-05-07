#ifndef BROMIDE_POOL_LAYER_H
#define BROMIDE_POOL_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Pool_layer : public Layer{
public:
	Pool_layer(Layer last_layer, std::string pool_func, int pool_x, int pool_y, int pad_x=0, int pad_y=0, int stride_x=1, int stride_y=1):
		Layer((last_layer.size[0] + 2 * pad_x - pool_x) / stride_x + 1, (last_layer.size[1] + 2 * pad_y - pool_y) / stride_y + 1,
            last_layer.size[2], last_layer.size[3], "pooling_layer") {

		//Halide::Func pad_maps("pad_maps");
		//pad_maps = Halide::BoundaryConditions::constant_exterior(last_layer.cnnff, 0.0f, 0, last_layer.size[0], 0, last_layer.size[1]);

		Halide::Func sub_maps("sub_maps");
		Halide::RDom dom(0, pool_x, 0, pool_y);
		if(pool_func == "max") {
			//sub_maps(x, y, z, w) = Halide::maximum(pad_maps(x + dom.x, y + dom.y, z, w));
			sub_maps(x, y, z, w) = Halide::maximum(last_layer.cnnff(x + dom.x, y + dom.y, z, w));
		}else if(pool_func == "average") {
			//sub_maps(x, y, z, w) = Halide::sum(pad_maps(x + dom.x, y + dom.y, z, w)) / (pool_x * pool_y);
			sub_maps(x, y, z, w) = Halide::sum(last_layer.cnnff(x + dom.x, y + dom.y, z, w)) / (pool_x * pool_y);
		}

		cnnff(x, y, z, w) = sub_maps(x * stride_x - pad_x, y * stride_y - pad_y, z, w);
		/*
		 * vectorize 
		 */
        cnnff.vectorize(x, 8);

		cnnff.compute_root();
	}

};

}

#endif