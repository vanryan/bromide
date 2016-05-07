#ifndef BROMIDE_LAYERS_FULLY_CONNECTED_COMP_H
#define BROMIDE_LAYERS_FULLY_CONNECTED_COMP_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Full_layer_comp: public Layer {
public:
	Full_layer_comp(Layer last_layer, Halide::ImageParam weights_index, Halide::ImageParam weights_quant_val, Halide::ImageParam Halide::ImageParam bias, int next_neuron_num) : 
		Layer(next_neuron_num, 1, 1, last_layer.size[3]) {

		Halide::RDom dom(0, last_layer.size[0]);
		cnnff(x, y, z, w) = Halide::sum(last_layer.cnnff(dom.x, y, z, w) * weights(dom.x, x)) + bias(x);

		cnnff.compute_root();
	}
};

}

#endif