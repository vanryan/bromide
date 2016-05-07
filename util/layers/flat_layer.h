#ifndef BROMIDE_LAYERS_FLATTEN_H
#define BROMIDE_LAYERS_FLATTEN_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Flatten_layer : public Layer {
public:
	Flatten_layer(Layer last_layer) : Layer(last_layer.size[0] * last_layer.size[1] * last_layer.size[2], 1, 1, last_layer.size[3], "flatten_layer") {
	  cnnff(x, y, z, w) = last_layer.cnnff(x % last_layer.size[0], (x / last_layer.size[0]) % last_layer.size[1], x / (last_layer.size[0] * last_layer.size[1]), w);
	}
};

}

#endif