#ifndef BROMIDE_INPUT_LAYER_H
#define BROMIDE_INPUT_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Input_layer : public Layer{
public:
	Input_layer(Halide::ImageParam input_image, int size1, int size2, int size3, int size4): Layer(size1, size2, size3, size4) {

		cnnff(x, y, z, w) = input_image(x, y, z, w);
		
	}

};

}

#endif