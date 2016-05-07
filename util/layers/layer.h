#ifndef BROMIDE_LAYER_H
#define BROMIDE_LAYER_H

#include <string.h>
#include "Halide.h"

namespace Bromide {

int do_schedule = 1;
bool use_gpu = false;

class Layer {
public:
	Layer(std::string s = "this_layer"): cnnff(s), x("x"), y("y"), z("z"), w("w") {size[0] = 0; size[1] = 0; size[2] = 0; size[3] = 0;}
	Layer(int size1, int size2, int size3, int size4, std::string s = "this_layer"): cnnff(s), x("x"), y("y"), z("z"), w("w") {size[0] = size1; size[1] = size2; size[2] = size3; size[3] = size4;}
	Halide::Func cnnff;

public:
	Halide::Var x, y, z, w;
	int size[4];
};

}

#endif