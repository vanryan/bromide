#ifndef BROMIDE_LOSS_LAYER_H
#define BROMIDE_LOSS_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Soft_layer : public Layer {
public:
	Soft_layer(Layer last_layer): Layer(last_layer.size[0], last_layer.size[1], last_layer.size[2], last_layer.size[3], "soft_layer") {

	    Halide::Func soft_exp("soft_exp");
	    Halide::Func soft_norm("soft_norm");
	    Halide::RDom r(0, last_layer.size[0]);

	    soft_exp(x, y, z, w) = Halide::fast_exp(last_layer.cnnff(x, y, z, w));

	    soft_norm(y, z, w) = Halide::sum(soft_exp(r.x, y, z, w));
	    cnnff(x, y, z, w) = soft_exp(x, y, z, w) / soft_norm(y, z, w);

	}
};

class Accu_layer : public Layer {
public:
	Accu_layer(Layer last_layer, Layer labels) : Layer(1, 1, 1, 1) {
		Halide::Func correct;
		Halide::RDom dom_x(0, last_layer.size[0]);
		Halide::RDom dom_w(0, last_layer.size[3]);

		correct(w) = Halide::argmax(dom_x, last_layer.cnnff(dom_x.x, 0, 0, w))[0] == labels.cnnff(0, 0, 0, w); 
		cnnff(x, y, z, w) = Halide::sum(Halide::select(correct(dom_w.x), 1, 0)) / last_layer.size[3]; 
	}
};

class Logi_loss_layer : public Layer {
public:
	Logi_loss_layer(Layer last_layer, Layer labels) : Layer(1, 1, 1, 1) {


		Halide::RDom dom(0, last_layer.w);

		cnnff(x, y, z, w) = -Halide::sum(Halide::fast_log(last_layer.cnnff(labels.cnnff(0, 0, 0, dom.x), 0, 0, dom.x))) / last_layer.w;
	}
};

}

#endif