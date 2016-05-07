#ifndef BROMIDE_LAYERS_FULLY_CONNECTED_H
#define BROMIDE_LAYERS_FULLY_CONNECTED_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Full_layer: public Layer {
public:
	Full_layer(Layer last_layer, Halide::ImageParam weights, Halide::ImageParam bias, int next_neuron_num) : 
		Layer(next_neuron_num, 1, 1, last_layer.size[3], "full_layer") {

		if (do_schedule == 0) {

			if (!use_gpu) {

			} else {

				Halide::RDom r(0, last_layer.size[0]);

				Halide::Func last_layer_one_dim("last_layer_one_dim");
				Halide::Func this_layer_one_dim("this_layer_one_dim");

				last_layer_one_dim(x, w) = last_layer.cnnff(x, 0, 0, w);
				this_layer_one_dim(x, w) += last_layer_one_dim(r.x, w) * weights(r.x, x);

				// tile:
				//Halide::Var x_outer, x_inner, w_outer, w_inner;
				//this_layer_one_dim.update().tile(x, w, x_outer, w_outer, x_inner, w_inner, 64, 16);
				
				//cnnff.reorder(w, x, y, z);
				//cnnff(x, y, z, w) = bias(x);
				// Dot product
				///cnnff(x, y, z, w) += last_layer.cnnff(r.x, y, z, w) * weights(r.x, x);

				Halide::Var par("par");
			    //cnnff.compute_root().fuse(x, w, par).parallel(par);   
			    //this_layer_one_dim.vectorize(w, 8); 

				this_layer_one_dim.gpu_tile(x, 256);

			    this_layer_one_dim.compute_root();
			    cnnff(x, y, z, w) = this_layer_one_dim(x, w);
			    //this_layer_one_dim.update().parallel(w, 8);
			    //this_layer_one_dim.print_loop_nest();
			                    
			    //cnnff.update().parallel(w, 8); 
			}
			

		}
		
		else { // test
			Halide::RDom dom(0, last_layer.size[0]);
			cnnff(x, y, z, w) = Halide::sum(last_layer.cnnff(dom.x, y, z, w) * weights(dom.x, x)) + bias(x); 

			cnnff.parallel(w);
			cnnff.vectorize(x, 8);
			cnnff.compute_root();

		}
	}
};

}

#endif
