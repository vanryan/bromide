#ifndef BROMIDE_DROP_LAYER_H
#define BROMIDE_DROP_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Drop_layer : public Layer {
public:
  Drop_layer(Layer last_layer, bool train=false, float p=0.5)
    : Layer(last_layer.size[0], last_layer.size[1], last_layer.size[2], last_layer.size[3], "drop_layer") {
    if (train) {
      cnnff(x, y, z, w) = Halide::select(Halide::random_float() > p, last_layer.cnnff(x, y, z, w) / (1 - p), 0);
    } else {
      cnnff(x, y, z, w) = last_layer.cnnff(x, y, z, w);
    }
  }
};

}
#endif