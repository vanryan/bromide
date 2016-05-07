#ifndef BROMIDE_LAYERS_ACTIVATION_H
#define BROMIDE_LAYERS_ACTIVATION_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {
/*
  Classes for activation layers
*/
class Sigmoid_layer : public Layer {
public:
  Sigmoid_layer(Layer input) : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = 1 / (1 + Halide::fast_exp(-input.cnnff(x, y, z, w)));
  }
};

class Abs_layer : public Layer {
public:
  Abs_layer(Layer input) : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = Halide::abs(input.cnnff(x, y, z, w));
  }
};

class Log_layer : public Layer {
public: 
  Log_layer(Layer input) : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = Halide::fast_log(input.cnnff(x, y, z, w));
}
};

class Pow_layer : public Layer {
public:
  Pow_layer(Layer input, float power = 2.0f)
    : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = Halide::fast_pow(input.cnnff(x, y, z, w), power);
  }
};

class ReLU_layer : public Layer { // used http://caffe.berkeleyvision.org/tutorial/layers.html
public:
  ReLU_layer(Layer input, float negative_slope = 0.0f) : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = Halide::max(input.cnnff(x, y, z, w), 0) + negative_slope * Halide::min(input.cnnff(x, y, z, w), 0);
  }
};

class Tanh_layer : public Layer {
public:
  Tanh_layer(Layer input) : Layer(input.size[0], input.size[1], input.size[2], input.size[3]) {
    cnnff(x, y, z, w) = Halide::tanh(input.cnnff(x, y, z, w));
}
};

} // end of namespace Bromide
#endif