#ifndef BROMIDE_NORM_LAYER_H
#define BROMIDE_NORM_LAYER_H

#include "Halide.h"
#include "layer.h"

namespace Bromide {

class Norm_layer : public Layer {
public:
  Norm_layer(Layer last_layer, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f)
    : Layer(last_layer.size[0], last_layer.size[1], last_layer.size[2], last_layer.size[3], "norm_layer") {

    Halide::Func clamped = Halide::BoundaryConditions::constant_exterior(last_layer.cnnff, 0.0f, 0, last_layer.size[0], 0, last_layer.size[1], 0, last_layer.size[2]);
    Halide::Func activation("activation");
    Halide::Func normalizer("normalizer");
    Halide::RDom r(-region_x / 2, region_x / 2 + 1, -region_y / 2, region_y / 2 + 1, -region_z / 2, region_z / 2 + 1);

    Halide::Expr val = clamped(x + r.x, y + r.y, z + r.z, w);

    activation(x, y, z, w) = Halide::sum(val * val);
    normalizer(x, y, z, w) = Halide::fast_pow(1.0f + (alpha / (region_x * region_y * region_z)) * activation(x, y, z, w), beta);
    cnnff(x, y, z, w) = clamped(x, y, z, w) / normalizer(x, y, z, w);

    cnnff.compute_root();
  }
};

}
#endif