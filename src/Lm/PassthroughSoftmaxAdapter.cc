#include "PassthroughSoftmaxAdapter.hh"

namespace Lm {

void PassthroughSoftmaxAdapter::init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) {
}

Score PassthroughSoftmaxAdapter::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    return nn_out->get(output_idx);
}

}  // namespace Lm
