#include "OnnxPassthroughSoftmaxAdapter.hh"

namespace Lm {

void OnnxPassthroughSoftmaxAdapter::init(Onnx::Session& session, Onnx::IOMapping& mapping) {
}

Score OnnxPassthroughSoftmaxAdapter::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    return nn_out->get(output_idx);
}

}  // namespace Lm
