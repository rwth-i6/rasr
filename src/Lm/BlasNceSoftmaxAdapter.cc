#include "BlasNceSoftmaxAdapter.hh"

#include <Math/Blas.hh>

#include "DummyCompressedVectorFactory.hh"

namespace Lm {

void BlasNceSoftmaxAdapter::init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) {
    auto const& weight_tensor_info = output_map.get_info("weights");
    auto const& bias_tensor_info   = output_map.get_info("bias");
    session.run({}, {weight_tensor_info.tensor_name(), bias_tensor_info.tensor_name()}, {}, tensors_);
}

Score BlasNceSoftmaxAdapter::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    std::vector<float> nn_output;
    float const* data;
    Lm::UncompressedVector<float> const* vec = dynamic_cast<Lm::UncompressedVector<float> const*>(nn_out.get());

    if (vec != nullptr) {
        data = vec->data();
    }
    else {
        nn_output.resize(nn_out->size());
        nn_out->uncompress(nn_output.data(), nn_output.size());
        data = nn_output.data();
    }

    float result = Math::dot(nn_out->size(), data, 1, tensors_[0].data<float>(output_idx, 0), 1);
    result += tensors_[1].data<float>()[output_idx];

    return result;
}

std::vector<Score> BlasNceSoftmaxAdapter::get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs) {
    std::vector<float> nn_output(nn_out->size());
    nn_out->uncompress(nn_output.data(), nn_output.size());

    std::vector<Score> result(output_idxs.size());
    for (size_t i = 0ul; i < output_idxs.size(); i++) {
        result[i] = Math::dot(nn_output.size(), nn_output.data(), 1, tensors_[0].data<float>(output_idxs[i], 0), 1);
        result[i] += tensors_[1].data<float>()[output_idxs[i]];
    }

    return result;
}

}

