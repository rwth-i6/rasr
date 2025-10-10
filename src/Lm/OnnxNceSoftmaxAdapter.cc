#include "OnnxNceSoftmaxAdapter.hh"

#include <Math/Blas.hh>

#include "DummyCompressedVectorFactory.hh"

namespace Lm {

const Core::ParameterString OnnxNceSoftmaxAdapter::paramWeightsFile("weights-file", "output embedding file", "");
const Core::ParameterString OnnxNceSoftmaxAdapter::paramBiasFile("bias-file", "output bias file", "");

void OnnxNceSoftmaxAdapter::init(Onnx::Session& session, Onnx::IOMapping& mapping) {
    Core::BinaryInputStream weight_stream(weightsFile_);
    weights_.read(weight_stream);

    Core::BinaryInputStream bias_stream(biasFile_);

    u32 numRows;
    bias_stream >> numRows;

    std::vector<float> elem(numRows);
    bias_stream.read<float>(elem.data(), numRows);

    bias_.resize(numRows, 0.0f, true);

    for (size_t i = 0; i < numRows; i++) {
        bias_[i] = elem[i];
    }
}

Score OnnxNceSoftmaxAdapter::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    std::vector<float>                   nn_output;
    float const*                         data;
    Lm::UncompressedVector<float> const* vec = dynamic_cast<Lm::UncompressedVector<float> const*>(nn_out.get());

    if (vec != nullptr) {
        data = vec->data();
    }
    else {
        nn_output.resize(nn_out->size());
        nn_out->uncompress(nn_output.data(), nn_output.size());
        data = nn_output.data();
    }

    float result = Math::dot(nn_out->size(), data, 1, &weights_(0, output_idx), 1);
    result += bias_[output_idx];

    return result;
}

std::vector<Score> OnnxNceSoftmaxAdapter::get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs) {
    std::vector<float> nn_output(nn_out->size());
    nn_out->uncompress(nn_output.data(), nn_output.size());

    std::vector<Score> result(output_idxs.size());
    for (size_t i = 0ul; i < output_idxs.size(); i++) {
        result[i] = Math::dot(nn_output.size(), nn_output.data(), 1, &weights_(0, output_idxs[i]), 1);
        result[i] += bias_[output_idxs[i]];
    }

    return result;
}

}  // namespace Lm
