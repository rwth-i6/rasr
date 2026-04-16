#ifndef _LM_ONNX_SOFTMAX_ADAPTER_HH
#define _LM_ONNX_SOFTMAX_ADAPTER_HH

#include <Core/Component.hh>
#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>

#include "CompressedVector.hh"

namespace Lm {

using Score = float;

class OnnxSoftmaxAdapter : public Core::Component {
public:
    using Precursor = Core::Component;

    OnnxSoftmaxAdapter(Core::Configuration const& config);
    virtual ~OnnxSoftmaxAdapter() = default;

    virtual void               init(Onnx::Session& session, Onnx::IOMapping& mapping)                     = 0;
    virtual Score              get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) = 0;
    virtual std::vector<Score> get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs);

private:
};

// inline implementations

inline OnnxSoftmaxAdapter::OnnxSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config) {
}

inline std::vector<Score> OnnxSoftmaxAdapter::get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs) {
    std::vector<Score> scores;
    scores.reserve(output_idxs.size());
    for (size_t output_idx : output_idxs) {
        scores.push_back(get_score(nn_out, output_idx));
    }
    return scores;
}

}  // namespace Lm

#endif /* _LM_ONNX_SOFTMAX_ADAPTER_HH */
