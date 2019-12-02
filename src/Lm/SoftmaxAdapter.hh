#ifndef _LM_SOFTMAX_ADAPTER_HH
#define _LM_SOFTMAX_ADAPTER_HH

#include <Core/Component.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/TensorMap.hh>

#include "CompressedVector.hh"

namespace Lm {

using Score = float;

class SoftmaxAdapter : public Core::Component {
public:
    using Precursor = Core::Component;

    SoftmaxAdapter(Core::Configuration const& config);
    virtual ~SoftmaxAdapter() = default;

    virtual void               init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) = 0;
    virtual Score              get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx)                                                     = 0;
    virtual std::vector<Score> get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs);

private:
};

// inline implementations

inline SoftmaxAdapter::SoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config) {
}

inline std::vector<Score> SoftmaxAdapter::get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs) {
    std::vector<Score> scores;
    scores.reserve(output_idxs.size());
    for (size_t output_idx : output_idxs) {
        scores.push_back(get_score(nn_out, output_idx));
    }
    return scores;
}

}  // namespace Lm

#endif /* _LM_SOFTMAX_ADAPTER_HH */
