#ifndef _LM_NCE_SOFTMAX_ADAPTER_HH
#define _LM_NCE_SOFTMAX_ADAPTER_HH

#include "SoftmaxAdapter.hh"

namespace Lm {

class NceSoftmaxAdapter : public SoftmaxAdapter {
public:
    using Precursor = SoftmaxAdapter;

    NceSoftmaxAdapter(Core::Configuration const& config);
    virtual ~NceSoftmaxAdapter() = default;

    virtual void init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);
    virtual std::vector<Score> get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs);
private:
    Tensorflow::Session*               session_;
    Tensorflow::TensorInputMap const*  input_map_;
    Tensorflow::TensorOutputMap const* output_map_;
};

// inline implementations

inline NceSoftmaxAdapter::NceSoftmaxAdapter(Core::Configuration const& config) : Precursor(config) {
}

}  // namespace Lm

#endif /* _LM_NCE_SOFTMAX_ADAPTER_HH */

