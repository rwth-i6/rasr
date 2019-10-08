#ifndef _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH
#define _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH

#include "SoftmaxAdapter.hh"

namespace Lm {

class PassthroughSoftmaxAdapter : public SoftmaxAdapter {
public:
    using Precursor = SoftmaxAdapter;

    PassthroughSoftmaxAdapter(Core::Configuration const& config);
    virtual ~PassthroughSoftmaxAdapter() = default;

    virtual void init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);
private:
};

// inline implementations

inline PassthroughSoftmaxAdapter::PassthroughSoftmaxAdapter(Core::Configuration const& config) : Precursor(config) {
}

}  // namespace Lm

#endif /* _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH */
