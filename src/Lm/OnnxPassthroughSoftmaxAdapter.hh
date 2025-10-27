#ifndef _LM_ONNX_PASSTHROUGH_SOFTMAX_ADAPTER_HH
#define _LM_ONNX_PASSTHROUGH_SOFTMAX_ADAPTER_HH

#include <Onnx/Session.hh>

#include "OnnxSoftmaxAdapter.hh"

namespace Lm {

class OnnxPassthroughSoftmaxAdapter : public OnnxSoftmaxAdapter {
public:
    using Precursor = OnnxSoftmaxAdapter;

    OnnxPassthroughSoftmaxAdapter(Core::Configuration const& config);
    virtual ~OnnxPassthroughSoftmaxAdapter() = default;

    virtual void  init(Onnx::Session& session, Onnx::IOMapping& mapping);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);

private:
};

// inline implementations

inline OnnxPassthroughSoftmaxAdapter::OnnxPassthroughSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif /* _LM_ONNX_PASSTHROUGH_SOFTMAX_ADAPTER_HH */
