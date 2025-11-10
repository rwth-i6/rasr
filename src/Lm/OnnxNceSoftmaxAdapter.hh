#ifndef _LM_ONNX_NCE_SOFTMAX_ADAPTER_HH
#define _LM_ONNX_NCE_SOFTMAX_ADAPTER_HH

#include <Onnx/Session.hh>

#include "OnnxSoftmaxAdapter.hh"

namespace Lm {

class OnnxNceSoftmaxAdapter : public OnnxSoftmaxAdapter {
public:
    using Precursor = OnnxSoftmaxAdapter;

    static const Core::ParameterString paramWeightsFile;
    static const Core::ParameterString paramBiasFile;

    OnnxNceSoftmaxAdapter(Core::Configuration const& config);
    virtual ~OnnxNceSoftmaxAdapter() = default;

    virtual void               init(Onnx::Session& session, Onnx::IOMapping& mapping);
    virtual Score              get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);
    virtual std::vector<Score> get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs);

private:
    const std::string weightsFile_;
    const std::string biasFile_;

    Math::FastMatrix<float> weights_;
    Math::FastVector<float> bias_;
};

inline OnnxNceSoftmaxAdapter::OnnxNceSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config),
          weightsFile_(paramWeightsFile(config)),
          biasFile_(paramBiasFile(config)) {
}

}  // namespace Lm

#endif  // _LM_ONNX_NCE_SOFTMAX_ADAPTER_HH
