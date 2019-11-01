#ifndef _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH
#define _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH

#include <Math/FastMatrix.hh>
#include <Math/FastVector.hh>

#include "SoftmaxAdapter.hh"

namespace Lm {

template<typename T>
class QuantizedBlasNceSoftmaxAdapter : public SoftmaxAdapter {
public:
    using Precursor = SoftmaxAdapter;

    static const Core::ParameterFloat paramNNOutputEpsilon;
    static const Core::ParameterFloat paramWeightsBiasEpsilon;

    QuantizedBlasNceSoftmaxAdapter(Core::Configuration const& config);
    virtual ~QuantizedBlasNceSoftmaxAdapter() = default;

    virtual void init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);
private:
    const float nnOutputEpsilon_;
    const float weightsBiasEpsilon_;

    Math::FastMatrix<T>     weights_;
    Math::FastVector<float> bias_;
};

using QuantizedBlasNceSoftmaxAdapter16Bit = QuantizedBlasNceSoftmaxAdapter<s16>;
using QuantizedBlasNceSoftmaxAdapter8Bit  = QuantizedBlasNceSoftmaxAdapter<s8>;

// inline implementations

template<typename T>
inline QuantizedBlasNceSoftmaxAdapter<T>::QuantizedBlasNceSoftmaxAdapter(Core::Configuration const& config)
    : Precursor(config), nnOutputEpsilon_(paramNNOutputEpsilon(config)), weightsBiasEpsilon_(paramWeightsBiasEpsilon(config)) {
}

template<typename T>
inline void QuantizedBlasNceSoftmaxAdapter<T>::init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) {
    auto const& weight_tensor_info = output_map.get_info("weights");
    auto const& bias_tensor_info   = output_map.get_info("bias");
    std::vector<Tensorflow::Tensor> tensors;
    session.run({}, {weight_tensor_info.tensor_name(), bias_tensor_info.tensor_name()}, {}, tensors);
    Math::FastMatrix<float> float_weights;
    tensors[0].get(float_weights, true);
    tensors[1].get(bias_);

    float inv_scale = 1.0f / weightsBiasEpsilon_;
    float min_val = std::numeric_limits<T>::min();
    float max_val = std::numeric_limits<T>::max();
    weights_ = Math::FastMatrix<s16>(float_weights.nRows(), float_weights.nColumns());
    for (size_t c = 0ul; c < float_weights.nColumns(); c++) {
        for (size_t r = 0ul; r < float_weights.nRows(); r++) {
            float val = float_weights(r, c) * inv_scale;
            val = std::max(val, min_val);
            val = std::min(val, max_val);
            weights_(r, c) =  val;
        }
    }
}


}  // namespace Lm

#endif /* _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH */
