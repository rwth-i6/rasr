/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH
#define _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH

#include <Math/FastMatrix.hh>
#include <Math/FastVector.hh>

#include "TFSoftmaxAdapter.hh"

namespace Lm {

template<typename T>
class TFQuantizedBlasNceSoftmaxAdapter : public TFSoftmaxAdapter {
public:
    using Precursor = TFSoftmaxAdapter;

    static const Core::ParameterFloat paramNNOutputEpsilon;
    static const Core::ParameterFloat paramWeightsBiasEpsilon;

    TFQuantizedBlasNceSoftmaxAdapter(Core::Configuration const& config);
    virtual ~TFQuantizedBlasNceSoftmaxAdapter() = default;

    virtual void  init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);

private:
    const float nnOutputEpsilon_;
    const float weightsBiasEpsilon_;

    Math::FastMatrix<T>     weights_;
    Math::FastVector<float> bias_;
};

using TFQuantizedBlasNceSoftmaxAdapter16Bit = TFQuantizedBlasNceSoftmaxAdapter<s16>;
using TFQuantizedBlasNceSoftmaxAdapter8Bit  = TFQuantizedBlasNceSoftmaxAdapter<s8>;

// inline implementations

template<typename T>
inline TFQuantizedBlasNceSoftmaxAdapter<T>::TFQuantizedBlasNceSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config),
          nnOutputEpsilon_(paramNNOutputEpsilon(config)),
          weightsBiasEpsilon_(paramWeightsBiasEpsilon(config)) {
}

template<typename T>
inline void TFQuantizedBlasNceSoftmaxAdapter<T>::init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) {
    auto const&                     weight_tensor_info = output_map.get_info("weights");
    auto const&                     bias_tensor_info   = output_map.get_info("bias");
    std::vector<Tensorflow::Tensor> tensors;
    session.run({}, {weight_tensor_info.tensor_name(), bias_tensor_info.tensor_name()}, {}, tensors);
    Math::FastMatrix<float> float_weights;
    tensors[0].get(float_weights, true);
    tensors[1].get(bias_);

    float inv_scale = 1.0f / weightsBiasEpsilon_;
    float min_val   = std::numeric_limits<T>::min();
    float max_val   = std::numeric_limits<T>::max();
    weights_        = Math::FastMatrix<s16>(float_weights.nRows(), float_weights.nColumns());
    for (size_t c = 0ul; c < float_weights.nColumns(); c++) {
        for (size_t r = 0ul; r < float_weights.nRows(); r++) {
            float val      = float_weights(r, c) * inv_scale;
            val            = std::max(val, min_val);
            val            = std::min(val, max_val);
            weights_(r, c) = val;
        }
    }
}

}  // namespace Lm

#endif /* _LM_QUANTIZED_BLAS_NCE_SOFTMAX_ADAPTER_HH */
