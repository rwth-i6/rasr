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
#ifndef _LM_BLAS_NCE_SOFTMAX_ADAPTER_HH
#define _LM_BLAS_NCE_SOFTMAX_ADAPTER_HH

#include "TFSoftmaxAdapter.hh"

namespace Lm {

class TFBlasNceSoftmaxAdapter : public TFSoftmaxAdapter {
public:
    using Precursor = TFSoftmaxAdapter;

    TFBlasNceSoftmaxAdapter(Core::Configuration const& config);
    virtual ~TFBlasNceSoftmaxAdapter() = default;

    virtual void               init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score              get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);
    virtual std::vector<Score> get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs);

private:
    std::vector<Tensorflow::Tensor> tensors_;
};

// inline implementations

inline TFBlasNceSoftmaxAdapter::TFBlasNceSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif /* _LM_BLAS_NCE_SOFTMAX_ADAPTER_HH */
