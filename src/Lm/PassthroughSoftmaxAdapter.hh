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
#ifndef _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH
#define _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH

#include "SoftmaxAdapter.hh"

namespace Lm {

class PassthroughSoftmaxAdapter : public SoftmaxAdapter {
public:
    using Precursor = SoftmaxAdapter;

    PassthroughSoftmaxAdapter(Core::Configuration const& config);
    virtual ~PassthroughSoftmaxAdapter() = default;

    virtual void  init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map);
    virtual Score get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx);

private:
};

// inline implementations

inline PassthroughSoftmaxAdapter::PassthroughSoftmaxAdapter(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif /* _LM_PASSTHROUGH_SOFTMAX_ADAPTER_HH */
