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
