/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#ifndef _LM_ONNX_STATELESS_LM_HH
#define _LM_ONNX_STATELESS_LM_HH

#include <deque>

#include <Onnx/Model.hh>

#include "AbstractNNLanguageModel.hh"

namespace Lm {

struct NNCacheWithScores : public Lm::NNCacheWithStats {
    virtual ~NNCacheWithScores() = default;

    std::vector<Score> scores;
};

/*
 * Simple ONNX Language Model without any state caching. The entire token history is fed into the ONNX model
 * for each score request. This trades efficiency for simplicity and flexibility. Thus, it is mostly useful
 * for prototyping and models with a relatively small search space.
 */
class OnnxStatelessLm : public AbstractNNLanguageModel {
    typedef AbstractNNLanguageModel Precursor;
    typedef NNCacheWithScores       HistoryDescriptor;

public:
    OnnxStatelessLm(const Core::Configuration& c, Bliss::LexiconRef l);
    ~OnnxStatelessLm() = default;

    // Single sentence-begin token
    History startHistory() const;

    // Append token to token sequence
    History extendedHistory(const History& hist, Token nextToken) const;

    // Scoring by forwarding histories through ONNX model
    Score score(const History& hist, Token nextToken) const;

private:
    mutable Onnx::Model onnxModel_;

    std::string inputTokensName_;
    std::string inputLengthsName_;
    std::string scoresName_;

    size_t maxBatchSize_;

    // When new histories are created through `extendedHistory`, they are put into this queue for batched forwarding
    // because it is expected that we need to compute scores for them in the future anyway.
    mutable std::deque<History> batchQueue_;

    // Batch of histories which are forwarded at once
    mutable std::vector<History> batch_;

    // Cached history object containing only a single sentence-begin token
    History startHistory_;

    // Initialize vocabulary and start history
    void load();

    // Creates a batch of histories that contains `hist`` plus additional histories fetched from the `batchQueue_`
    void makeBatch(History const& hist) const;

    // Score all histories inside `batch_`
    void scoreBatch() const;
};

}  // namespace Lm

#endif  // _LM_ONNX_SIMPLE_TRANSFORMER_LM_HH
