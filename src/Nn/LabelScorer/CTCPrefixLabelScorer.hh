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

#ifndef PREFIX_LABEL_SCORER_HH
#define PREFIX_LABEL_SCORER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Speech/Feature.hh>
#include "LabelScorer.hh"
#include "ScoringContext.hh"

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>

namespace Nn {

class CTCPrefixLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

    static const Core::ParameterInt paramBlankIndex;
    static const Core::ParameterInt paramVocabSize;

public:
    CTCPrefixLabelScorer(Core::Configuration const& config);
    virtual ~CTCPrefixLabelScorer() = default;

    void reset() override;
    void signalNoMoreFeatures() override;
    void addInput(DataView const& input) override;
    void addInputs(DataView const& inputs, size_t nTimesteps) override;

    ScoringContextRef getInitialScoringContext() override;

protected:
    ScoringContextRef extendedScoringContextInternal(LabelScorer::Request const& request) override;

    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTimeInternal(LabelScorer::Request const& request, std::optional<size_t> scorerIdx) override;

private:
    Math::FastMatrix<Score> ctcScores_;  // Cached T x V matrix of scores

    LabelIndex             blankIndex_;
    size_t                 vocabSize_;
    Core::Ref<LabelScorer> ctcScorer_;
    bool                   expectMoreFeatures_;

    /*
     * Retrieve matrix of CTC scores from sub-scorer. Assumes that these scores only depend on timestep and label index, not history or transition type.
     */
    void setupCTCScores();

    /*
     * Compute updated prefix scores.
     */
    void finalizeScoringContext(CTCPrefixScoringContextRef const& scoringContext) const;
};

}  // namespace Nn

#endif  // PREFIX_LABEL_SCORER_HH
