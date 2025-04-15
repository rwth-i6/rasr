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

public:
    CTCPrefixLabelScorer(Core::Configuration const& config);
    virtual ~CTCPrefixLabelScorer() = default;

    void reset() override;
    void signalNoMoreFeatures() override;
    void addInput(DataView const& input) override;
    void addInputs(DataView const& inputs, size_t nTimesteps) override;

    ScoringContextRef getInitialScoringContext() override;

    ScoringContextRef extendedScoringContext(LabelScorer::Request const& request) override;

    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTime(LabelScorer::Request const& request) override;

private:
    Math::FastMatrix<Score> ctcScores_;
    PrefixScoringContextRef getProperInitialScoringContext();

    LabelIndex             blankIndex_;
    Core::Ref<LabelScorer> ctcScorer_;
    bool                   expectMoreFeatures_;
};

}  // namespace Nn

#endif  // PREFIX_LABEL_SCORER_HH
