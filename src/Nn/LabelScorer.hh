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

#ifndef LABEL_SCORER_HH
#define LABEL_SCORER_HH

#include <Core/Component.hh>
#include "Core/Parameter.hh"
#include "Decoder.hh"
#include "Encoder.hh"
#include "LabelHistory.hh"

namespace Nn {

// struct ContextScorer {
//     std::vector<Score> getScores(std::vector<History> histories, std::vector<LabelId> labels, std::vector<bool> isLoop);  // TODO: maybe pack in struct?
// };

// TODO: Bonus points for LegacyFeatureScorerLabelScorer

// Define enum values for different predefined label scorers with specific encoder and decoder
enum LabelScorerType {
    NoOpLabelScorer,
    OnnxEncoderLabelScorer,
    LegacyFeatureScorerLabelScorer,
};

// Glue class that couples encoder and decoder
// Purpose is creation of the right encoder/decoder combination according to a set of predefined types
// as well as automatic information flow between encoder and decoder
class LabelScorer : public virtual Core::Component, public Core::ReferenceCounted {
public:
    static const Core::Choice          choiceType;
    static const Core::ParameterChoice paramType;

    LabelScorer(const Core::Configuration& config);
    virtual ~LabelScorer() = default;

    // Reset encoder and decoder
    void reset();

    // Get start history for decoder
    Core::Ref<LabelHistory> getStartHistory();

    // Extend history for decoder
    void extendHistory(Core::Ref<LabelHistory> history, LabelIndex label, bool isLoop);

    // Add a single input feature to the encoder
    void addInput(FeatureVectorRef input);
    void addInput(Core::Ref<const Speech::Feature> input);

    // Tells the LabelScorer that there will be no more input features coming in the current segment
    void signalSegmentEnd();

    // Runs requests through decoder function of the same name
    void getDecoderStepScores(std::vector<ScoreRequest>& requests);

protected:
    Core::Ref<Encoder> encoder_;
    Core::Ref<Decoder> decoder_;

private:
    LabelScorerType type_;
    void            initEncoderDecoder();

    void encode();
};

}  // namespace Nn

#endif  // LABEL_SCORER_HH
