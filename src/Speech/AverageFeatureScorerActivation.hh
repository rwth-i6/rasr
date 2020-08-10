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
#ifndef _SPEECH_AVERAGE_FEATURE_SCORER_ACTIVATION_HH
#define _SPEECH_AVERAGE_FEATURE_SCORER_ACTIVATION_HH

#include <Mm/ScaledFeatureScorer.hh>

#include "DataExtractor.hh"

namespace Speech {

/* This class computes the average scores given by a Feature scorer,
 * this can be useful to compute soft priors */
class AverageFeatureScorerActivation : public FeatureExtractor {
public:
    typedef FeatureExtractor Precursor;
    typedef Mm::Score        Score;

    static const Core::ParameterInt  paramPrecision;
    static const Core::ParameterBool paramTransformToProbabilities;

    AverageFeatureScorerActivation(const Core::Configuration& configuration, bool loadFromFile = true);
    virtual ~AverageFeatureScorerActivation();

    virtual void leaveSpeechSegment(Bliss::SpeechSegment* segment);
    virtual void processFeature(Core::Ref<const Feature>);

    void write();

private:
    bool transform_to_probabilities_;

    Core::Ref<Mm::ScaledFeatureScorer>     feature_scorer_;
    std::vector<Mm::FeatureScorer::Scorer> scorers_;
    u32                                    n_frames_;
    std::vector<Score>                     scores_;

    Core::XmlChannel outputChannel_;
};

}  // namespace Speech

#endif  // _SPEECH_AVERAGE_FEATURE_SCORER_ACTIVATION_HH
