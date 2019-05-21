/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "AverageFeatureScorerActivation.hh"

#include <Mm/Module.hh>

using namespace Speech;

const Core::ParameterInt  AverageFeatureScorerActivation::paramPrecision("output-precision", "precision of the output channel", 20);
const Core::ParameterBool AverageFeatureScorerActivation::paramTransformToProbabilities("transform-to-probabilities", "wether to transform scores into probability domain", true);

AverageFeatureScorerActivation::AverageFeatureScorerActivation(const Core::Configuration& config, bool loadFromFile)
        : Core::Component(config),
          Precursor(config, loadFromFile),
          transform_to_probabilities_(paramTransformToProbabilities(config)),
          feature_scorer_(Mm::Module::instance().createScaledFeatureScorer(select("mixture-set"), Core::Ref<Mm::AbstractMixtureSet>(Mm::Module::instance().readMixtureSet(select("mixture-set"))))),
          scorers_(),
          n_frames_(0u),
          scores_(0ul),
          outputChannel_(config, "output", Core::Channel::disabled) {}

AverageFeatureScorerActivation::~AverageFeatureScorerActivation() {}

void AverageFeatureScorerActivation::leaveSpeechSegment(Bliss::SpeechSegment* segment) {
    for (size_t i = 0ul; i < scorers_.size(); i++) {
        if (scores_.empty()) {
            scores_.resize(scorers_[i]->nEmissions(), 0.0);
        }
        for (size_t j = 0ul; j < scores_.size(); j++) {
            Score s = scorers_[i]->score(j);
            if (transform_to_probabilities_) {
                scores_[j] += std::exp(-s);
            }
            else {
                scores_[j] += s;
            }
        }
    }
    n_frames_ += scorers_.size();
    scorers_.clear();
    feature_scorer_->reset();

    Precursor::leaveSpeechSegment(segment);
}

void AverageFeatureScorerActivation::processFeature(Core::Ref<const Feature> f) {
    scorers_.push_back(feature_scorer_->getScorer(f));
}

void AverageFeatureScorerActivation::write() {
    std::transform(scores_.begin(), scores_.end(), scores_.begin(), [=](Score e) { return e / n_frames_; });
    if (outputChannel_.isOpen()) {
        outputChannel_.precision(paramPrecision(config));

        outputChannel_ << Core::XmlOpen("activations");
        outputChannel_ << Core::XmlOpen("num_frames");
        outputChannel_ << n_frames_;
        outputChannel_ << Core::XmlClose("num_frames");
        for (size_t i = 0ul; i < scores_.size(); i++) {
            outputChannel_ << Core::XmlOpen("score") + Core::XmlAttribute("emission", i);
            outputChannel_ << scores_[i];
            outputChannel_ << Core::XmlClose("score");
        }
        outputChannel_ << Core::XmlClose("activations");
    }
    else {
        criticalError("Could not dump scores since channel \"output\" is not open.");
    }
}
