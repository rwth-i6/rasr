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

#include "UnconstrainedGreedySearch.hh"
#include <Lattice/LatticeAdaptor.hh>

namespace Search {

const Core::ParameterBool UnconstrainedGreedySearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt  UnconstrainedGreedySearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the lexicon. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool UnconstrainedGreedySearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);
const Core::ParameterBool UnconstrainedGreedySearch::paramUseSentenceEnd("use-sentence-end", "Declare one sentence-end label such that search stops once this label is hypothesized.", false);
const Core::ParameterBool UnconstrainedGreedySearch::paramSentenceEndIndex("sentence-end-index", "Index of the sentence-end label in the lexicon. Only necessarry if use-sentence-end is true.", 0);

UnconstrainedGreedySearch::UnconstrainedGreedySearch(const Core::Configuration& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          useBlank_(paramUseBlank(config)),
          useSentenceEnd_(paramUseSentenceEnd(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          sentenceEndIndex_(paramSentenceEndIndex(config)),
          labelScorer_(),
          numClasses_(0ul),
          hyp_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_() {
}

void UnconstrainedGreedySearch::reset() {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    hyp_.reset();
    hyp_.scoringContext = labelScorer_->getInitialScoringContext();
    initializationTime_.toc();
}

Speech::ModelCombination::Mode UnconstrainedGreedySearch::modelCombinationNeeded() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool UnconstrainedGreedySearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    reset();
    return true;
}

void UnconstrainedGreedySearch::enterSegment() {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    initializationTime_.toc();
}

void UnconstrainedGreedySearch::enterSegment(Bliss::SpeechSegment const*) {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    initializationTime_.toc();
}

void UnconstrainedGreedySearch::finishSegment() {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.toc();
    decodeMore();
}

void UnconstrainedGreedySearch::addFeature(Nn::FeatureVectorRef feature) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInput(feature);
    featureProcessingTime_.toc();
}

void UnconstrainedGreedySearch::addFeature(Core::Ref<const Speech::Feature> feature) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInput(feature);
    featureProcessingTime_.toc();
}

Core::Ref<const SearchAlgorithmV2::Traceback> UnconstrainedGreedySearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(hyp_.traceback));
}

Core::Ref<const LatticeAdaptor> UnconstrainedGreedySearch::getCurrentBestWordLattice() const {
    if (hyp_.traceback.empty()) {
        return Core::ref(new Lattice::WordLatticeAdaptor());
    }

    // use default LemmaAlphabet mode of StandardWordLattice
    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);

    // create a linear lattice from the traceback
    Fsa::State* currentState = result->initialState();
    for (auto it = hyp_.traceback.begin(); it != hyp_.traceback.end(); ++it) {
        // wordBoundaries->set(currentState->id(), Lattice::WordBoundary(static_cast<Speech::TimeframeIndex>(it->time.endTime())));
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(it->time));
        Fsa::State* nextState;
        if (std::next(it) == hyp_.traceback.end()) {
            nextState = result->finalState();
        }
        else {
            nextState = result->newState();
        }
        ScoreVector scores = it->scores;
        if (it != hyp_.traceback.begin()) {
            scores -= std::prev(it)->scores;
        }
        result->newArc(currentState, nextState, it->lemma, scores.acoustic, scores.lm);
        currentState = nextState;
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

void UnconstrainedGreedySearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
}

void UnconstrainedGreedySearch::logStatistics() const {
    clog() << Core::XmlOpen("initialization-time") + Core::XmlAttribute("unit", "milliseconds") << initializationTime_.total << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") + Core::XmlAttribute("unit", "milliseconds") << featureProcessingTime_.total << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") + Core::XmlAttribute("unit", "milliseconds") << scoringTime_.total << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") + Core::XmlAttribute("unit", "milliseconds") << contextExtensionTime_.total << Core::XmlClose("context-extension-time");
}

Nn::LabelScorer::TransitionType UnconstrainedGreedySearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = (prevLabel == blankLabelIndex_);
    bool nextIsBlank = (nextLabel == blankLabelIndex_);

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::BLANK_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::LABEL_TO_BLANK;
        }
        else if (allowLabelLoop_ and prevLabel == nextLabel) {
            return Nn::LabelScorer::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

void UnconstrainedGreedySearch::LabelHypothesis::reset() {
    scoringContext = Nn::ScoringContextRef();
    currentLabel   = Core::Type<Nn::LabelIndex>::max;
    score          = 0.0f;
    traceback.clear();
}

void UnconstrainedGreedySearch::LabelHypothesis::extend(const HypothesisExtension& extension) {
    this->scoringContext = extension.scoringContext;
    this->score += extension.score;
    this->currentLabel = extension.label;
    switch (extension.transitionType) {
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            this->traceback.push_back(TracebackItem(nullptr, extension.lemma, extension.timestep, ScoreVector(score, {})));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            if (not this->traceback.empty()) {
                this->traceback.back().scores.acoustic = score;
            }
            break;
    }
}

bool UnconstrainedGreedySearch::decodeStep() {
    verify(labelScorer_);
    verify(hyp_.scoringContext);

    HypothesisExtension bestExtension;

    // Fetch prev label from hypothesis because this may be expanded with a loop transition
    Nn::LabelIndex prevLabel = hyp_.currentLabel;

    // assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto                                  lemmas = lexicon_->lemmas();
    std::vector<Nn::LabelScorer::Request> requests;
    requests.reserve(lexicon_->nLemmas());
    for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
        const Bliss::Lemma* lemma(*lemmaIt);
        Nn::LabelIndex      idx = lemma->id();

        auto transitionType = inferTransitionType(prevLabel, idx);
        requests.push_back({hyp_.scoringContext, idx, transitionType});
    }

    scoringTime_.tic();
    auto result = labelScorer_->getScoresWithTimes(requests);
    scoringTime_.toc();
    if (not result.has_value()) {
        return false;
    }
    const auto& [scores, times] = result.value();

    auto  bestIdx     = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
    auto& bestRequest = requests.at(bestIdx);
    contextExtensionTime_.tic();
    auto newScoringContext = labelScorer_->extendedScoringContext({bestRequest.context, bestRequest.nextToken, bestRequest.transitionType});
    contextExtensionTime_.toc();

    hyp_.extend({lemmas.first[bestIdx], newScoringContext, bestRequest.nextToken, scores[bestIdx], times.at(bestIdx), bestRequest.transitionType});

    if (useSentenceEnd_ and bestRequest.nextToken == sentenceEndIndex_) {
        return false;
    }

    return true;
}

}  // namespace Search
