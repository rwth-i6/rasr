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

#include "GreedySearch.hh"
#include <Lattice/LatticeAdaptor.hh>

namespace Search {

const Core::ParameterBool GreedySearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt  GreedySearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the lexicon. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool GreedySearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);

GreedySearch::GreedySearch(const Core::Configuration& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          useBlank_(paramUseBlank(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          labelScorer_(),
          numClasses_(0ul),
          hyp_() {
}

void GreedySearch::reset() {
    verify(labelScorer_);
    labelScorer_->reset();
    hyp_.reset();
    hyp_.history = labelScorer_->getStartHistory();
}

Speech::ModelCombination::Mode GreedySearch::modelCombinationNeeded() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool GreedySearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    reset();
    return true;
}

void GreedySearch::enterSegment() {
    verify(labelScorer_);
    labelScorer_->reset();
}

void GreedySearch::enterSegment(Bliss::SpeechSegment const*) {
    verify(labelScorer_);
    labelScorer_->reset();
}

void GreedySearch::finishSegment() {
    verify(labelScorer_);
    labelScorer_->signalNoMoreFeatures();
    decodeMore();
}

void GreedySearch::addFeature(Nn::FeatureVectorRef feature) {
    verify(labelScorer_);
    labelScorer_->addInput(feature);
}

void GreedySearch::addFeature(Core::Ref<const Speech::Feature> feature) {
    verify(labelScorer_);
    labelScorer_->addInput(feature);
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedySearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(hyp_.traceback));
}

Core::Ref<const LatticeAdaptor> GreedySearch::getCurrentBestWordLattice() const {
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

void GreedySearch::resetStatistics() {}

void GreedySearch::logStatistics() const {}

Nn::LabelScorer::TransitionType GreedySearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
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

void GreedySearch::LabelHypothesis::reset() {
    history      = Nn::LabelHistoryRef();
    currentLabel = Core::Type<Nn::LabelIndex>::max;
    score        = 0.0f;
    traceback.clear();
}

void GreedySearch::LabelHypothesis::extend(const HypothesisExtension& extension) {
    this->history = extension.history;
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

bool GreedySearch::decodeStep() {
    verify(labelScorer_);
    verify(hyp_.history);

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
        requests.push_back({hyp_.history, idx, transitionType});
    }

    auto result = labelScorer_->getScoresWithTime(requests);
    if (not result.has_value()) {
        return false;
    }
    const auto& [scores, times] = result.value();

    auto  bestIdx     = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
    auto& bestRequest = requests.at(bestIdx);
    auto  newHistory  = labelScorer_->extendedHistory({bestRequest.history, bestRequest.nextToken, bestRequest.transitionType});
    hyp_.extend({lemmas.first[bestIdx], newHistory, bestRequest.nextToken, scores[bestIdx], times.at(bestIdx), bestRequest.transitionType});

    return true;
}

}  // namespace Search
