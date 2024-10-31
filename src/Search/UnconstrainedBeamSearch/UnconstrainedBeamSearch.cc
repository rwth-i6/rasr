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

#include "UnconstrainedBeamSearch.hh"
#include <Lattice/LatticeAdaptor.hh>
#include <algorithm>
#include <numeric>
#include <strings.h>

namespace Search {

const Core::ParameterInt   UnconstrainedBeamSearch::paramMaxBeamSize("max-beam-size", "Maximum number of elements in the search beam.", 1);
const Core::ParameterInt   UnconstrainedBeamSearch::paramTopKTokens("top-k-tokens", "Only consider the k most likely successor tokens for each hypothesis expansion.", Core::Type<int>::max);
const Core::ParameterFloat UnconstrainedBeamSearch::paramScoreThreshold("score-threshold", "Prune any hypotheses whose score is at least this much worse than the best hypothesis.", Core::Type<Score>::max);
const Core::ParameterBool  UnconstrainedBeamSearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt   UnconstrainedBeamSearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the lexicon. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool  UnconstrainedBeamSearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);
const Core::ParameterBool  UnconstrainedBeamSearch::paramUseSentenceEnd("use-sentence-end", "Declare one sentence-end label such that search stops once this label is hypothesized.", false);
const Core::ParameterBool  UnconstrainedBeamSearch::paramSentenceEndIndex("sentence-end-index", "Index of the sentence-end label in the lexicon. Only necessarry if use-sentence-end is true.", 0);

UnconstrainedBeamSearch::UnconstrainedBeamSearch(const Core::Configuration& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          topKTokens_(paramTopKTokens(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          useBlank_(paramUseBlank(config)),
          useSentenceEnd_(paramUseSentenceEnd(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          sentenceEndIndex_(paramSentenceEndIndex(config)),
          labelScorer_(),
          numClasses_(0ul),
          beam_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_() {
    beam_.reserve(maxBeamSize_);
    useTokenPruning_ = topKTokens_ != Core::Type<int>::max;
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
}

void UnconstrainedBeamSearch::reset() {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    beam_.clear();
    beam_.push_back({});
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    initializationTime_.toc();
}

Speech::ModelCombination::Mode UnconstrainedBeamSearch::modelCombinationNeeded() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool UnconstrainedBeamSearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    useTokenPruning_ = topKTokens_ < lexicon_->nLemmas();

    reset();
    return true;
}

void UnconstrainedBeamSearch::enterSegment() {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    initializationTime_.toc();
}

void UnconstrainedBeamSearch::enterSegment(Bliss::SpeechSegment const*) {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    initializationTime_.toc();
}

void UnconstrainedBeamSearch::finishSegment() {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.toc();
    decodeMore();
}

void UnconstrainedBeamSearch::addFeature(Nn::FeatureVectorRef feature) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInput(feature);
    featureProcessingTime_.toc();
}

void UnconstrainedBeamSearch::addFeature(Core::Ref<const Speech::Feature> feature) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInput(feature);
    featureProcessingTime_.toc();
}

Core::Ref<const SearchAlgorithmV2::Traceback> UnconstrainedBeamSearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(beam_.front().traceback));
}

Core::Ref<const LatticeAdaptor> UnconstrainedBeamSearch::getCurrentBestWordLattice() const {
    if (beam_.front().traceback.empty()) {
        return Core::ref(new Lattice::WordLatticeAdaptor());
    }

    // use default LemmaAlphabet mode of StandardWordLattice
    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);

    // create a linear lattice from the traceback
    Fsa::State* currentState = result->initialState();
    for (auto it = beam_.front().traceback.begin(); it != beam_.front().traceback.end(); ++it) {
        // wordBoundaries->set(currentState->id(), Lattice::WordBoundary(static_cast<Speech::TimeframeIndex>(it->time.endTime())));
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(it->time));
        Fsa::State* nextState;
        if (std::next(it) == beam_.front().traceback.end()) {
            nextState = result->finalState();
        }
        else {
            nextState = result->newState();
        }
        ScoreVector scores = it->scores;
        if (it != beam_.front().traceback.begin()) {
            scores -= std::prev(it)->scores;
        }
        result->newArc(currentState, nextState, it->lemma, scores.acoustic, scores.lm);
        currentState = nextState;
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

void UnconstrainedBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
}

void UnconstrainedBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("initialization-time") + Core::XmlAttribute("unit", "milliseconds") << initializationTime_.total << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") + Core::XmlAttribute("unit", "milliseconds") << featureProcessingTime_.total << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") + Core::XmlAttribute("unit", "milliseconds") << scoringTime_.total << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") + Core::XmlAttribute("unit", "milliseconds") << contextExtensionTime_.total << Core::XmlClose("context-extension-time");
}

Nn::LabelScorer::TransitionType UnconstrainedBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
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

UnconstrainedBeamSearch::LabelHypothesis::LabelHypothesis(const UnconstrainedBeamSearch::LabelHypothesis& base)
        : scoringContext(base.scoringContext), currentLabel(base.currentLabel), score(base.score), traceback(base.traceback) {}

UnconstrainedBeamSearch::LabelHypothesis::LabelHypothesis(const UnconstrainedBeamSearch::LabelHypothesis& base, const UnconstrainedBeamSearch::HypothesisExtension& extension)
        : scoringContext(extension.scoringContext), currentLabel(extension.label), score(base.score + extension.score), traceback(base.traceback) {
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

std::string UnconstrainedBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", traceback: ";
    for (auto& item : traceback) {
        ss << item.lemma->symbol() << " ";
    }
    return ss.str();
}

bool UnconstrainedBeamSearch::decodeStep() {
    verify(labelScorer_);

    if (useSentenceEnd_) {
        // If all hypotheses in the beam have reached sentence-end, no further decode step is performed
        if (not std::any_of(beam_.begin(), beam_.end(), [this](const auto& hyp) { return hyp.currentLabel != sentenceEndIndex_; })) {
            return false;
        }
    }

    /*
     * Create scoring requests for the label scorer.
     * Each (unfinished) hypothesis together with each possible successor makes up a request.
     */

    auto nLemmas = lexicon_->nLemmas();

    std::vector<Nn::LabelScorer::Request> requests;
    requests.reserve(nLemmas * beam_.size());

    std::vector<LabelHypothesis*> baseHyps;  // Track the hypothesis that each request is based on
    baseHyps.reserve(nLemmas * beam_.size());

    // assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    size_t numUnfinishedHyps = 0ul;

    for (auto& hyp : beam_) {
        if (useSentenceEnd_ and hyp.currentLabel == sentenceEndIndex_) {
            // Hypothesis is finished and no successors are considered
            continue;
        }
        ++numUnfinishedHyps;

        // Iterate over possible successors
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      idx = lemma->id();

            auto transitionType = inferTransitionType(hyp.currentLabel, idx);
            requests.push_back({hyp.scoringContext, idx, transitionType});
            baseHyps.push_back(&hyp);
        }
    }

    /*
     * Perform scoring of the requests with the label scorer
     */
    scoringTime_.tic();
    auto result = labelScorer_->getScoresWithTimes(requests);
    scoringTime_.toc();
    if (not result) {
        return false;
    }
    const auto& scoresWithTimes = result.value();

    std::vector<size_t> indices;  // Index vector to keep track of which requests survive pruning

    /*
     * Perform top-k pruning for the successor tokens of each unfinished hypothesis
     */
    if (useTokenPruning_) {
        verify(topKTokens_ < lexicon_->nLemmas());
        indices.resize(numUnfinishedHyps * topKTokens_);
        for (size_t hypIndex = 0ul; hypIndex < numUnfinishedHyps; ++hypIndex) {
            std::vector<size_t> hypRequestIndices(nLemmas);  // Indices in requests vector that belong to current hyp
            std::iota(hypRequestIndices.begin(), hypRequestIndices.end(), hypIndex * nLemmas);

            // Partially sort indices such that the topKTokens_ first indices belong to the best scoring successors
            std::nth_element(hypRequestIndices.begin(), hypRequestIndices.begin() + topKTokens_, hypRequestIndices.end(),
                             [&scoresWithTimes](size_t a, size_t b) {
                                 return scoresWithTimes.scores[a] < scoresWithTimes.scores[b];
                             });
            // Copy best indices back into global index vector
            std::copy(hypRequestIndices.begin(), hypRequestIndices.begin() + topKTokens_, indices.begin() + hypIndex * topKTokens_);
        }
    }

    /*
     * Perform pre-pruning to maxBeamSize_ of all unfinished hypothesis extensions
     */
    if (indices.size() > maxBeamSize_) {
        // Sort index tensor by associated score value such that the first `beamSize_` elements are the best and sorted
        std::partial_sort(indices.begin(), indices.begin() + maxBeamSize_, indices.end(),
                          [&baseHyps, &scoresWithTimes](size_t a, size_t b) {
                              return baseHyps[a]->score + scoresWithTimes.scores[a] < baseHyps[b]->score + scoresWithTimes.scores[b];  // Compare combined scores
                          });
        indices.resize(maxBeamSize_);  // Get rid of excessive elements
    }

    /*
     * Score-based pruning of the unfinished hypotheses
     */
    if (useScorePruning_) {
        auto pruningThreshold = baseHyps[indices.front()]->score + scoresWithTimes.scores[indices.front()] + scoreThreshold_;
        indices.erase(
                std::remove_if(indices.begin(), indices.end(),
                               [&pruningThreshold, &baseHyps, &scoresWithTimes](size_t index) {
                                   return baseHyps[index]->score + scoresWithTimes.scores[index] > pruningThreshold;
                               }),
                indices.end());
    }

    /*
     * Create new beam containing all finished hypotheses from before and new extensions of unfinished hypotheses
     */
    std::vector<LabelHypothesis> newBeam;
    newBeam.reserve(2 * maxBeamSize_);  // beamSize_ expansions plus up to beamSize_ reviving hypotheses that have reached sentence-end before

    // Unfinished hyps
    for (auto index : indices) {
        const auto& request = requests.at(index);

        contextExtensionTime_.tic();
        auto newScoringContext = labelScorer_->extendedScoringContext({request.context, request.nextToken, request.transitionType});
        contextExtensionTime_.toc();

        newBeam.push_back(
                {*baseHyps.at(index),
                 {lemmas.first[request.nextToken],
                  newScoringContext,
                  request.nextToken,
                  scoresWithTimes.scores.at(index),
                  scoresWithTimes.timesteps.at(index),
                  request.transitionType}});
    }

    // Finished hyps
    if (useSentenceEnd_) {
        for (auto& hyp : beam_) {
            if (hyp.currentLabel == sentenceEndIndex_) {
                newBeam.push_back({hyp});
            }
        }
    }

    /*
     * Final pruning down to maxBeamSize_ elements
     */
    if (newBeam.size() > maxBeamSize_) {
        std::partial_sort(newBeam.begin(), newBeam.begin() + maxBeamSize_, newBeam.end(),
                          [](LabelHypothesis& hyp1, LabelHypothesis& hyp2) {
                              return hyp1.score < hyp2.score;
                          });
        newBeam.resize(maxBeamSize_);
    }

    /*
     * Score-based pruning of the final remaining hypotheses
     */
    if (useScorePruning_) {
        auto pruningThreshold = newBeam.front().score + scoreThreshold_;
        newBeam.erase(
                std::remove_if(newBeam.begin(), newBeam.end(),
                               [&pruningThreshold](const LabelHypothesis& hyp) {
                                   return hyp.score > pruningThreshold;
                               }),
                newBeam.end());
    }

    beam_.swap(newBeam);

    return true;
}

}  // namespace Search
