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

#include "LexiconfreeBeamSearch.hh"
#include <Lattice/LatticeAdaptor.hh>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <strings.h>
#include "Core/XmlStream.hh"
#include "Nn/LabelScorer/LabelScorer.hh"
#include "Nn/LabelScorer/ScoringContext.hh"
#include "Nn/LabelScorer/SharedDataHolder.hh"

namespace Search {

const Core::ParameterInt   LexiconfreeBeamSearch::paramMaxBeamSize("max-beam-size", "Maximum number of elements in the search beam.", 1);
const Core::ParameterInt   LexiconfreeBeamSearch::paramTopKTokens("top-k-tokens", "Only consider the k most likely successor tokens for each hypothesis expansion.", Core::Type<int>::max);
const Core::ParameterFloat LexiconfreeBeamSearch::paramScoreThreshold("score-threshold", "Prune any hypotheses whose score is at least this much worse than the best hypothesis.", Core::Type<Score>::max);
const Core::ParameterFloat LexiconfreeBeamSearch::paramLengthNormScale("length-norm-scale", "Scaling factor for the hypothesis length normalization.", 0.0);
const Core::ParameterBool  LexiconfreeBeamSearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt   LexiconfreeBeamSearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the lexicon. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool  LexiconfreeBeamSearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);
const Core::ParameterBool  LexiconfreeBeamSearch::paramUseSentenceEnd("use-sentence-end", "Declare one sentence-end label such that search stops once this label is hypothesized.", false);
const Core::ParameterBool  LexiconfreeBeamSearch::paramSentenceEndIndex("sentence-end-index", "Index of the sentence-end label in the lexicon. Only necessarry if use-sentence-end is true.", 0);
const Core::ParameterBool  LexiconfreeBeamSearch::paramLogStepwiseStatistics("log-stepwise-statistics", "Log statistics about the search space at every step.", false);

LexiconfreeBeamSearch::LexiconfreeBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          topKTokens_(paramTopKTokens(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          useBlank_(paramUseBlank(config)),
          useSentenceEnd_(paramUseSentenceEnd(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          sentenceEndIndex_(paramSentenceEndIndex(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
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

void LexiconfreeBeamSearch::reset() {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    beam_.clear();
    beam_.push_back({});
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    initializationTime_.toc();
}

Speech::ModelCombination::Mode LexiconfreeBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    useTokenPruning_ = topKTokens_ < lexicon_->nLemmas();

    reset();
    return true;
}

void LexiconfreeBeamSearch::enterSegment(Bliss::SpeechSegment const*) {
    verify(labelScorer_);
    initializationTime_.tic();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.toc();
}

void LexiconfreeBeamSearch::finishSegment() {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.toc();
    decodeMore();
    logStatistics();
}

void LexiconfreeBeamSearch::passFeature(Nn::SharedDataHolder const& data, size_t featureSize) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInput(data, featureSize);
    featureProcessingTime_.toc();
}

void LexiconfreeBeamSearch::passFeatures(Nn::SharedDataHolder const& data, size_t timeSize, size_t featureSize) {
    verify(labelScorer_);
    featureProcessingTime_.tic();
    labelScorer_->addInputs(data, timeSize, featureSize);
    featureProcessingTime_.toc();
}

Core::Ref<const Traceback> LexiconfreeBeamSearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(beam_.front().traceback));
}

Core::Ref<const LatticeAdaptor> LexiconfreeBeamSearch::getCurrentBestWordLattice() const {
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
        ScoreVector scores = it->score;
        if (it != beam_.front().traceback.begin()) {
            scores -= std::prev(it)->score;
        }
        result->newArc(currentState, nextState, it->pronunciation->lemma(), scores.acoustic, scores.lm);
        currentState = nextState;
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

void LexiconfreeBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
}

void LexiconfreeBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.total << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.total << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.total << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.total << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
}

Nn::LabelScorer::TransitionType LexiconfreeBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
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

LexiconfreeBeamSearch::LabelHypothesis::LabelHypothesis(LexiconfreeBeamSearch::LabelHypothesis const& base)
        : scoringContext(base.scoringContext), currentLabel(base.currentLabel), score(base.score), length(base.length), traceback(base.traceback) {}

LexiconfreeBeamSearch::LabelHypothesis::LabelHypothesis(LexiconfreeBeamSearch::LabelHypothesis const& base, LexiconfreeBeamSearch::HypothesisExtension const& extension)
        : scoringContext(extension.scoringContext), currentLabel(extension.label), score(extension.score), length(base.length + 1), traceback(base.traceback) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            this->traceback.push_back(TracebackItem(extension.pron, extension.timestep, ScoreVector(score, {}), {}));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            if (not this->traceback.empty()) {
                this->traceback.back().score.acoustic = score;
            }
            break;
    }
}

std::string LexiconfreeBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", traceback: ";
    for (auto& item : traceback) {
        ss << item.pronunciation->lemma()->symbol() << " ";
    }
    return ss.str();
}

void        LexiconfreeBeamSearch::topKTokenPruning(std::vector<size_t>& indices, std::vector<Score> const& extensionScores, size_t numUnfinishedHyps) {
#pragma omp parallel for
    for (size_t hypIndex = 0ul; hypIndex < numUnfinishedHyps; ++hypIndex) {
        // Start and end index of scores coming from current hypothesis
        size_t start = hypIndex * lexicon_->nLemmas();
        size_t end   = start + lexicon_->nLemmas();

        // Make sure the best topKTokens_ scores are moved to the beginning of the [start, end) interval
        std::nth_element(indices.begin() + start, indices.begin() + start + topKTokens_, indices.begin() + end,
                         [&extensionScores](size_t a, size_t b) {
                             return extensionScores[a] < extensionScores[b];
                         });

        // Copy the topKTokens_ best indices to the front of indices (note: this location doesn't overlap with [start, end) of the next hyps because topKTokens_ < nLemmas)
        if (hypIndex > 0ul) {
            std::copy(indices.begin() + start, indices.begin() + start + topKTokens_, indices.begin() + hypIndex * topKTokens_);
        }
    }

    indices.resize(numUnfinishedHyps * topKTokens_);  // Throw away all pruned indices
}

template<typename T>
void LexiconfreeBeamSearch::beamPruning(std::vector<T>& hypotheses, std::function<bool(T const&, T const&)>&& compare) {
    if (hypotheses.size() > maxBeamSize_) {
        // Sort the hypotheses by associated score value such that the first `beamSize_` elements are the best and sorted
        std::partial_sort(hypotheses.begin(), hypotheses.begin() + maxBeamSize_, hypotheses.end(), compare);
        hypotheses.resize(maxBeamSize_);  // Get rid of excessive elements
    }
    else {
        // Fully sort the hypotheses by their score
        std::sort(hypotheses.begin(), hypotheses.end(), compare);
    }
}

template<typename T>
void LexiconfreeBeamSearch::scorePruning(std::vector<T>& hypotheses, std::function<float(T const&)>&& getScore) {
    // Compute the pruning threshold
    auto   pruningThreshold = getScore(hypotheses.front()) + scoreThreshold_;
    size_t numSurvivingHyps = 0ul;
    // Use the fact that hypotheses are sorted by corresponding score and prune all indices after the first one that
    // violates the score threshold
    for (auto& hyp : hypotheses) {
        if (getScore(hyp) > pruningThreshold) {
            break;
        }
        ++numSurvivingHyps;
    }
    hypotheses.resize(numSurvivingHyps);  // Resize the hypotheses to keep only the surviving items
}

template<typename T>
void LexiconfreeBeamSearch::recombination(std::vector<T>& hypotheses) {
    std::vector<T> newHypotheses;

    std::unordered_set<Nn::ScoringContextRef, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (const auto& hyp : hypotheses) {
        if (seenScoringContexts.find(hyp.scoringContext) == seenScoringContexts.end()) {
            newHypotheses.push_back(hyp);
            seenScoringContexts.insert(hyp.scoringContext);
        }
    }
    hypotheses.swap(newHypotheses);
}

bool LexiconfreeBeamSearch::decodeStep() {
    verify(labelScorer_);

    // If all hypotheses in the beam have reached sentence-end, no further decode step is performed
    if (useSentenceEnd_ and std::all_of(beam_.begin(), beam_.end(), [this](auto const& hyp) { return hyp.currentLabel == sentenceEndIndex_; })) {
        return false;
    }

    /*
     * Create scoring requests for the label scorer.
     * Each (unfinished) hypothesis together with each possible successor makes up a request.
     */

    auto nLemmas = lexicon_->nLemmas();

    // assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    std::vector<Nn::LabelScorer::Request> requests;
    std::vector<LabelHypothesis*>         baseHyps;        // Track the hypothesis that each request is based on
    std::vector<size_t>                   unfinishedHyps;  // Indices of hypotheses in beam that need to be extended
    std::vector<size_t>                   finishedHyps;

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        if (useSentenceEnd_ and beam_[hypIndex].currentLabel == sentenceEndIndex_) {
            finishedHyps.push_back(hypIndex);
        }
        else {
            unfinishedHyps.push_back(hypIndex);
        }
    }
    size_t numUnfinishedHyps = unfinishedHyps.size();
    size_t numFinishedHyps   = finishedHyps.size();

    requests.reserve(nLemmas * numUnfinishedHyps);
    baseHyps.reserve(nLemmas * numUnfinishedHyps);

    for (auto hypIndex : unfinishedHyps) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      idx = lemma->id();

            requests.push_back({hyp.scoringContext, idx, inferTransitionType(hyp.currentLabel, idx)});
            baseHyps.push_back(&hyp);
        }
    }

    /*
     * Perform scoring of the requests with the label scorer
     */
    scoringTime_.tic();
    auto result = labelScorer_->computeScoresWithTimes(requests);
    scoringTime_.toc();
    if (not result) {
        return false;
    }
    auto const&        scoresWithTimes = result.value();
    std::vector<Score> extensionScores = scoresWithTimes.scores;

    std::vector<Score> combinedScores(numUnfinishedHyps * nLemmas);
    for (size_t requestIndex = 0ul; requestIndex < requests.size(); ++requestIndex) {
        // Normalized the score by the hypothesis length:
        // normalized-score = total-score / ((current-length + 1) ^ lengthNormScale_)
        // with total-score = current-score * (current-length ^ lengthNormScale_) + extension-score
        // when lengthNormScale_ = 0 (per default), no normalization will be performed
        combinedScores[requestIndex] = (baseHyps[requestIndex]->score * std::pow(baseHyps[requestIndex]->length, lengthNormScale_) + extensionScores[requestIndex]) / std::pow(baseHyps[requestIndex]->length + 1, lengthNormScale_);
    }

    std::vector<size_t> indices(requests.size());  // Index vector to keep track of which requests survive pruning
    std::iota(indices.begin(), indices.end(), 0ul);

    /*
     * Perform top-k pruning for the successor tokens of each unfinished hypothesis
     */
    if (useTokenPruning_) {
        topKTokenPruning(indices, extensionScores, numUnfinishedHyps);
    }

    /*
     * Perform pre-pruning to maxBeamSize_ of all unfinished hypothesis extensions
     */
    beamPruning(
            indices,
            std::function<bool(size_t const&, size_t const&)>(
                    [&combinedScores](size_t a, size_t b) { return combinedScores[a] < combinedScores[b]; }));

    /*
     * Score-based pruning of the unfinished hypotheses
     */
    if (useScorePruning_) {
        scorePruning(
                indices,
                std::function<Score(const size_t&)>(
                        [&combinedScores](size_t index) { return combinedScores[index]; }));
    }

    /*
     * Create new beam containing all finished hypotheses from before and new extensions of unfinished hypotheses
     */
    std::vector<LabelHypothesis> newBeam;
    newBeam.reserve(indices.size() + numFinishedHyps);  // expansions surviving hypotheses that have reached sentence-end before

    // Unfinished hyps
    for (auto index : indices) {
        auto const& request = requests[index];

        contextExtensionTime_.tic();
        auto newScoringContext = labelScorer_->extendedScoringContext({request.context, request.nextToken, request.transitionType});
        contextExtensionTime_.toc();

        newBeam.push_back(
                {*baseHyps[index],
                 {lemmas.first[request.nextToken]->pronunciations().first,
                  newScoringContext,
                  request.nextToken,
                  combinedScores[index],
                  scoresWithTimes.timeframes[index],
                  request.transitionType}});
    }

    // Finished hyps
    for (auto hypIndex : finishedHyps) {
        newBeam.push_back(beam_[hypIndex]);
    }

    /*
     * Final pruning down to maxBeamSize_ elements
     */
    beamPruning(
            newBeam,
            std::function<bool(const LabelHypothesis&, const LabelHypothesis&)>(
                    [](const LabelHypothesis& hyp1, const LabelHypothesis& hyp2) {
                        return hyp1.score < hyp2.score;
                    }));

    /*
     * Score-based pruning of the final remaining hypotheses
     */
    if (useScorePruning_) {
        scorePruning(
                newBeam,
                std::function<Score(const LabelHypothesis&)>(
                        [](const LabelHypothesis& hyp) { return hyp.score; }));
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination(newBeam);

    beam_.swap(newBeam);

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
        clog() << Core::XmlOpen("active-hyps") << beam_.size() << Core::XmlClose("active-hyps");
        clog() << Core::XmlOpen("best-hyp-score") << beam_.front().score << Core::XmlClose("best-hyp-score");
        clog() << Core::XmlOpen("worst-hyp-score") << beam_.back().score << Core::XmlClose("worst-hyp-score");
        clog() << Core::XmlClose("search-step-stats");
    }

    return true;
}

}  // namespace Search
