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
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Nn/LabelScorer/SharedDataHolder.hh>
#include <algorithm>
#include <strings.h>

namespace Search {

const Core::ParameterInt   LexiconfreeBeamSearch::paramMaxBeamSize("max-beam-size", "Maximum number of elements in the search beam.", 1, 1);
const Core::ParameterInt   LexiconfreeBeamSearch::paramMaxBeamSizePerScorer("max-beam-size-per-scorer", "Maximum number of intermediate hypotheses to keep after the application of each scorer.", Core::Type<int>::max, 1);
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
          maxBeamSizePerScorer_(paramMaxBeamSizePerScorer(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          useBlank_(paramUseBlank(config)),
          useSentenceEnd_(paramUseSentenceEnd(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          sentenceEndIndex_(paramSentenceEndIndex(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          labelScorers_(),
          numClasses_(0ul),
          beam_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_() {
    beam_.reserve(maxBeamSize_);
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
}

void LexiconfreeBeamSearch::reset() {
    initializationTime_.tic();
    std::vector<Nn::ScoringContextRef> scoringContexts;
    scoringContexts.reserve(labelScorers_.size());
    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
        scoringContexts.push_back(labelScorer->getInitialScoringContext());
    }
    beam_.clear();
    beam_.push_back({});
    beam_.front().scoringContext = Core::ref(new Nn::CombineScoringContext(std::move(scoringContexts)));
    initializationTime_.toc();
}

Speech::ModelCombination::Mode LexiconfreeBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorers | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    reset();
    return true;
}

void LexiconfreeBeamSearch::enterSegment(Bliss::SpeechSegment const*) {
    initializationTime_.tic();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }
    resetStatistics();
    initializationTime_.toc();
}

void LexiconfreeBeamSearch::finishSegment() {
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.toc();
    decodeMore();
    logStatistics();
}

void LexiconfreeBeamSearch::passFeature(Nn::SharedDataHolder const& data, size_t featureSize) {
    featureProcessingTime_.tic();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(data, featureSize);
    }
    featureProcessingTime_.toc();
}

void LexiconfreeBeamSearch::passFeatures(Nn::SharedDataHolder const& data, size_t timeSize, size_t featureSize) {
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(data, timeSize, featureSize);
    }
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
        : scoringContext(base.scoringContext), currentLabel(base.currentLabel), score(base.score), length(base.length), traceback(base.traceback), lastTransitionType(base.lastTransitionType), finished(base.finished) {}

LexiconfreeBeamSearch::LabelHypothesis::LabelHypothesis(LexiconfreeBeamSearch::LabelHypothesis const& base, LexiconfreeBeamSearch::HypothesisExtension const& extension)
        : scoringContext(extension.scoringContext), currentLabel(extension.label), score(extension.score), length(base.length + 1), traceback(base.traceback), lastTransitionType(extension.transitionType), finished(base.finished) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            this->traceback.push_back(TracebackItem(extension.pron, extension.timestep + 1, ScoreVector(extension.score - base.score, {}), {}));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            if (not this->traceback.empty()) {
                this->traceback.back().score.acoustic += extension.score - base.score;
                this->traceback.back().time = extension.timestep + 1;
            }
            break;
    }
}

Score LexiconfreeBeamSearch::LabelHypothesis::lengthNormalizedScore(Score scale) const {
    if (scale == 0) {
        return score;
    }
    return score / std::pow(length, scale);
}

std::string LexiconfreeBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", traceback: ";
    for (auto& item : traceback) {
        ss << item.pronunciation->lemma()->symbol() << " ";
    }
    return ss.str();
}

template<typename T>
void LexiconfreeBeamSearch::beamPruning(std::vector<T>& hypotheses, std::function<bool(T const&, T const&)>&& compare, size_t maxSize) {
    if (hypotheses.size() <= maxBeamSize_) {
        return;
    }

    // Sort the hypotheses by associated score value such that the first `beamSize_` elements are the best and sorted
    std::partial_sort(hypotheses.begin(), hypotheses.begin() + maxBeamSize_, hypotheses.end(), compare);
    hypotheses.resize(maxSize);  // Get rid of excessive elements
}

template<typename T>
void LexiconfreeBeamSearch::scorePruning(std::vector<T>& hypotheses, std::function<float(T const&)>&& getScore) {
    std::sort(hypotheses.begin(), hypotheses.end(), [getScore](T const& hyp1, T const& hyp2) { return getScore(hyp1) < getScore(hyp2); });

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

    std::vector<size_t> unfinishedHyps;  // Indices of hypotheses in beam that need to be extended
    std::vector<size_t> finishedHyps;

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        if (beam_[hypIndex].finished) {
            finishedHyps.push_back(hypIndex);
        }
        else {
            unfinishedHyps.push_back(hypIndex);
        }
    }

    std::vector<HypothesisExtension> extensions;
    extensions.reserve(unfinishedHyps.size() * nLemmas);

    for (auto hypIndex : unfinishedHyps) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      idx = lemma->id();

            extensions.push_back(
                    {lemma->pronunciations().first,
                     hyp.scoringContext,
                     idx,
                     hyp.score,
                     0,
                     inferTransitionType(hyp.currentLabel, idx),
                     hypIndex});
        }
    }

    for (size_t labelScorerIdx = 0ul; labelScorerIdx < labelScorers_.size(); ++labelScorerIdx) {
        auto& labelScorer = labelScorers_[labelScorerIdx];

        std::vector<Nn::LabelScorer::Request> requests;
        requests.reserve(extensions.size());
        for (const auto& extension : extensions) {
            requests.push_back({extension.scoringContext->scoringContexts[labelScorerIdx], extension.label, extension.transitionType});
        }
        /*
         * Perform scoring of the requests with the label scorer
         */
        scoringTime_.tic();
        auto result = labelScorer->computeScoresWithTimes(requests);
        scoringTime_.toc();
        if (not result) {
            return false;
        }

        for (size_t idx = 0ul; idx < extensions.size(); ++idx) {
            extensions[idx].score += result->scores[idx];
            extensions[idx].timestep = std::max(extensions[idx].timestep, result->timeframes[idx]);
        }

        if (labelScorers_.size() > 1) {
            beamPruning(extensions, {[](HypothesisExtension const& ext1, HypothesisExtension const& ext2) { return ext1.score < ext2.score; }}, maxBeamSizePerScorer_);
            if (useScorePruning_) {
                scorePruning(extensions, {[](HypothesisExtension const& ext) { return ext.score; }});
            }
        }
    }

    std::vector<LabelHypothesis> newBeam;
    newBeam.reserve(extensions.size() + unfinishedHyps.size());

    for (auto const& extension : extensions) {
        newBeam.push_back({beam_[extension.baseHypIndex],
                           extension});
    }
    for (auto finishedHypIdx : finishedHyps) {
        newBeam.push_back(beam_[finishedHypIdx]);
    }

    beamPruning(newBeam, {[this](LabelHypothesis const& hyp1, LabelHypothesis const& hyp2) { return hyp1.lengthNormalizedScore(lengthNormScale_) < hyp2.lengthNormalizedScore(lengthNormScale_); }}, maxBeamSize_);
    if (useScorePruning_) {
        scorePruning(newBeam, {[](LabelHypothesis const& hyp) { return hyp.score; }});
    }

    for (auto& hyp : newBeam) {
        if (hyp.finished) {
            continue;
        }

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        newScoringContexts.reserve(labelScorers_.size());
        for (size_t labelScorerIdx = 0ul; labelScorerIdx < labelScorers_.size(); ++labelScorerIdx) {
            contextExtensionTime_.tic();
            newScoringContexts.push_back(labelScorers_[labelScorerIdx]->extendedScoringContext({
                    hyp.scoringContext->scoringContexts[labelScorerIdx],
                    hyp.currentLabel,
                    hyp.lastTransitionType,
            }));
            contextExtensionTime_.toc();
        }
        hyp.scoringContext = Core::ref(new Nn::CombineScoringContext(std::move(newScoringContexts)));
        if (useSentenceEnd_ and hyp.currentLabel == sentenceEndIndex_) {
            hyp.finished = true;
        }
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
