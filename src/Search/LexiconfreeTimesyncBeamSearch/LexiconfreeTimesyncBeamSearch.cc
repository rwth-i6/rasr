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

#include "LexiconfreeTimesyncBeamSearch.hh"
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <algorithm>
#include <strings.h>

namespace Search {

/*
 * =====================================
 * === LexiconfreeTimesyncBeamSearch ===
 * =====================================
 */

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of elements in the search beam.",
        1, 1);

const Core::ParameterFloat LexiconfreeTimesyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. If not set, the search will not use blank.",
        Core::Type<int>::max);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramAllowLabelLoop(
        "allow-label-loop",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramDebugLogging(
        "debug-logging",
        "Enable detailed logging for debugging purposes.",
        false);

LexiconfreeTimesyncBeamSearch::LexiconfreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugLogging_(paramDebugLogging(config)),
          labelScorer_(),
          beam_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_() {
    beam_.reserve(maxBeamSize_);
    useBlank_        = blankLabelIndex_ != Core::Type<int>::max;
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
}

void LexiconfreeTimesyncBeamSearch::reset() {
    initializationTime_.tic();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();

    initializationTime_.toc();
}

Speech::ModelCombination::Mode LexiconfreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    reset();
    return true;
}

void LexiconfreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.tic();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.toc();
}

void LexiconfreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.tic();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.toc();
    decodeManySteps();
    logStatistics();
}

void LexiconfreeTimesyncBeamSearch::putFeature(std::shared_ptr<const f32[]> const& data, size_t featureSize) {
    featureProcessingTime_.tic();
    labelScorer_->addInput(data, featureSize);
    featureProcessingTime_.toc();
}

void LexiconfreeTimesyncBeamSearch::putFeature(std::vector<f32> const& data) {
    featureProcessingTime_.tic();
    labelScorer_->addInput(data);
    featureProcessingTime_.toc();
}

void LexiconfreeTimesyncBeamSearch::putFeatures(std::shared_ptr<const f32[]> const& data, size_t timeSize, size_t featureSize) {
    featureProcessingTime_.tic();
    labelScorer_->addInputs(data, timeSize, featureSize);
    featureProcessingTime_.toc();
}

Core::Ref<const Traceback> LexiconfreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(beam_.front().traceback));
}

Core::Ref<const LatticeAdaptor> LexiconfreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    if (beam_.front().traceback.empty()) {
        return Core::ref(new Lattice::WordLatticeAdaptor());
    }

    // use default LemmaAlphabet mode of StandardWordLattice
    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);

    // create a linear lattice from the traceback
    Fsa::State* currentState = result->initialState();
    for (auto it = beam_.front().traceback.begin(); it != beam_.front().traceback.end(); ++it) {
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

void LexiconfreeTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
}

void LexiconfreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.getTotalMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.getTotalMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.getTotalMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.getTotalMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
}

Nn::LabelScorer::TransitionType LexiconfreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    // These checks will result in false if `blankLabelIndex_` is still `Core::Type<int>::max`, i.e. no blank is used
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

void LexiconfreeTimesyncBeamSearch::beamPruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions) const {
    if (extensions.size() <= maxBeamSize_) {
        return;
    }

    // Sort the hypotheses by associated score value such that the first `beamSize_` elements are the best and sorted
    std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize_, extensions.end());
    extensions.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeTimesyncBeamSearch::scorePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions) const {
    // Compute the pruning threshold
    auto   pruningThreshold = extensions.front().score + scoreThreshold_;
    size_t numSurvivingHyps = 0ul;
    // Use the fact that hypotheses are sorted by corresponding score and prune all indices after the first one that
    // violates the score threshold
    for (auto const& ext : extensions) {
        if (ext.score > pruningThreshold) {
            break;
        }
        ++numSurvivingHyps;
    }
    extensions.resize(numSurvivingHyps);  // Resize the hypotheses to keep only the surviving items
}

void LexiconfreeTimesyncBeamSearch::recombination(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    std::vector<LabelHypothesis> recombinedHypotheses;

    std::unordered_set<Nn::ScoringContextRef, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        if (seenScoringContexts.find(hyp.scoringContext) == seenScoringContexts.end()) {
            recombinedHypotheses.push_back(hyp);
            seenScoringContexts.insert(hyp.scoringContext);
        }
    }
    hypotheses.swap(recombinedHypotheses);
}

bool LexiconfreeTimesyncBeamSearch::decodeStep() {
    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    /*
     * Collect all possible extensions for all hypotheses in the beam.
     */
    std::vector<ExtensionCandidate> extensions;
    extensions.reserve(beam_.size() * lexicon_->nLemmas());

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            extensions.push_back(
                    {tokenIdx,
                     lemma->pronunciations().first,
                     hyp.score,
                     0,
                     inferTransitionType(hyp.currentToken, tokenIdx),
                     hypIndex});
        }
    }

    /*
     * Create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    std::vector<Nn::LabelScorer::Request> requests;
    requests.reserve(extensions.size());
    for (const auto& extension : extensions) {
        requests.push_back({beam_[extension.baseHypIndex].scoringContext, extension.nextToken, extension.transitionType});
    }

    /*
     * Perform scoring of all the requests with the label scorer.
     */
    scoringTime_.tic();
    auto result = labelScorer_->computeScoresWithTimes(requests);
    scoringTime_.toc();

    if (not result) {
        // LabelScorer could not compute scores -> no search step can be made.
        return false;
    }

    for (size_t requestIdx = 0ul; requestIdx < extensions.size(); ++requestIdx) {
        extensions[requestIdx].score += result->scores[requestIdx];
        extensions[requestIdx].timeframe = result->timeframes[requestIdx];
    }

    /*
     * Prune set of possible extensions by max beam size and possibly also by score.
     */
    beamPruning(extensions);
    if (debugLogging_) {
        log() << extensions.size() << " candidates survived beam pruning";
    }

    std::sort(extensions.begin(), extensions.end());

    if (useScorePruning_) {
        // Extensions are sorted by score after `beamPruning`.
        scorePruning(extensions);

        if (debugLogging_) {
            log() << extensions.size() << " candidates survived score pruning";
        }
    }

    /*
     * Create new beam from surviving extensions.
     */
    std::vector<LabelHypothesis> newBeam;
    newBeam.reserve(extensions.size());

    for (auto const& extension : extensions) {
        auto const& baseHyp           = beam_[extension.baseHypIndex];
        auto        newScoringContext = labelScorer_->extendedScoringContext({baseHyp.scoringContext, extension.nextToken, extension.transitionType});
        newBeam.push_back({baseHyp, extension, newScoringContext});
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination(newBeam);
    if (debugLogging_) {
        log() << newBeam.size() << " hypotheses after recombination";

        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < newBeam.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << newBeam[hypIdx].toString() << "\n";
        }
        log() << ss.str();
    }

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

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Core::Type<Nn::LabelIndex>::max),
          score(0.0),
          traceback() {}

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeTimesyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                             newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          score(extension.score),
          traceback(base.traceback) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            this->traceback.push_back(TracebackItem(extension.pron, extension.timeframe + 1u, ScoreVector(extension.score, 0.0), {}));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            if (not this->traceback.empty()) {
                this->traceback.back().score.acoustic = extension.score;
                this->traceback.back().time           = extension.timeframe + 1u;
            }
            break;
    }
}

std::string LexiconfreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", label sequence: ";
    for (auto& item : traceback) {
        if (item.pronunciation != nullptr) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

/*
 * =======================
 * === TimeStatistic =====
 * =======================
 */

void LexiconfreeTimesyncBeamSearch::TimeStatistic::reset() {
    total = 0.0;
}

void LexiconfreeTimesyncBeamSearch::TimeStatistic::tic() {
    startTime = std::chrono::steady_clock::now();
}

void LexiconfreeTimesyncBeamSearch::TimeStatistic::toc() {
    auto endTime = std::chrono::steady_clock::now();
    // Duration in milliseconds
    total += std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(endTime - startTime).count();
}

double LexiconfreeTimesyncBeamSearch::TimeStatistic::getTotalMilliseconds() const {
    return total;
}
}  // namespace Search
