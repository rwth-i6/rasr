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

#include <algorithm>
#include <strings.h>

#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Traceback.hh>

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
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Core::Type<int>::max);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramAllowLabelLoop(
        "allow-label-loop",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

LexiconfreeTimesyncBeamSearch::LexiconfreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug", Core::Channel::standard),
          labelScorer_(),
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          recombinedHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numHypsAfterScorePruning_("num-hyps-after-score-pruning"),
          numHypsAfterBeamPruning_("num-hyps-after-beam-pruning"),
          numActiveHyps_("num-active-hyps") {
    beam_.reserve(maxBeamSize_);
    extensions_.reserve(maxBeamSize_ * lexicon_->nLemmas());
    newBeam_.reserve(maxBeamSize_);
    requests_.reserve(extensions_.size());
    recombinedHypotheses_.reserve(maxBeamSize_);
    useBlank_ = blankLabelIndex_ != Core::Type<int>::max;
    if (useBlank_) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
}

void LexiconfreeTimesyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();

    initializationTime_.stop();
}

Speech::ModelCombination::Mode LexiconfreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    auto blankLemma = lexicon_->specialLemma("blank");
    if (blankLemma) {
        if (blankLabelIndex_ == Core::Type<int>::max) {
            blankLabelIndex_ = blankLemma->id();
            useBlank_        = true;
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id() << " but is overwritten by config parameter with value " << blankLabelIndex_;
        }
    }

    reset();
    return true;
}

void LexiconfreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
}

void LexiconfreeTimesyncBeamSearch::putFeature(std::shared_ptr<const f32[]> const& data, size_t featureSize) {
    featureProcessingTime_.start();
    labelScorer_->addInput(data, featureSize);
    featureProcessingTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::putFeature(std::vector<f32> const& data) {
    featureProcessingTime_.start();
    labelScorer_->addInput(data);
    featureProcessingTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::putFeatures(std::shared_ptr<const f32[]> const& data, size_t timeSize, size_t featureSize) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(data, timeSize, featureSize);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LexiconfreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp          = beam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

void LexiconfreeTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numHypsAfterScorePruning_.clear();
    numHypsAfterBeamPruning_.clear();
    numActiveHyps_.clear();
}

void LexiconfreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsAfterScorePruning_.write(clog());
    numHypsAfterBeamPruning_.write(clog());
    numActiveHyps_.write(clog());
}

Nn::LabelScorer::TransitionType LexiconfreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = (useBlank_ and prevLabel == blankLabelIndex_);
    bool nextIsBlank = (useBlank_ and nextLabel == blankLabelIndex_);

    if (prevLabel == Core::Type<Nn::LabelIndex>::max) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::INITIAL_BLANK;
        }
        else {
            return Nn::LabelScorer::TransitionType::INITIAL_LABEL;
        }
    }

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

    // Reorder the hypotheses by associated score value such that the first `beamSize_` elements are the best
    std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize_, extensions.end());
    extensions.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeTimesyncBeamSearch::scorePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions) const {
    if (extensions.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions.begin(), extensions.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold_;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());
}

void LexiconfreeTimesyncBeamSearch::recombination(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    recombinedHypotheses_.clear();
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<Nn::ScoringContextRef, LabelHypothesis*, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace(hyp.scoringContext, nullptr);

        if (inserted) {
            // First time seeing this scoring context so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
        }
        else {
            verify(not hyp.trace->sibling);

            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better -> replace in `newHypotheses` and add existing one as sibling
                hyp.trace->sibling = existingHyp->trace;
                *existingHyp       = std::move(hyp);  // Overwrite in-place
            }
            else {
                // New hyp is worse -> add to existing one as sibling
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(recombinedHypotheses_);
}

bool LexiconfreeTimesyncBeamSearch::decodeStep() {
    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    /*
     * Collect all possible extensions for all hypotheses in the beam.
     */
    extensions_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            extensions_.push_back(
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
    requests_.clear();
    for (const auto& extension : extensions_) {
        requests_.push_back({beam_[extension.baseHypIndex].scoringContext, extension.nextToken, extension.transitionType});
    }

    /*
     * Perform scoring of all the requests with the label scorer.
     */
    scoringTime_.start();
    auto result = labelScorer_->computeScoresWithTimes(requests_);
    scoringTime_.stop();

    if (not result) {
        // LabelScorer could not compute scores -> no search step can be made.
        return false;
    }

    for (size_t requestIdx = 0ul; requestIdx < extensions_.size(); ++requestIdx) {
        extensions_[requestIdx].score += result->scores[requestIdx];
        extensions_[requestIdx].timeframe = result->timeframes[requestIdx];
    }

    /*
     * Prune set of possible extensions by max beam size and possibly also by score.
     */

    if (useScorePruning_) {
        scorePruning(extensions_);

        numHypsAfterScorePruning_ += extensions_.size();

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-score-pruning", extensions_.size());
        }
    }

    beamPruning(extensions_);
    numHypsAfterBeamPruning_ += extensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", extensions_.size());
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeam_.clear();

    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});

        newBeam_.push_back({baseHyp, extension, newScoringContext});
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination(newBeam_);
    numActiveHyps_ += newBeam_.size();

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", newBeam_.size());
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < newBeam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << newBeam_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    beam_.swap(newBeam_);

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        clog() << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        clog() << Core::XmlClose("search-step-stats");
    }

    return true;
}

LexiconfreeTimesyncBeamSearch::LabelHypothesis const& LexiconfreeTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LexiconfreeTimesyncBeamSearch::LabelHypothesis const& LexiconfreeTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
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
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeTimesyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                             newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          score(extension.score),
          trace() {
    switch (extension.transitionType) {
        case Nn::LabelScorer::INITIAL_BLANK:
        case Nn::LabelScorer::INITIAL_LABEL:
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            trace = Core::ref(new LatticeTrace(
                    base.trace,
                    extension.pron,
                    extension.timeframe + 1,
                    {extension.score, 0},
                    {}));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            // Copy base trace and update it
            trace                 = Core::ref(new LatticeTrace(*base.trace));
            trace->sibling        = {};
            trace->score.acoustic = extension.score;
            trace->time           = extension.timeframe + 1;
            break;
    }
}

std::string LexiconfreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

}  // namespace Search
