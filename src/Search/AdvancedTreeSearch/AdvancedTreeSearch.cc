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
// $Id: ConditionedTreeSearch.cc 7090 2009-04-03 11:17:55Z nolden $

#include "AdvancedTreeSearch.hh"

#include <Core/Hash.hh>
#include <Core/Statistics.hh>
#include <Core/Utility.hh>
#include <Lattice/Lattice.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Search/StateTree.hh>
#include <vector>
#include "SearchSpace.hh"
#include "SearchSpaceStatistics.hh"

#ifdef MODULE_LM_FSA
#include <Lm/FsaLm.hh>
#endif

using namespace Search;
using Core::Ref;
using Core::tie;

// ===========================================================================
// Bookkeeping

void AdvancedTreeSearchManager::traceback(Ref<Trace> end, SearchAlgorithm::Traceback& result, Ref<Trace> boundary) const {
    result.clear();
    for (; end && end != boundary; end = end->predecessor) {
        result.push_back(*end);
    }
    std::reverse(result.begin(), result.end());
}

// ===========================================================================
// class ConditionedTreeSearch

const Core::ParameterInt paramStartTreesInterval(
        "start-trees-interval",
        "only compute word-ends at start new trees at a specific interval (word boundaries in the traceback will be also aligned to that interval, thus slightly losing precision)",
        1);

const Core::ParameterFloat paramOnlineSegmentationLength(
        "online-segmentation-length",
        "",
        Core::Type<f32>::max, 0);

const Core::ParameterFloat paramOnlineSegmentationMargin(
        "online-segmentation-margin",
        "",
        0.2, 0);

const Core::ParameterFloat paramOnlineSegmentationTolerance(
        "online-segmentation-tolerace",
        "",
        0.7, 0, 1.0);

const Core::ParameterBool paramOnlineSegmentationIncludeGap(
        "online-segmentation-include-gap",
        "",
        true);

const Core::ParameterInt paramCleanupInterval(
        "cleanup-interval",
        "",
        10);

const Core::ParameterBool paramCreateLattice(
        "create-lattice",
        "enable generation of word lattice",
        false);

const Core::ParameterBool paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "allow for fallback solution if no active sentence end hypothesis exists",
        true);

const Core::Choice latticeOptimizationMethodChoice(
        "no", AdvancedTreeSearchManager::noLatticeOptimization,              // for backward compatibility - remove asap
        "yes", AdvancedTreeSearchManager::simpleSilenceLatticeOptimization,  // for backward compatibility - remove asap
        "none", AdvancedTreeSearchManager::noLatticeOptimization,
        "simple", AdvancedTreeSearchManager::simpleSilenceLatticeOptimization,
        Core::Choice::endMark());

const Core::ParameterChoice paramOptimizeLattice(
        "optimize-lattice",
        &latticeOptimizationMethodChoice,
        "optimization method for word lattice generation (dafault is 'simple silence approximation')",
        AdvancedTreeSearchManager::simpleSilenceLatticeOptimization);

const Core::ParameterFloat paramFrameShift(
        "frame-shift",
        "length of the frame shift in milliseconds",
        10.0,
        0.0);

AdvancedTreeSearchManager::AdvancedTreeSearchManager(
        const Core::Configuration& c)
        : Core::Component(c),
          SearchAlgorithm(c),
          silence_(0),
          wpScale_(0),
          ss_(0),
          statisticsChannel_(config, "statistics") {
    shallCreateLattice_       = paramCreateLattice(config);
    allowSentenceEndFallBack_ = paramSentenceEndFallBack(config);
    startTreesInterval_       = paramStartTreesInterval(config);
    cleanupInterval_          = paramCleanupInterval(config);
    onlineSegmentationLength_ = paramOnlineSegmentationLength(config);
    if (onlineSegmentationLength_ != Core::Type<f32>::max)
        log() << "Performing online-segmentation with length " << onlineSegmentationLength_;
    onlineSegmentationMargin_     = paramOnlineSegmentationMargin(config);
    onlineSegmentationTolerance_  = paramOnlineSegmentationTolerance(config);
    onlineSegmentationIncludeGap_ = paramOnlineSegmentationIncludeGap(config);
    frameShift_                   = paramFrameShift(config);

    if (shallCreateLattice_) {
        shallOptimizeLattice_ = LatticeOptimizationMethod(paramOptimizeLattice(config));
    }
    else {
        shallOptimizeLattice_ = noLatticeOptimization;
    }
}

void AdvancedTreeSearchManager::setGrammar(Fsa::ConstAutomatonRef g) {
    log("Set grammar");
#ifdef MODULE_LM_FSA
    require(lm_);
    const Lm::FsaLm* constFsaLm = dynamic_cast<const Lm::FsaLm*>(lm_->unscaled().get());
    require(constFsaLm);
    Lm::FsaLm* fsaLm = const_cast<Lm::FsaLm*>(constFsaLm);
    fsaLm->setFsa(g);
#else
    criticalError("Module LM_FSA is not available");
#endif

    delete ss_;
    ss_ = nullptr;
}

void AdvancedTreeSearchManager::setAllowHmmSkips(bool allow) {
    ss_->setAllowHmmSkips(allow);
}

bool AdvancedTreeSearchManager::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    silence_       = lexicon_->specialLemma("silence");
    acousticModel_ = modelCombination.acousticModel();
    lm_            = modelCombination.languageModel();
    wpScale_       = modelCombination.pronunciationScale();
    verify(!ss_);  // Changing the model-combination with an active search space is not supported ATM

    // Initialize the search space now, so that it won't be initialized at segment-start, because that
    // may alter the measured real-time factor.
    restart();

    return true;
}

void AdvancedTreeSearchManager::restart() {
    if (!ss_) {
        verify(lexicon_);  // setModelCombination must have been called already

        ss_ = new SearchSpace(config, acousticModel_, lexicon_, lm_, wpScale_);
        ss_->initialize();
        dynamicBeamPruningStrategy_ = createDynamicBeamPruningStrategy(select("dynamic-beam-pruning-strategy"), ss_->describePruning());
    }
    else {
        ss_->clear();
    }

    time_                = 0;
    currentSegmentStart_ = 0;
    ss_->addStartupWordEndHypothesis(time_);
    ss_->hypothesizeEpsilonPronunciations(0);
    sentenceEnd_.reset();
    if (dynamicBeamPruningStrategy_) {
        auto new_pruning = dynamicBeamPruningStrategy_->startNewSegment();
        if (new_pruning) {
            ss_->resetPruning(new_pruning);
        }
    }
    segmentStartTime_ = std::chrono::steady_clock::now();
}

void AdvancedTreeSearchManager::setSegment(Bliss::SpeechSegment const* segment) {
    const_cast<Lm::ScaledLanguageModel*>(lm_.get())->setSegment(segment);
}

bool AdvancedTreeSearchManager::shouldComputeWordEnds() {
    if (ss_->nActiveTrees() == 0)
        return true;

    return (time_ % startTreesInterval_) == 0;
}

void AdvancedTreeSearchManager::feed(const Mm::FeatureScorer::Scorer& emissionScores) {
    require(emissionScores->nEmissions() >= acousticModel_->nEmissions());
    sentenceEnd_.reset();

    PerformanceCounter perf(*ss_->statistics, "feed");
    auto               feed_start = std::chrono::steady_clock::now();

    // Set current timeframe, compute acoustic look-ahead
    ss_->setCurrentTimeFrame(time_, emissionScores);

    if (shouldComputeWordEnds())
        ss_->startNewTrees();

    // Expand HMM
    ss_->expandHmm();

    ++time_;

    ss_->pruneAndAddScores();

    if (time_ % cleanupInterval_ == 0 || ss_->needCleanup()) {
        //We have to rescale before activating the word ends
        ss_->rescale(ss_->bestScore());
        ss_->cleanup();
    }

    if (shouldComputeWordEnds()) {
        /// Handle word-ends
        ss_->findWordEnds();

        ss_->statistics->wordEndsBeforePruning += ss_->nEarlyWordEndHypotheses();

        // Standard word end pruning
        ss_->pruneEarlyWordEnds();

        ss_->createTraces(time_);

        // Recombine traces according to their reduced history
        ss_->recombineWordEnds(shallCreateLattice_);

        ss_->hypothesizeEpsilonPronunciations(ss_->minimumWordEndScore());

        switch (shallOptimizeLattice_) {
            case noLatticeOptimization:
                break;
            case simpleSilenceLatticeOptimization:
                ss_->optimizeSilenceInWordLattice(silence_);
                break;
            default: defect();
        }

        ss_->statistics->wordEndsAfterRecombination += ss_->nWordEndHypotheses();
    }
    else {
        ss_->statistics->wordEndsAfterRecombination += 0;
        ss_->statistics->wordEndsAfterPruning += 0;
        ss_->statistics->epsilonWordEndsAdded += 0;
        ss_->statistics->wordEndsBeforePruning += 0;
        ss_->statistics->wordEndsAfterSecondPruning += 0;
    }

    if (dynamicBeamPruningStrategy_) {
        auto feed_end       = std::chrono::steady_clock::now();
        f64  frame_duration = std::chrono::duration<f64, std::milli>(feed_end - feed_start).count();
        f64  delay          = std::chrono::duration<f64, std::milli>(feed_end - segmentStartTime_).count() - time_ * frameShift_;
        dynamicBeamPruningStrategy_->frameFinished(time_, frame_duration, delay);
        auto new_pruning = dynamicBeamPruningStrategy_->newPruningThresholds();
        if (new_pruning) {
            ss_->resetPruning(new_pruning);
        }
    }
}

Ref<Trace> AdvancedTreeSearchManager::getCorrectedCommonPrefix() {
    Core::Ref<Trace> t = ss_->getCommonPrefix();
    if (t->pronunciation == epsilonLemmaPronunciation())
        t = t->predecessor;
    mergeEpsilonTraces(t);
    return t;
}

void AdvancedTreeSearchManager::getPartialSentence(Traceback& result) {
    Ref<Trace> t = getCorrectedCommonPrefix();
    traceback(t, result, lastPartialTrace_);
    lastPartialTrace_ = t;
}

struct Correction {
    Correction(const Trace* _trace = 0, int _timeOffset = 0, Score _scoreOffset = 0, Lattice::WordBoundary::Transit _transit = Lattice::WordBoundary::Transit())
            : trace(_trace), timeOffset(_timeOffset), scoreOffset(_scoreOffset), transit(_transit) {}
    const Trace*                   trace;
    int                            timeOffset;
    Score                          scoreOffset;
    Lattice::WordBoundary::Transit transit;
    bool                           operator==(const Correction& rhs) const {
        return trace == rhs.trace && timeOffset == rhs.timeOffset && scoreOffset == rhs.scoreOffset && transit == rhs.transit;
    }

    template<class Key>
    struct StandardHash {
        inline u32 operator()(Key a) {
            a = (a ^ 0xc761c23c) ^ (a >> 19);
            a = (a + 0xfd7046c5) + (a << 3);
            return a;
        }
    };

    struct Hash {
        size_t operator()(const Correction& correction) const {
            return StandardHash<size_t>()(correction.transit.final << 16 + correction.transit.initial +
                                                                              StandardHash<size_t>()(reinterpret_cast<size_t>(correction.trace) +
                                                                                                     StandardHash<int>()(correction.timeOffset +
                                                                                                                         StandardHash<int>()(reinterpret_cast<const int&>(correction.scoreOffset)))));
        }
    };
};

Ref<Trace> AdvancedTreeSearchManager::sentenceEnd() const {
    if (ss_->nWordEndHypotheses() == 0 && startTreesInterval_ > 1) {
        ss_->findWordEnds();
        // pruneWordEnds is required to transform _early_ word-end-hypotheses into real word-end-hypotheses
        ss_->pruneEarlyWordEnds();
        ss_->createTraces(time_);
    }

    if (!sentenceEnd_) {
        sentenceEnd_ = ss_->getSentenceEnd(time_ + 1, shallCreateLattice_);
        if (!sentenceEnd_) {
            warning("No active word end hypothesis at sentence end.");
            if (allowSentenceEndFallBack_) {
                sentenceEnd_ = ss_->getSentenceEndFallBack(time_ + 1, shallCreateLattice_);
            }
        }
        if (sentenceEnd_) {
            switch (shallOptimizeLattice_) {
                case noLatticeOptimization:
                case simpleSilenceLatticeOptimization:
                    break;
                default: defect();
            }
        }

        // Eventually log the path-trace
        Ref<Trace> current = sentenceEnd_;
        while (current) {
            current->pathTrace.log(*this, current->pronunciation != epsilonLemmaPronunciation() ? current->pronunciation : current->predecessor->pronunciation);
            current = current->predecessor;
        }
    }

    mergeEpsilonTraces(sentenceEnd_);

    return sentenceEnd_;
}

void AdvancedTreeSearchManager::mergeEpsilonTraces(Ref<Trace> trace) const {
    verify(trace->pronunciation != epsilonLemmaPronunciation());
    if (!trace->predecessor)
        return;
    // Merge correcting epsilon trace entries into their corresponding predecessor trace entries,
    // to create a valid lattice.
    typedef std::unordered_map<Correction, Ref<Trace>, Correction::Hash> CorrectionHash;
    CorrectionHash                                                       corrections;
    std::stack<Ref<Trace>>                                               stack;
    stack.push(trace);

    std::unordered_map<const Trace*, bool, Core::conversion<const Trace*, size_t>> visited;
    while (!stack.empty()) {
        trace = stack.top();
        stack.pop();
        if (visited.count(trace.get()))
            continue;
        for (Ref<Trace> arcTrace = trace; arcTrace; arcTrace = arcTrace->sibling) {
            Ref<Trace> temp = arcTrace;  // For security
            verify(arcTrace);
            visited[arcTrace.get()] = true;
            verify(arcTrace->predecessor);  // We don't put traces without predecessor onto the stack
            if (arcTrace->pronunciation == epsilonLemmaPronunciation()) {
                // Apply to predecessor
                verify(arcTrace->predecessor);
                verify(!arcTrace->sibling);
                Correction correction(arcTrace->predecessor.get(),
                                      arcTrace->time - arcTrace->predecessor->time,
                                      arcTrace->score.acoustic - arcTrace->predecessor->score.acoustic,
                                      arcTrace->transit);
                verify(correction.timeOffset >= 0);
                verify(corrections.find(correction) == corrections.end() || corrections[correction] == arcTrace);
                corrections[correction] = arcTrace;
                Ref<Trace> oldpre       = arcTrace->predecessor;
                *arcTrace               = *oldpre;

                Core::Ref<Trace> currentTrace = arcTrace;
                while (currentTrace) {
                    verify(currentTrace->pronunciation != epsilonLemmaPronunciation());
                    currentTrace->score.acoustic += correction.scoreOffset;
                    currentTrace->time += correction.timeOffset;
                    currentTrace->transit = correction.transit;
                    if (currentTrace->sibling) {
                        currentTrace->sibling = Core::Ref<Trace>(new Trace(*currentTrace->sibling));
                        currentTrace          = currentTrace->sibling;
                    }
                    else {
                        currentTrace = Core::Ref<Trace>();
                    }
                }
            }
            verify(arcTrace->pronunciation != epsilonLemmaPronunciation());
            Ref<Trace> preTrace = arcTrace->predecessor;
            if (preTrace->predecessor) {
                if (preTrace->pronunciation == epsilonLemmaPronunciation()) {
                    // Share another correction
                    Correction                     correction(preTrace->predecessor.get(),
                                          preTrace->time - preTrace->predecessor->time,
                                          preTrace->score.acoustic - preTrace->predecessor->score.acoustic,
                                          preTrace->transit);
                    CorrectionHash::const_iterator it = corrections.find(correction);
                    if (it != corrections.end()) {
                        preTrace = arcTrace->predecessor = it->second;
                    }
                    else {
                        corrections[correction] = preTrace;
                    }
                }

                if (visited.find(preTrace.get()) == visited.end()) {
                    stack.push(preTrace);
                }
            }
            else {
                verify(preTrace->pronunciation != epsilonLemmaPronunciation());
            }
        }
    }
}

void AdvancedTreeSearchManager::getCurrentBestSentence(Traceback& result) const {
    Ref<Trace> t = sentenceEnd();
    if (!t) {
        error("Cannot determine sentence hypothesis: No active word end hypothesis.");
        result.clear();
        return;
    }
    traceback(t, result);
}

void AdvancedTreeSearchManager::getCurrentBestSentencePartial(SearchAlgorithm::Traceback& result) const {
    Ref<Trace> t = sentenceEnd();
    if (!t) {
        result.clear();
        return;
    }
    traceback(t, result, lastPartialTrace_);
}

Core::Ref<const LatticeAdaptor> AdvancedTreeSearchManager::buildLatticeForTrace(Ref<Trace> trace) const {
    if (!trace) {
        warning("Cannot create word lattice.");
        // should not abort, as it can be fixed by incremental decoding
        // just create a dummy lattice with only an epsilon arc and correct length
        Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
        Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);
        wordBoundaries->set((Fsa::StateId) /* WARNING: this looses a few bits */ (long)result->initialState(), Lattice::WordBoundary(0));
        Fsa::State* finalState = result->newState();
        wordBoundaries->set(finalState->id(), Lattice::WordBoundary(time_));
        result->newArc(result->initialState(), finalState, (const Bliss::LemmaPronunciation*)0, 0, 0);
        result->setWordBoundaries(wordBoundaries);
        result->addAcyclicProperty();
        return Core::ref(new Lattice::WordLatticeAdaptor(result));
    }

    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);
    Ref<Trace>                              initialTrace;

    std::unordered_map<const Trace*, Fsa::State*, Core::conversion<const Trace*, size_t>> state;
    state[trace.get()] = result->finalState();
    std::stack<Ref<Trace>> stack;
    stack.push(trace);

    Fsa::State *previousState, *currentState;
    while (!stack.empty()) {
        trace = stack.top();
        stack.pop();
        currentState = state[trace.get()];
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(trace->time, trace->transit));

        for (Ref<Trace> arcTrace = trace; arcTrace; arcTrace = arcTrace->sibling) {
            verify(arcTrace->pronunciation != epsilonLemmaPronunciation());
            Ref<Trace> preTrace = arcTrace->predecessor;
            if (preTrace->predecessor) {
                if (state.find(preTrace.get()) == state.end()) {
                    previousState = state[preTrace.get()] = result->newState();
                    stack.push(preTrace);
                }
                else {
                    previousState = state[preTrace.get()];
                }
            }
            else {
                previousState = result->initialState();
                initialTrace  = preTrace;
            }

            result->newArc(previousState, currentState, arcTrace->pronunciation,
                           (arcTrace->score.acoustic - preTrace->score.acoustic),
                           (arcTrace->score.lm - preTrace->score.lm));
        }
    }
    verify(initialTrace);
    wordBoundaries->set(result->initialState()->id(),
                        Lattice::WordBoundary(initialTrace->time, initialTrace->transit));
    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();
    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

Core::Ref<const LatticeAdaptor> AdvancedTreeSearchManager::getCurrentWordLattice() const {
    return buildLatticeForTrace(sentenceEnd());
}

Core::Ref<const LatticeAdaptor> AdvancedTreeSearchManager::getPartialWordLattice() {
    Core::Ref<const LatticeAdaptor> ret;
    if ((time_ % 100) == 0 && time_ - currentSegmentStart_ > (onlineSegmentationLength_ * 1.0 / onlineSegmentationTolerance_) * 100) {
        Ref<Trace> t = getCorrectedCommonPrefix();

        std::vector<std::pair<Ref<Trace>, Ref<Trace>>> gaps;
        Ref<Trace>                                     current = t;
        bool                                           isGap   = false;
        while (current) {
            if (current->pronunciation && !pronunciationHasEvaluationTokens(current->pronunciation)) {
                if (isGap) {
                    gaps.back().first = current;
                }
                else {
                    isGap = true;
                    gaps.push_back(std::make_pair(current, current));
                }
            }
            else {
                isGap = false;
            }
            current = current->predecessor;
        }
        log() << "online segmentation: found " << gaps.size() << " gaps between " << currentSegmentStart_ << " and " << time_;

        f32                               bestGapLength = 0;
        std::pair<Ref<Trace>, Ref<Trace>> bestGap;
        for (std::vector<std::pair<Ref<Trace>, Ref<Trace>>>::const_iterator it = gaps.begin(); it != gaps.end(); ++it) {
            f32 startTime = it->first->predecessor->time;
            f32 endTime   = it->second->time;
            f32 length    = endTime - startTime;
            if (length >= onlineSegmentationMargin_ * 100 && startTime > onlineSegmentationTolerance_ * onlineSegmentationLength_ * 100) {
                log() << "online segmentation: found acceptable gap: " << startTime << " -> " << endTime << " (" << endTime - startTime << ")";
                if (length > bestGapLength) {
                    bestGapLength = length;
                    bestGap       = *it;
                }
            }
        }

        if (bestGapLength) {
            log() << "online segmentation: using gap: " << bestGap.first->predecessor->time
                  << " -> " << bestGap.second->time << " (" << bestGap.second->time - bestGap.first->predecessor->time << ")";
            ret = buildLatticeForTrace(onlineSegmentationIncludeGap_ ? bestGap.first : bestGap.first->predecessor);

            Ref<Trace> newInitialTrace;
            if (bestGap.second != bestGap.first && onlineSegmentationIncludeGap_) {
                newInitialTrace = bestGap.second->predecessor;
            }
            else {
                newInitialTrace = bestGap.second;
            }

            ss_->changeInitialTrace(newInitialTrace);

            currentSegmentStart_ = newInitialTrace->time;
        }
    }

    return ret;
}

void AdvancedTreeSearchManager::resetStatistics() {
    ss_->resetStatistics();
}

void AdvancedTreeSearchManager::logStatistics() const {
    if (statisticsChannel_.isOpen())
        ss_->logStatistics(statisticsChannel_);
}

AdvancedTreeSearchManager::~AdvancedTreeSearchManager() {
    delete ss_;
}

u32 AdvancedTreeSearchManager::lookAheadLength() {
    return ss_->lookAheadLength();
}

Search::SearchAlgorithm::RecognitionContext AdvancedTreeSearchManager::setContext(RecognitionContext context) {
    return ss_->setContext(context);
}

void AdvancedTreeSearchManager::setLookAhead(const std::vector<Mm::FeatureVector>& lookahead) {
    if ((int)lookahead.size() < lookAheadLength())
        // Disable acoustic look-ahead if we don't have enough data
        ss_->setLookAhead(std::vector<Mm::FeatureVector>());
    else
        ss_->setLookAhead(lookahead);
}

bool AdvancedTreeSearchManager::relaxPruning(f32 factor, f32 offset) {
    return ss_->relaxPruning(factor, offset);
}

void AdvancedTreeSearchManager::resetPruning(SearchAlgorithm::PruningRef pruning) {
    ss_->resetPruning(pruning);
}

SearchAlgorithm::PruningRef AdvancedTreeSearchManager::describePruning() {
    return ss_->describePruning();
}
