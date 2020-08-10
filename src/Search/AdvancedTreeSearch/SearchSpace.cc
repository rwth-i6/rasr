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
#include "SearchSpace.hh"

#include <chrono>
#include <random>

#include <Am/ClassicAcousticModel.hh>
#include <Core/MappedArchive.hh>
#include <Lm/BackingOff.hh>
#include <Lm/Module.hh>
#include <Mm/GaussDiagonalMaximumFeatureScorer.hh>

#include "AcousticLookAhead.hh"
#include "PersistentStateTree.hh"
#include "PrefixFilter.hh"
#include "Pruning.hh"
#include "SearchNetworkTransformation.hh"
#include "SearchSpaceStatistics.hh"
#include "TreeBuilder.hh"

using namespace AdvancedTreeSearch;

enum {
    ForbidSecondOrderExpansion = 1
};

namespace Search {
/// Pruning (NOTE: pruning values are relative to the lm-scale, if not noted otherwise):

static const f32 defaultBeamPruning = 12.0;

const Core::ParameterFloat paramBeamPruning(
        "beam-pruning",
        "Beam size used for pruning state hypotheses, relative to the lm-scale. \
   Typically between 8 and 12. Default: 12.0 if nothing else is set.",
        Core::Type<f32>::max, 0);

const Core::ParameterBool paramHistogramIsMasterPruning(
        "histogram-pruning-is-master-pruning",
        "",
        false);

const Core::ParameterFloat paramAcousticPruning(
        "acoustic-pruning",
        "DEPRECATED: Use beam-pruning instead! (difference: beam-pruning is relative to the lm-scale, while this is not)",
        Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramBeamPruningLimit(
        "beam-pruning-limit",
        "maximum number of active states, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime.",
        500000, 1);

const Core::ParameterInt paramAcousticPruningLimit(
        "acoustic-pruning-limit",
        "DEPRECATED: Use beam-pruning-limit instead.",
        Core::Type<s32>::max, 1);

const Core::ParameterFloat paramWordEndPruning(
        "word-end-pruning",
        "threshold for pruning of word end hypotheses \
   If the value is below 1.0, eg. 0.7, then it is relative to acoustic-pruning (recommended).",
        Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramWordEndPruningLimit(
        "word-end-pruning-limit",
        "maximum number of word ends, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime \
   20000 is a good default value, reduce it more if the runtime becomes too slow for some segments.",
        Core::Type<s32>::max, 1);

const Core::ParameterFloat paramLmPruning(
        "lm-pruning",
        "DEPRECATED: Use word-end-pruning instead (difference: word-end-pruning is relative to the lm-scale, while this value is absolute).",
        Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramLmPruningLimit(
        "lm-pruning-limit",
        "DEPRECATED: Use word-end-pruning-limit instead.",
        Core::Type<s32>::max, 1);

const Core::ParameterFloat lmStatePruning(
        "lm-state-pruning",
        "pruning that is applied to all state hypotheses which are on the same state in the prefix network (can be lower than lm-pruning) \
   If the value is below one, eg. 0.7, then it is relative to word-end-pruning (recommended). \
   This pruning is effective only if the search network is minimized (eg. build-minimized-tree-from-scratch=true and min-phones <= 1)",
        Core::Type<f32>::max);

const Core::ParameterFloat paramAcousticLookaheadTemporalApproximationScale(
        "acoustic-lookahead-temporal-approximation-scale",
        "scaling factor of temporal acoustic look-ahead (1.5 is a good value)",
        0);

const Core::ParameterFloat paramPerInstanceAcousticPruningScale(
        "per-instance-acoustic-pruning-scale",
        "when using per instance pruning thresholds the pruning threshold is the beam-pruning threshold times this scale",
        1.0);

const Core::ParameterFloat paramEarlyWordEndPruningMinimumLmScore(
        "early-word-end-pruning-minimum-lm-score",
        "expected lm-score that will be used for early word-end pruning (safe if it is always lower than the real score)",
        0);

const Core::ParameterFloat paramWordEndPhonemePruningThreshold(
        "word-end-phoneme-pruning",
        "pruning applied to word ends which have the same final phoneme (relative to word-end-pruning if the value is below 1.0)",
        Core::Type<Score>::max);

const Core::ParameterInt paramWordEndPruningFadeInInterval(
        "word-end-pruning-fadein",
        "inverted depth at which the lm pruning influence reaches zero",
        0, 0, MaxFadeInPruningDistance);

/// Internal parameters (with good default-values)

const Core::ParameterBool paramBuildMinimizedTreeFromScratch(
        "build-minimized-network-from-scratch",
        "",
        true);

const Core::ParameterBool paramConditionPredecessorWord(
        "condition-on-predecessor-word",
        "",
        false);

const Core::ParameterBool paramDecodeMesh(
        "decode-mesh",
        "produce a mesh-like reduced lattice, which can later be expanded by mesh-construction and lattice-decoding to the full search space",
        false);

const Core::ParameterInt paramDecodeMeshPhones(
        "decode-mesh-phones",
        "-1 means full pronunciation, eg. word pair approximation. 0 means no context. otherwise number of condition phones.",
        -1, -1);

const Core::ParameterBool paramEnableLmLookahead(
        "lm-lookahead",
        "enable language model lookahead (recommended)",
        true);

const Core::ParameterBool paramSeparateLookaheadLm(
        "separate-lookahead-lm",
        "use a separate lm for lookahead (one that is not provided by the main language-model)",
        false);

const Core::ParameterBool paramSeparateRecombinationLm(
        "separate-recombination-lm",
        "use a separate lm for recombination (one that is not provided by the main language-model)",
        false);

const Core::ParameterBool paramDisableUnigramLookahead(
        "disable-unigram-lookahead",
        "",
        false);

const Core::ParameterBool paramSparseLmLookAhead(
        "sparse-lm-lookahead",
        "use sparse n-gram LM look-ahead (recommended)",
        true);

const Core::ParameterBool paramSymmetrizePenalties(
        "symmetrize-penalties",
        "",
        false);

const Core::ParameterInt paramReduceLookAheadBeforeDepth(
        "full-lookahead-min-depth",  // minor effect
        "only apply unigram lookahead for states that have a lookahead-network-depth lower than this. negative values allow considering the pushed fan-out.",
        0);

const Core::ParameterInt paramReduceLookAheadStateMinimum(
        "full-lookahead-min-states",
        "apply full lookahead in instances that more than this number of active states",
        0);

const Core::ParameterFloat paramReduceLookAheadDominanceMinimum(
        "full-lookahead-min-dominance",
        "apply full-order lookahead in instances that have at least this dominance",
        0.05);

const Core::ParameterBool paramEarlyBeamPruning(
        "early-beam-pruning",
        "Whether beam pruning should already be performed before computing the acoustic scores, but after look-ahead scores have been applied.",
        true);

const Core::ParameterBool paramEarlyWordEndPruning(
        "early-word-end-pruning",
        "enable earlier pruning of word-ends during the recombiniation",
        true);

const Core::ParameterBool paramReducedContextWordRecombination(
        "reduced-context-word-recombination",
        "reduce the context of word-end hypotheses before recombination",
        false);

const Core::ParameterInt paramReducedContextWordRecombinationLimit(
        "reduced-context-word-recombination-limit",
        "the maximum context length to consider when doing word combination",
        1, 0);

const Core::ParameterBool paramReducedContextTreeKey(
        "reduced-context-tree-key",
        "reduce the context of tree instance key (to reuse the tree)",
        false);

const Core::ParameterBool paramOnTheFlyRescoring(
        "on-the-fly-rescoring",
        "keep track of recombined histories and use those aswell when searching for word ends",
        false);

const Core::ParameterInt paramOnTheFlyRescoringMaxHistories(
        "on-the-fly-rescoring-history-limit",
        "what is the maximum number of alternative histories that should be kept",
        5, 0);

const Core::ParameterInt paramMaximumMutableSuffixLength(
        "maximum-mutable-suffix-length",
        "maximum length of words that are allowed to change",
        Core::Type<int>::max, 0);

const Core::ParameterInt paramMaximumMutableSuffixPruningInterval(
        "maximum-mutable-suffix-pruning-interval",
        "perform mutable-suffix-pruning every n frames",
        0, 0);

const Core::ParameterBool paramExtendedStatistics(
        "expensive-statistics",
        "add additional performance-wise expensive statistics",
        false);

const Core::ParameterBool paramEarlyBackOff(
        "early-backoff",
        "enable early backing-off right at the root states, as done in WFST based decoders (lazy dominance-based look-ahead activation is recommended, eg. for example full-lookahead-min-dominance=0.1)",
        false);

const Core::ParameterBool paramCorrectPushedWordBoundaryTimes(
        "correct-pushed-word-boundary-times",
        "correct the word boundary times that are changed through word-end pushing. Activate this if you want to generate alignments or similar",
        true);

const Core::ParameterBool paramCorrectPushedAcousticScores(
        "correct-pushed-acoustic-scores",
        "correct the acoustic scores that were changed through word-end pushing. Activate this if you need to compute confidence-scores or similar",
        true);

const Core::ParameterFloat paramUnigramLookaheadBackOffFactor(
        "unigram-lookahead-backoff-factor",
        "",
        0.0);

const Core::ParameterBool paramOverflowLmScoreToAm(
        "overflow-lm-score-to-am",
        "if the models can produce negative scores, then sometimes it can happen that an acoustic word score is negative in the lattice, thereby making the lattice invalid. \
   with this option, the acoustic score 'overflows' into the LM score, leading to a valid lattice with correct per-word scores, but with wrong score distribution \
   between AM/LM",
        false);

const Core::ParameterBool paramSparseLmLookaheadSlowPropagation(
        "sparse-lm-lookahead-slow-propagation",
        "prevent skipping multiple look-ahead n-gram order levels at the same timeframe (very minor effect)",
        false);

const Core::ParameterInt paramWordEndPruningBins(
        "word-end-pruning-bins",
        "number of bins for histogram pruning of word ends (very minor effect)",
        100, 2);

const Core::ParameterInt paramAcousticPruningBins(
        "acoustic-pruning-bins",
        "number of bins for histogram pruning of states (very minor effect)",
        100, 2);

const Core::ParameterInt paramInstanceDeletionLatency(
        "instance-deletion-latency",
        "timeframes of inactivity before an instance is deleted",
        3, 0);

const Core::ParameterString paramDumpDotGraph(
        "search-network-dump-dot-graph",
        "",
        "");

/// Special parameters for auto-correcting search:

const Core::ParameterBool paramEncodeStateInTrace(
        "encode-state-in-trace",
        "encode the network state in the boundary transition-information of lattices. this is only useful in auto-correcting search, and only actually used if lattice-generation is explicitly disabled in the recognition-context.",
        true);

const Core::ParameterBool paramEncodeStateInTraceAlways(
        "encode-state-in-trace-always",
        "",
        false);

const Core::ParameterFloat paramMinimumBeamPruning(
        "minimum-beam-pruning",
        "minimum beam pruning allowed during automatic tightening for auto-correcting search",
        2.0);

const Core::ParameterFloat paramMaximumBeamPruning(
        "maximum-beam-pruning",
        "maximum beam pruning allowed during automatic relaxation for auto-correcting search",
        100, 0);

const Core::ParameterInt paramMaximumAcousticPruningLimit(
        "maximum-beam-pruning-limit",
        "",
        250000, 1);

const Core::ParameterInt paramMinimumAcousticPruningLimit(
        "minimum-beam-pruning-limit",
        "",
        100, 1);

const Core::ParameterFloat paramMinimumWordLemmasAfterRecombination(
        "minimum-word-lemmas-after-recombination",
        "minimum number of average different observed word lemmas per timeframe to consider the search-space non-degenerated for auto-correcting search",
        0);

const Core::ParameterFloat paramMinimumStatesAfterPruning(
        "minimum-states-after-pruning",
        "minimum number of average states after pruning to consider the search-space non-degenerated for auto-correcting search (better: use minimum-word-lemmas-after-recombination)",
        50);

const Core::ParameterFloat paramMinimumWordEndsAfterPruning(
        "minimum-word-ends-after-pruning",
        "minimum number of average word ends after pruning to consider the search-space non-degenerated for auto-correcting search (better: use minimum-word-lemmas-after-recombination)",
        10);

const Core::ParameterFloat paramMaximumAcousticPruningSaturation(
        "maximum-acoustic-pruning-saturation",
        "maximum percentage of frames at which the acoustic-pruning-limit may be hit during auto-correcting search",
        0.5, 0.0, 0.9);

const Core::ParameterFloat paramMaximumStatesAfterPruning(
        "maximum-states-after-pruning",
        "maximum absolute number of states after pruning allowed during auto-correcting-search (better: use maximum-acoustic-pruning-saturation and acoustic-pruning-limit instead)",
        Core::Type<Score>::max);

const Core::ParameterFloat paramMaximumWordEndsAfterPruning(
        "maximum-word-ends-after-pruning",
        "maximum absolute number of word end hypotheses after pruning allowed during auto-correcting-search (better: use maximum-acoustic-pruning-saturation and acoustic-pruning-limit instead)",
        Core::Type<Score>::max);

namespace {
template<class From, class To>
void truncate(const std::vector<From>& source, std::vector<To>& to) {
    to.clear();
    for (typename std::vector<From>::const_iterator it = source.begin(); it != source.end(); ++it) {
        if (*it < Core::Type<To>::min)
            to.push_back(Core::Type<To>::min);
        else if (*it > Core::Type<To>::max)
            to.push_back(Core::Type<To>::max);
        else
            to.push_back(*it);
    }
}
}  // namespace

StaticSearchAutomaton::StaticSearchAutomaton(Core::Configuration config, Core::Ref<const Am::AcousticModel> acousticModel, Bliss::LexiconRef lexicon)
        : Precursor(config),
          hmmLength(acousticModel->hmmTopologySet()->getDefault().nPhoneStates() * acousticModel->hmmTopologySet()->getDefault().nSubStates()),
          minimized(paramBuildMinimizedTreeFromScratch(config)),
          network(config, acousticModel, lexicon),
          prefixFilter(nullptr),
          acousticModel_(acousticModel),
          lexicon_(lexicon) {
}

StaticSearchAutomaton::~StaticSearchAutomaton() {
    if (prefixFilter) {
        delete prefixFilter;
    }
}

void StaticSearchAutomaton::buildNetwork() {
    /// @todo Track the TreeBuilder configuration in transformation if minimizedTree
    int transformation = minimized ? 32 : 0;
    if (!network.read(transformation)) {
        log() << "persistent network image could not be loaded, building it";

        if (minimized) {  // Use TreeStructure.hh
            TreeBuilder builder(config, *lexicon_, *acousticModel_, network);
            builder.build();
        }
        else {  // Use StateTree.hh
            network.build();
            network.cleanup();
            network.cleanup();  // Additional cleanup, to make sure that the exits are ordered correctly
        }

        if (network.write(transformation)) {
            log() << "writing network image ready";
        }
        else {
            log() << "writing network image failed";
        }
    }
}

void StaticSearchAutomaton::buildDepths(bool onlyFromRoot) {
    clearDepths();
    stateDepths.resize(network.structure.stateCount(), Core::Type<int>::min);
    invertedStateDepths.resize(network.structure.stateCount(), Core::Type<int>::min);
    fillStateDepths(network.rootState, 0);
    fillStateDepths(network.ciRootState, 0);

    bool offsetted = false;

    if (!onlyFromRoot) {
        for (std::set<StateId>::const_iterator it = network.unpushedCoarticulatedRootStates.begin(); it != network.unpushedCoarticulatedRootStates.end(); ++it)
            fillStateDepths(*it, 0);

        for (StateId state = 1; state < network.structure.stateCount(); ++state) {
            findStateDepth(state);
        }

        for (std::set<StateId>::const_iterator it = network.coarticulatedRootStates.begin(); it != network.coarticulatedRootStates.end(); ++it) {
            int depth = findStateDepth(*it);

            if (depth < 0) {
                log() << "offsetting depths by " << depth;
                offsetted = true;
                for (u32 a = 1; a < stateDepths.size(); ++a)
                    if (stateDepths[a] != Core::Type<int>::min)
                        stateDepths[a] += -depth;
                depth = 0;
            }
            else if (depth == Core::Type<int>::max) {
                log() << "disconnected subnetwork found";
                depth = 0;
            }
            fillStateDepths(*it, depth);
        }

        if (!offsetted) {
            for (std::set<StateId>::const_iterator it = network.coarticulatedRootStates.begin(); it != network.coarticulatedRootStates.end(); ++it) {
                verify(stateDepths[*it] == 0);
            }
        }

        for (u32 a = 1; a < stateDepths.size(); ++a) {
            verify(stateDepths[a] != Core::Type<int>::min);
        }
    }

    // Verify the correctness of the depths
    for (u32 a = 1; a < stateDepths.size(); ++a) {
        if (stateDepths[a] != Core::Type<int>::min && stateDepths[a] != Core::Type<int>::max) {
            for (HMMStateNetwork::SuccessorIterator it = network.structure.successors(a); it; ++it) {
                if (!it.isLabel()) {
                    verify(stateDepths[*it] > stateDepths[a]);
                }
            }
        }
    }

    if (!offsetted) {
        verify(stateDepths[network.rootState] == 0);
    }

    truncate(invertedStateDepths, truncatedInvertedStateDepths);
    truncate(stateDepths, truncatedStateDepths);
}

void StaticSearchAutomaton::clearDepths() {
    stateDepths.clear();
    invertedStateDepths.clear();
}

int StaticSearchAutomaton::fillStateDepths(StateId state, int depth) {
    if (stateDepths[state] != Core::Type<int>::min) {
        if (stateDepths[state] != depth) {  /// @todo Find out why this happens on some languages
            std::cout << "conflicting state depths: " << stateDepths[state] << " vs " << depth << std::endl;
        }
        if (depth > stateDepths[state]) {
            stateDepths[state] = Core::Type<int>::min;  // Re-fill successor depths
        }
        else {
            return depth;
        }
    }

    stateDepths[state] = depth;

    int localDepth;

    localDepth = 0;

    for (HMMStateNetwork::SuccessorIterator it = network.structure.successors(state); it; ++it) {
        if (not it.isLabel()) {
            int d = fillStateDepths(*it, depth + 1);

            if (d > localDepth) {
                localDepth = d;
            }
        }
    }

    verify(localDepth != Core::Type<int>::max);

    invertedStateDepths[state] = localDepth;
    return localDepth + 1;
}

int StaticSearchAutomaton::findStateDepth(Search::StateId state) {
    if (stateDepths[state] != Core::Type<int>::min) {
        return stateDepths[state];
    }

    int nextDepth = Core::Type<int>::max;

    for (HMMStateNetwork::SuccessorIterator it = network.structure.successors(state); it; ++it) {
        if (not it.isLabel()) {
            int d = findStateDepth(*it);
            if (nextDepth == Core::Type<int>::max) {
                nextDepth = d;
            }
            else {
                if (d != nextDepth && d != Core::Type<int>::max) {
                    // This can happen when phones have inconsistent lengths,
                    // eg. if there are noise/silence phones within words
                    if (d < nextDepth) {
                        nextDepth = d;
                    }
                }
            }
        }
    }

    if (nextDepth != Core::Type<int>::max) {
        return nextDepth - 1;
    }
    else {
        return Core::Type<int>::max;
    }
}

void StaticSearchAutomaton::buildLabelDistances() {
    labelDistance.resize(network.structure.stateCount(), Core::Type<unsigned>::max);

    std::vector<StateId> toposort(network.structure.stateCount());
    std::iota(toposort.begin(), toposort.end(), 0u);
    std::sort(toposort.begin(), toposort.end(), [&](StateId a, StateId b) {
        return stateDepths[a] > stateDepths[b];
    });

    for (StateId s : toposort) {
        HMMState const& state = network.structure.state(s);

        for (auto successorIt = network.structure.successors(state); successorIt; ++successorIt) {
            if (successorIt.isLabel()) {
                Bliss::LemmaPronunciation::Id lemmaPronId = network.exits[successorIt.label()].pronunciation;
                if (lexicon_->lemmaPronunciation(lemmaPronId)->lemma()->syntacticTokenSequence().size() > 0) {
                    labelDistance[s] = 0u;
                    break;
                }
            }
            else {
                labelDistance[s] = std::min(labelDistance[s], labelDistance[*successorIt] + 1);
            }
        }
    }
}

void StaticSearchAutomaton::buildBatches() {
    bool symmetrize = paramSymmetrizePenalties(config);

    secondOrderEdgeSuccessorBatches.push_back(0);
    secondOrderEdgeSuccessorBatches.push_back(0);

    u32 validSecondOrderBatches = 0, invalidSecondOrderBatches = 0;
    u32 validFirstOrderBatches        = 0;
    u32 symmetrizedSecondOrderBatches = 0;
    u32 invalidFirstOrderBatches[4]   = {0, 0, 0, 0};
    u32 continuousLabelLists = 0, discontinuousLabelLists = 0;

    u32 currentExit        = 0;
    u32 multiExits         = 0;
    u32 nonContinuousExits = 0;
    u32 singleExits        = 0;

    quickLabelBatches.push_back(currentExit);  // First state is invalid
    quickLabelBatches.push_back(currentExit);  // Second state starts at zero
    singleLabels.push_back(0);                 /// TODO: Use bitmask instead of label-batches

    // Build the second-order structure for speedup
    for (u32 a = 1; a < network.structure.stateCount(); ++a) {
        HMMState const& state = network.structure.state(a);

        int  firstSecondOrderSuccessor       = -1;
        int  endSecondOrderSuccessor         = -1;
        bool secondOrderSuccessorsContinuous = true;
        bool labelsContinuous                = true;
        bool hadLabels                       = false;
        u32  singleLabel                     = Core::Type<u32>::max;

        {
            std::pair<int, int> directSuccessors = network.structure.batchSuccessorsSimple<false>(state.successors);
            if (directSuccessors.first == -1) {
                ++invalidFirstOrderBatches[-directSuccessors.second];
            }
            else
                ++validFirstOrderBatches;
        }
        for (HMMStateNetwork::SuccessorIterator successorIt = network.structure.successors(state); successorIt; ++successorIt)
            if (successorIt.isLabel()) {
                if (!hadLabels) {
                    hadLabels   = true;
                    singleLabel = successorIt.label();
                }
                else {
                    singleLabel = Core::Type<u32>::max;
                }
                if (currentExit == successorIt.label()) {
                    ++currentExit;
                }
                else {
                    currentExit      = successorIt.label() + 1;
                    labelsContinuous = false;
                }
            }

        for (HMMStateNetwork::SuccessorIterator successorIt = network.structure.successors(state); successorIt; ++successorIt) {
            if (successorIt.isLabel()) {
                if (successorIt.isLastBatch())
                    ++continuousLabelLists;
                else
                    ++discontinuousLabelLists;
                continue;
            }

            for (HMMStateNetwork::SuccessorIterator successorIt2 = network.structure.successors(*successorIt); successorIt2; ++successorIt2) {
                if (successorIt2.isLabel())
                    continue;

                int t = *successorIt2;
                if (firstSecondOrderSuccessor == -1) {
                    firstSecondOrderSuccessor = t;
                    endSecondOrderSuccessor   = t + 1;
                }
                else {
                    if (endSecondOrderSuccessor == t) {
                        ++endSecondOrderSuccessor;
                    }
                    else if (firstSecondOrderSuccessor == t + 1) {
                        --firstSecondOrderSuccessor;
                    }
                    else {
                        secondOrderSuccessorsContinuous = false;
                    }
                }
            }
        }

        if (symmetrize && (stateDepths[a] == stateDepths[network.rootState] || stateDepths[a] == stateDepths[network.rootState] + hmmLength)) {
            symmetrizedSecondOrderBatches += 1;
            secondOrderEdgeSuccessorBatches.push_back(ForbidSecondOrderExpansion);
            secondOrderEdgeSuccessorBatches.push_back(ForbidSecondOrderExpansion);
        }
        else if (secondOrderSuccessorsContinuous) {
            secondOrderEdgeSuccessorBatches.push_back(firstSecondOrderSuccessor);
            secondOrderEdgeSuccessorBatches.push_back(endSecondOrderSuccessor);
            ++validSecondOrderBatches;
        }
        else {
            secondOrderEdgeSuccessorBatches.push_back(0);
            secondOrderEdgeSuccessorBatches.push_back(0);
            ++invalidSecondOrderBatches;
        }

        if (hadLabels) {
            if (singleLabel != Core::Type<u32>::max) {
                ++singleExits;
            }
            else {
                ++multiExits;
                if (!labelsContinuous)
                    ++nonContinuousExits;
            }
        }

        if (!hadLabels)
            singleLabels.push_back(-1);
        else if (singleLabel != Core::Type<u32>::max)
            singleLabels.push_back(singleLabel);
        else if (labelsContinuous)
            singleLabels.push_back(-2);
        else {
            singleLabels.push_back(-(3 + slowLabelBatches.size()));
            for (HMMStateNetwork::SuccessorIterator successorIt = network.structure.successors(state); successorIt; ++successorIt)
                if (successorIt.isLabel())
                    slowLabelBatches.push_back(successorIt.label());
            slowLabelBatches.push_back(-1);
        }

        quickLabelBatches.push_back(currentExit);
    }

    log() << "valid first-order batches: " << validFirstOrderBatches
          << " invalid first-order batches (reason 1): " << invalidFirstOrderBatches[1]
          << " invalid first-order batches (reason 2): " << invalidFirstOrderBatches[2]
          << " invalid first-order batches (reason 3): " << invalidFirstOrderBatches[3];
    log() << "valid second-order batches: " << validSecondOrderBatches << " invalid second-order batches: " << invalidSecondOrderBatches;
    log() << "continuous label lists: " << continuousLabelLists << " discontinuous label lists: " << discontinuousLabelLists;
    log() << "continuous label lists: " << continuousLabelLists << " discontinuous label lists: " << discontinuousLabelLists;
    log() << "single-label lists: " << singleExits << " multi-label lists: " << multiExits;
    log() << "irregular exit-list items: " << slowLabelBatches.size();
    if (symmetrizedSecondOrderBatches)
        log() << "symmetrized states (skips forbidden): " << symmetrizedSecondOrderBatches;

    if (!paramDumpDotGraph(config).empty())
        network.dumpDotGraph(paramDumpDotGraph(config), stateDepths);

    // Print some useful statistics about pushed and unpushed labels
    verify(!network.unpushedCoarticulatedRootStates.empty());

    u32 unpushedLabels = 0;
    u32 pushedLabels   = 0;
    for (StateId state = 1; state < network.structure.stateCount(); ++state) {
        for (HMMStateNetwork::SuccessorIterator successor = network.structure.successors(state); successor; ++successor) {
            if (successor.isLabel()) {
                StateId transit    = network.exits[successor.label()].transitState;
                bool    isUnpushed = network.unpushedCoarticulatedRootStates.count(transit) || transit == network.ciRootState || transit == network.rootState;
                if (isUnpushed) {
                    ++unpushedLabels;
                    std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::iterator it = network.rootTransitDescriptions.find(transit);
                    verify(it != network.rootTransitDescriptions.end());
                }
                else {
                    ++pushedLabels;
                }
            }
        }
    }

    log() << "number of pushed labels: " << pushedLabels << " unpushed: " << unpushedLabels;

    network.removeOutputs();
}

// ------------------------------- Search Space --------------------------------

SearchSpace::SearchSpace(const Core::Configuration&               config,
                         Core::Ref<const Am::AcousticModel>       acousticModel,
                         Bliss::LexiconRef                        lexicon,
                         Core::Ref<const Lm::ScaledLanguageModel> lm,
                         Score                                    wpScale)
        : Core::Component(config),
          statistics(new SearchSpaceStatistics),
          globalScoreOffset_(0.0),
          timeFrame_(0),
          lexicon_(lexicon),
          acousticModel_(acousticModel),
          lm_(lm),
          lookaheadLm_(),
          recombinationLm_(),
          ssaLm_(dynamic_cast<Lm::SearchSpaceAwareLanguageModel const*>(lm_->unscaled().get())),
          lmLookahead_(0),
          automaton_(new StaticSearchAutomaton(config, acousticModel, lexicon)),
          acousticLookAhead_(0),
          conditionPredecessorWord_(paramConditionPredecessorWord(config)),
          decodeMesh_(paramDecodeMesh(config)),
          correctPushedBoundaryTimes_(paramCorrectPushedWordBoundaryTimes(config)),
          correctPushedAcousticScores_(paramCorrectPushedAcousticScores(config)),
          earlyBeamPruning_(paramEarlyBeamPruning(config)),
          earlyWordEndPruning_(paramEarlyWordEndPruning(config)),
          histogramPruningIsMasterPruning_(false),
          reducedContextWordRecombination_(paramReducedContextWordRecombination(config)),
          reducedContextWordRecombinationLimit_(paramReducedContextWordRecombinationLimit(config)),
          onTheFlyRescoring_(paramOnTheFlyRescoring(config)),
          onTheFlyRescoringMaxHistories_(paramOnTheFlyRescoringMaxHistories(config)),
          maximumMutableSuffixLength_(paramMaximumMutableSuffixLength(config)),
          maximumMutableSuffixPruningInterval_(paramMaximumMutableSuffixPruningInterval(config)),
          acousticPruning_(0),
          acousticPruningLimit_(0),
          wordEndPruning_(0),
          lmStatePruning_(lmStatePruning(config)),
          acousticProspectFactor_(1.0 + paramAcousticLookaheadTemporalApproximationScale(config)),
          perInstanceAcousticPruningScale_(paramPerInstanceAcousticPruningScale(config)),
          minimumBeamPruning_(paramMinimumBeamPruning(config)),
          maximumBeamPruning_(paramMaximumBeamPruning(config)),
          minimumAcousticPruningLimit_(paramMinimumAcousticPruningLimit(config)),
          maximumAcousticPruningLimit_(paramMaximumAcousticPruningLimit(config)),
          minimumStatesAfterPruning_(paramMinimumStatesAfterPruning(config)),
          minimumWordEndsAfterPruning_(paramMinimumWordEndsAfterPruning(config)),
          minimumWordLemmasAfterRecombination_(paramMinimumWordLemmasAfterRecombination(config)),
          maximumStatesAfterPruning_(paramMaximumStatesAfterPruning(config)),
          maximumWordEndsAfterPruning_(paramMaximumWordEndsAfterPruning(config)),
          maximumAcousticPruningSaturation_(paramMaximumAcousticPruningSaturation(config)),
          earlyWordEndPruningAnticipatedLmScore_(paramEarlyWordEndPruningMinimumLmScore(config)),
          wordEndPruningFadeInInterval_(paramWordEndPruningFadeInInterval(config)),
          instanceDeletionLatency_(paramInstanceDeletionLatency(config)),
          fullLookAheadStateMinimum_(paramReduceLookAheadStateMinimum(config)),
          fullLookAheadDominanceMinimum_(paramReduceLookAheadDominanceMinimum(config)),
          currentLookaheadInstanceStateThreshold_(fullLookAheadStateMinimum_),
          fullLookaheadAfterId_(Core::Type<AdvancedTreeSearch::LanguageModelLookahead::LookaheadId>::max),
          sparseLookahead_(paramSparseLmLookAhead(config)),
          overflowLmScoreToAm_(paramOverflowLmScoreToAm(config)),
          sparseLookaheadSlowPropagation_(paramSparseLmLookaheadSlowPropagation(config)),
          unigramLookaheadBackoffFactor_(paramUnigramLookaheadBackOffFactor(config)),
          earlyBackoff_(paramEarlyBackOff(config)),
          allowSkips_(true),
          wpScale_(wpScale),
          extendStatistics_(paramExtendedStatistics(config)),
          encodeStateInTrace_(paramEncodeStateInTrace(config)),
          encodeStateInTraceAlways_(paramEncodeStateInTraceAlways(config)),
          bestScore_(Core::Type<Score>::max),
          bestProspect_(Core::Type<Score>::max),
          minWordEndScore_(Core::Type<Score>::max),
          stateHistogram_(paramAcousticPruningBins(config)),
          wordEndHistogram_(paramWordEndPruningBins(config)),
          hadWordEnd_(true),
          currentStatesAfterPruning("current states after pruning"),
          currentWordEndsAfterPruning("current word ends after pruning"),
          currentWordLemmasAfterRecombination("current word lemmas after recombination"),
          currentAcousticPruningSaturation("current acoustic-pruning saturation"),
          applyLookaheadPerf_(new PerformanceCounter(*statistics, "apply lookahead", false)),
          applyLookaheadSparsePerf_(new PerformanceCounter(*statistics, "apply sparse lookahead", false)),
          applyLookaheadSparsePrePerf_(new PerformanceCounter(*statistics, "pre-apply unigram lookahead", false)),
          applyLookaheadStandardPerf_(new PerformanceCounter(*statistics, "apply standard lookahead", false)),
          computeLookaheadPerf_(new PerformanceCounter(*statistics, "compute LM lookahead", false)),
          extendedPerf_(new PerformanceCounter(*statistics, "test", false)) {
    if (decodeMesh_) {
        WordEndHypothesis::meshHistoryPhones = paramDecodeMeshPhones(config);
        log() << "generating mesh-lattice with " << WordEndHypothesis::meshHistoryPhones << " history-phones";
    }

    if (fullLookAheadDominanceMinimum_)
        log() << "activating context-dependent LM look-ahead only for instances with dominance above " << fullLookAheadDominanceMinimum_;

    log() << "HMM length of a phoneme: " << automaton_->hmmLength;

    if (paramSeparateLookaheadLm(config)) {
        log() << "using new lookahead lm";
        lookaheadLm_ = Lm::Module::instance().createScaledLanguageModel(select("lookahead-lm"), lexicon_);
    }
    else if (lm_->lookaheadLanguageModel().get() != nullptr) {
        lookaheadLm_ = Core::Ref<const Lm::ScaledLanguageModel>(new Lm::LanguageModelScaling(select("lookahead-lm"),
                                                                                             Core::Ref<Lm::LanguageModel>(const_cast<Lm::LanguageModel*>(lm_->lookaheadLanguageModel().get()))));
    }
    else {
        lookaheadLm_ = lm_;
    }

    if (paramSeparateRecombinationLm(config)) {
        log() << "using new recombination lm";
        recombinationLm_ = Lm::Module::instance().createLanguageModel(select("recombination-lm"), lexicon_);
    }
    else if (lm_->recombinationLanguageModel().get() != nullptr) {
        log() << "using the recombination lm from the score lm";
        recombinationLm_ = lm_->recombinationLanguageModel();
    }
    else {
        log() << "using the scoring lm for recombination";
        recombinationLm_ = lm_;
    }

    if (sparseLookahead_ && !dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get())) {
        warning() << "Not using sparse LM lookahead, because the LM is not a backing-off LM! Memory- and runtime efficiency will be worse.";
        sparseLookahead_ = false;
    }

    statesOnDepth_.initialize(100, 100);
    statesOnInvertedDepth_.initialize(100, 100);
}

void SearchSpace::setAllowHmmSkips(bool allow) {
    allowSkips_ = allow;
}

SearchSpace::~SearchSpace() {
    clear();
    for (u32 t = 0; t < activeInstances.size(); ++t)
        delete activeInstances[t];

    delete acousticLookAhead_;
    delete computeLookaheadPerf_;
    delete applyLookaheadPerf_;
    delete applyLookaheadSparsePerf_;
    delete applyLookaheadSparsePrePerf_;
    delete applyLookaheadStandardPerf_;
    delete extendedPerf_;
    delete statistics;
    if (lmLookahead_) {
        unigramLookAhead_.reset();
        delete lmLookahead_;
    }
    delete automaton_;
}

void SearchSpace::initializePruning() {
    acousticPruning_ = paramAcousticPruning(config);

    Score beamPruning = paramBeamPruning(config);

    histogramPruningIsMasterPruning_ = paramHistogramIsMasterPruning(config);

    if (acousticPruning_ == Core::Type<f32>::max || beamPruning != Core::Type<f32>::max) {
        if (beamPruning == Core::Type<f32>::max) {
            beamPruning = defaultBeamPruning;
            log() << "using default beam-pruning of " << beamPruning;
        }
        else if (acousticPruning_ != Core::Type<f32>::max) {
            log() << "ignoring configured acoustic-pruning because beam-pruning was set too. the configured acoustic-pruning value WOULD correspond to beam-pruning=" << (acousticPruning_ / lm_->scale());
        }

        acousticPruning_ = beamPruning * lm_->scale();
        log() << "set acoustic-pruning to " << acousticPruning_ << " from beam-pruning " << beamPruning << " with lm-scale " << lm_->scale();
    }

    acousticPruningLimit_ = std::min(paramBeamPruningLimit(config), paramAcousticPruningLimit(config));

    log() << "using acoustic pruning limit " << acousticPruningLimit_;

    wordEndPruning_ = paramWordEndPruning(config);
    if (wordEndPruning_ != Core::Type<f32>::max) {
        if (wordEndPruning_ > 1.0)
            wordEndPruning_ *= lm_->scale();
        if (paramLmPruning(config) != Core::Type<f32>::max)
            warning() << "lm-pruning and word-end-pruning were set at the same time. using word-end-pruning, because lm-pruning is DEPRECATED";
    }
    else {
        wordEndPruning_ = paramLmPruning(config);
    }

    if (wordEndPruning_ <= 1.0)
        wordEndPruning_ *= acousticPruning_;

    wordEndPruningLimit_ = std::min(paramWordEndPruningLimit(config), paramLmPruningLimit(config));

    log() << "using word end pruning " << wordEndPruning_ << " limit " << wordEndPruningLimit_;

    lmStatePruning_ = lmStatePruning(config);
    if (lmStatePruning_ != Core::Type<f32>::max) {
        if (lmStatePruning_ > 1.0)
            lmStatePruning_ *= lm_->scale();
        else
            lmStatePruning_ *= wordEndPruning_;
        log() << "using lm state pruning " << lmStatePruning_;
    }

    wordEndPhonemePruningThreshold_ = paramWordEndPhonemePruningThreshold(config);
    if (wordEndPhonemePruningThreshold_ != Core::Type<Score>::max) {
        if (wordEndPhonemePruningThreshold_ > 1.0)
            wordEndPhonemePruningThreshold_ *= lm_->scale();
        else
            wordEndPhonemePruningThreshold_ *= wordEndPruning_;
        log() << "using word end phoneme pruning " << wordEndPhonemePruningThreshold_;
    }
}

void SearchSpace::initialize() {
    PerformanceCounter perf(*statistics, "initialize");

    getTransitionModels();
    initializePruning();

    StaticSearchAutomaton* automaton = const_cast<StaticSearchAutomaton*>(automaton_);

    PersistentStateTree& net = automaton->network;
    automaton->buildNetwork();

    automaton->buildDepths();
    log() << "depth of root-state: " << automaton->stateDepths[net.rootState] << " hmm-length " << automaton->hmmLength;
    if (automaton->stateDepths[net.rootState] == 0 && automaton->minimized) {
        log() << "tail minimization was not used, root-state has depth 0";
        automaton->minimized = false;
    }

    if (!(automaton->stateDepths[net.rootState] == (automaton->minimized ? automaton->hmmLength : 0)) &&
        !(automaton->stateDepths[net.rootState] == (automaton->minimized ? automaton->hmmLength + 1 : 1))) {
        error() << "bad state depths! root-state has depth " << automaton->stateDepths[net.rootState] << ", should be " << (automaton->minimized ? automaton->hmmLength : 0);
    }

    automaton->buildLabelDistances();

    // The filter must be created _before_ the outputs are cut off the search network
    automaton->prefixFilter = new PrefixFilter(net, lexicon_, config);
    if (!automaton->prefixFilter->haveFilter()) {
        delete automaton->prefixFilter;
        automaton->prefixFilter = nullptr;
    }

    acousticLookAhead_ = new AdvancedTreeSearch::AcousticLookAhead(config, net.getChecksum());
    if (acousticLookAhead_->isEnabled() && !acousticLookAhead_->loaded())
        acousticLookAhead_->initializeModelsFromNetwork(net);

    initializeLanguageModel();

    // Initialization of the search network cuts away the outputs from the network
    // and puts them into the outputBatches_ data structures instead.
    automaton->buildBatches();

    stateHypothesisRecombinationArray.resize(net.structure.stateCount());
}

void SearchSpace::initializeLanguageModel() {
    PersistentStateTree const& net = network();

    unigramHistory_ = lookaheadLm_->reducedHistory(lookaheadLm_->startHistory(), 0);

    StaticSearchAutomaton* automaton = const_cast<StaticSearchAutomaton*>(automaton_);

    if (paramEnableLmLookahead(config)) {
        lmLookahead_ = new AdvancedTreeSearch::LanguageModelLookahead(Core::Configuration(config, "lm-lookahead"),
                                                                      wpScale_,
                                                                      lookaheadLm_,
                                                                      net.structure,
                                                                      net.rootState,
                                                                      net.exits,
                                                                      acousticModel_);

        std::set<AdvancedTreeSearch::LanguageModelLookahead::LookaheadId> rootStates;

        rootStates.insert(lmLookahead_->lookaheadId(net.rootState));

        for (std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::const_iterator it = net.rootTransitDescriptions.begin(); it != net.rootTransitDescriptions.end(); ++it)
            rootStates.insert(lmLookahead_->lookaheadId((*it).first));

        int reduceBeforeDepth = paramReduceLookAheadBeforeDepth(config);
        if (reduceBeforeDepth > -1000 && reduceBeforeDepth != Core::Type<int>::max) {
            int rootDepth         = lmLookahead_->nodeDepth(lmLookahead_->lookaheadId(net.rootState));
            s32 minDepth          = reduceBeforeDepth + rootDepth;
            fullLookaheadAfterId_ = lmLookahead_->lastNodeOnDepth(minDepth);
            log() << "depth of root lookahead state " << rootDepth
                  << " using full-lookahead behind state " << fullLookaheadAfterId_
                  << " out of " << lmLookahead_->numNodes()
                  << "  (deduced from relative depth " << reduceBeforeDepth << ")";
        }

        unigramLookAhead_ = lmLookahead_->getLookahead(unigramHistory_);

        if (paramDisableUnigramLookahead(config))
            lmLookahead_->fillZero(unigramLookAhead_);
        else
            lmLookahead_->fill(unigramLookAhead_);

        automaton->lookAheadIds.resize(net.structure.stateCount(), std::make_pair(0u, 0u));
        automaton->lookAheadIdAndHash.resize(net.structure.stateCount(), std::make_pair(0u, 0u));
        for (StateId state = 1; state < net.structure.stateCount(); ++state) {
            if (acousticLookAhead_->isEnabled()) {
                automaton->lookAheadIds[state]       = std::make_pair<uint, uint>(lmLookahead_->lookaheadId(state), acousticLookAhead_->getLookaheadId(state));
                automaton->lookAheadIdAndHash[state] = std::make_pair<uint, uint>(lmLookahead_->lookaheadHash(state), acousticLookAhead_->getLookaheadId(state));
            }
            else {
                automaton->lookAheadIds[state]       = std::make_pair<uint, uint>(lmLookahead_->lookaheadId(state), 0);
                automaton->lookAheadIdAndHash[state] = std::make_pair<uint, uint>(lmLookahead_->lookaheadHash(state), 0);
            }
        }
    }
}

/// Search Management -----------------------------------------------------------------------------

void SearchSpace::clear() {
    for (auto& t : altHistTraces_) {
        if (t) {
            t->alternativeHistories.container().clear();
        }
    }
    altHistTraces_.clear();

    currentStatesAfterPruning.clear();
    currentAcousticPruningSaturation.clear();
    currentWordEndsAfterPruning.clear();
    currentWordLemmasAfterRecombination.clear();
    hadWordEnd_                             = false;
    currentLookaheadInstanceStateThreshold_ = fullLookAheadStateMinimum_;
    scorer_.reset();
    acousticLookAhead_->clear();
    globalScoreOffset_ = 0.0;
    stateHypotheses.clear();
    newStateHypotheses.clear();
    for (u32 t = 0; t < activeInstances.size(); ++t) {
        activeInstances[t]->backOffInstance = 0;  // Disable cross-instance dependency
        activeInstances[t]->backOffParent   = 0;
        delete activeInstances[t];
    }
    activeInstanceMap.clear();
    activeInstances.clear();
    wordEndHypotheses.clear();
    earlyWordEndHypotheses.clear();
    wordEndHypothesisMap.clear();
    stateHistogram_.clear();
    wordEndHistogram_.clear();
    bestProspect_    = Core::Type<Score>::max;
    bestScore_       = Core::Type<Score>::max;
    minWordEndScore_ = Core::Type<Score>::max;
    cleanup();
    trace_manager_.clear();
}

inline bool SearchSpace::eventuallyDeactivateTree(Instance* at, bool increaseInactiveCounter) {
    if (not at->mayDeactivate())
        return false;

    if (!at->states.empty()) {
        at->inactive = 0;
        return false;
    }
    else {
        if (at->inactive < instanceDeletionLatency_) {
            if (increaseInactiveCounter)
                ++(at->inactive);
            return false;
        }
        else {
            std::unordered_map<InstanceKey, Instance*, InstanceKey::Hash>::iterator it = activeInstanceMap.find(at->key);
            if (it != activeInstanceMap.end()) {
                if ((*it).second == at) {
                    activeInstanceMap.erase(it);
                }
            }
            delete at;
            return true;
        }
    }
}

void SearchSpace::activateOrUpdateStateHypothesisLoop(const Search::StateHypothesis& hyp, Score score) {
    StateHypothesisIndex& recombination = stateHypothesisRecombinationArray[hyp.state];  // Look-up at node index, contains positions in newStateHypotheses.
    StateHypothesis&      sh(newStateHypotheses.data()[recombination]);                  //We may be referencing a not allocated position, so we use data()
    // Check if present in current tree (starting at currentTreeFirstNewStateHypothesis).
    if (recombination < currentTreeFirstNewStateHypothesis || recombination >= newStateHypotheses.size() || sh.state != hyp.state) {
        recombination = newStateHypotheses.size();
        addNewStateHypothesis(hyp);
        newStateHypotheses.back().score = score;
    }
    else {
        //Update existing hypothesis
        if (sh.score >= score) {
            sh.score = score;
            sh.trace = hyp.trace;
        }
    }
}

void SearchSpace::activateOrUpdateStateHypothesisTransition(const Search::StateHypothesis& hyp, Score score, StateId successorState) {
    StateHypothesisIndex& recombination = stateHypothesisRecombinationArray[successorState];
    StateHypothesis&      sh(newStateHypotheses.data()[recombination]);  //We may be referencing a not allocated position, so we use data()
    // Check if present in current tree (starting at currentTreeFirstNewStateHypothesis).
    if (recombination < currentTreeFirstNewStateHypothesis || recombination >= newStateHypotheses.size() || sh.state != successorState) {
        recombination = newStateHypotheses.size();
        addNewStateHypothesis(hyp);
        newStateHypotheses.back().score = score;
        newStateHypotheses.back().state = successorState;
    }
    else {
        //Update existing hypothesis
        if (sh.score >= score) {
            sh.score = score;
            sh.trace = hyp.trace;
        }
    }
}

void SearchSpace::activateOrUpdateStateHypothesisDirectly(const Search::StateHypothesis& hyp) {
    StateHypothesisIndex& recombination = stateHypothesisRecombinationArray[hyp.state];
    StateHypothesis&      sh(newStateHypotheses.data()[recombination]);  //We may be referencing a not allocated position, so we use data()

    if (recombination < currentTreeFirstNewStateHypothesis || recombination >= newStateHypotheses.size() || sh.state != hyp.state) {
        recombination = newStateHypotheses.size();
        addNewStateHypothesis(hyp);
    }
    else {
        //Update existing hypothesis
        if (sh.score >= hyp.score) {
            sh.score = hyp.score;
            sh.trace = hyp.trace;
        }
    }
}

template<bool expandForward, bool expandSkip>
void SearchSpace::expandStateSlow(const Search::StateHypothesis& hyp) {
    PersistentStateTree const& net                             = network();
    std::vector<int> const&    secondOrderEdgeSuccessorBatches = automaton_->secondOrderEdgeSuccessorBatches;
    HMMState const&            state                           = net.structure.state(hyp.state);

    const Am::StateTransitionModel& tdp(*transitionModel(state.stateDesc));

    Score skipScore = hyp.score + tdp[Am::StateTransitionModel::skip];

    bool doSkip = expandSkip && skipScore < Core::Type<Score>::max;

    u32 secondStart = secondOrderEdgeSuccessorBatches[hyp.state * 2];
    u32 secondEnd   = secondOrderEdgeSuccessorBatches[hyp.state * 2 + 1];

    if (doSkip && secondStart != 0) {
        doSkip = false;  // Omit the second order expansion later.

        // Use the second-order structure to do the skips directly
        for (int a = secondStart; a < secondEnd; ++a)
            activateOrUpdateStateHypothesisTransition(hyp, skipScore, a);
    }

    Score forwardScore = hyp.score + tdp[Am::StateTransitionModel::forward];

    if (forwardScore < Core::Type<Score>::max) {
        std::pair<int, int> successors = net.structure.batchSuccessorsSimple<true>(state.successors);
        if (successors.first != -1) {
            //Fast iteration
            for (StateId successor = successors.first; successor != successors.second; ++successor) {
                if (expandForward)
                    activateOrUpdateStateHypothesisTransition(hyp, forwardScore, successor);  // Already covered by expandState?

                if (expandSkip && doSkip)  // TODO: only doSkip is sufficient
                {                          // Second order expansion (successors of successor).
                    std::pair<int, int> skipSuccessors = net.structure.batchSuccessorsSimple<true>(net.structure.state(successor).successors);
                    if (skipSuccessors.first != -1) {
                        //Fast iteration
                        for (StateId skipSuccessor = skipSuccessors.first; skipSuccessor != skipSuccessors.second; ++skipSuccessor)
                            activateOrUpdateStateHypothesisTransition(hyp, skipScore, skipSuccessor);
                    }
                    else {
                        for (HMMStateNetwork::SuccessorIterator skipSuccessorIt = net.structure.successors(successor); skipSuccessorIt; ++skipSuccessorIt)
                            activateOrUpdateStateHypothesisTransition(hyp, skipScore, *skipSuccessorIt);
                    }
                }
            }
        }
        else {
            for (HMMStateNetwork::SuccessorIterator successorIt = net.structure.batchSuccessors(state.successors); successorIt; ++successorIt) {
                StateId successor = *successorIt;

                if (expandForward)
                    activateOrUpdateStateHypothesisTransition(hyp, forwardScore, successor);

                if (expandSkip && doSkip)  // TODO: only doSkip is sufficient
                {
                    for (HMMStateNetwork::SuccessorIterator skipSuccessorIt = net.structure.successors(successor); skipSuccessorIt; ++skipSuccessorIt)
                        activateOrUpdateStateHypothesisTransition(hyp, skipScore, *skipSuccessorIt);
                }
            }
        }
    }
}

// Origin of the state hypotheses:
// a) jump to current time frame within the same instance: in instance.begin..end
// b) jump to current time frame from another instance: in rootStateHypotheses
// For inverted alignment, they continue with loop/forward/skip, recombination with array.
// The recombination array is for the next frame.
// rootStateHypotheses:
// The hypotheses have already emitted before being moved to wordEndHypotheses.
// The hypotheses are in the transitstate, which is part of WordEndHypothesis, which was created in pruneEarlyWordEnds.
// Due to recombineWordEnds(), there is only one hypothesis in each transit state per history (instance).
template<bool allowSkip>
inline void SearchSpace::expandState(const Search::StateHypothesis& hyp) {
    // This is the 'fast' state-expansion step, that should work in 99.9% of the expansions
    // Labels were already removed from the network before starting, so they can be ignored
    PersistentStateTree const&      net                             = network();
    std::vector<int> const&         secondOrderEdgeSuccessorBatches = automaton_->secondOrderEdgeSuccessorBatches;
    HMMState const&                 state                           = net.structure.state(hyp.state);
    Am::StateTransitionModel const& tdp                             = *transitionModel(state.stateDesc);

    // loops
    Score loopScore = hyp.score + tdp[Am::StateTransitionModel::loop];

    if (loopScore < Core::Type<Score>::max)
        activateOrUpdateStateHypothesisLoop(hyp, loopScore);

    // forward transition
    if ((state.successors & SingleSuccessorBatchMask) == SingleSuccessorBatchMask) {
        // The common case: Usually one hyp is connected to exactly one follower hyp

        StateId forwardSuccessor = state.successors & (~SingleSuccessorBatchMask);

        Score forwardScore = hyp.score + tdp[Am::StateTransitionModel::forward];

        if (forwardScore < Core::Type<Score>::max)
            activateOrUpdateStateHypothesisTransition(hyp, forwardScore, forwardSuccessor);
    }
    else {
        // There are multiple successors
        std::pair<int, int> successors = net.structure.batchSuccessorsSimpleIgnoreLabels(state.successors);

        if (successors.first == -1) {
            // The successor structure has irregular linked-list form, use the slow non-optimized expansion
            expandStateSlow<true, allowSkip>(hyp);
            return;
        }

        Score forwardScore = hyp.score + tdp[Am::StateTransitionModel::forward];

        if (forwardScore < Core::Type<Score>::max)
            for (int successor = successors.first; successor < successors.second; ++successor)
                activateOrUpdateStateHypothesisTransition(hyp, forwardScore, successor);
    }

    if (allowSkip) {
        // skip transition
        u32 secondStart = secondOrderEdgeSuccessorBatches[hyp.state * 2], secondEnd = secondOrderEdgeSuccessorBatches[hyp.state * 2 + 1];

        if (secondStart != secondEnd) {
            Score skipScore = hyp.score + tdp[Am::StateTransitionModel::skip];

            if (skipScore < Core::Type<Score>::max)
                for (StateId successor2 = secondStart; successor2 < secondEnd; ++successor2)
                    activateOrUpdateStateHypothesisTransition(hyp, skipScore, successor2);
        }
        else if (secondStart == 0) {
            //The secondOrderEdgeSuccessorBatches_ structure can not hold the successors, so use slow expansion to expand the second-order followers
            expandStateSlow<false, true>(hyp);
        }
    }
}

void SearchSpace::expandHmm() {
    PerformanceCounter expandPerf(*statistics, "expand HMM");

    bestProspect_ = Core::Type<Score>::max;
    bestScore_    = Core::Type<Score>::max;

    for (u32 treeIdx = 0; treeIdx < activeInstances.size(); ++treeIdx) {
        Instance& instance(*activeInstances[treeIdx]);  // All hypotheses in one context/tree.

        statistics->rootStateHypothesesPerTree += instance.rootStateHypotheses.size();

        const u32 oldStart = instance.states.begin, oldEnd = instance.states.end;

        instance.states.begin              = newStateHypotheses.size();
        currentTreeFirstNewStateHypothesis = instance.states.begin;

        // Expand entry state hypotheses

        if (allowSkips_) {
            for (std::vector<StateHypothesis>::const_iterator sh = instance.rootStateHypotheses.begin();
                 sh != instance.rootStateHypotheses.end(); ++sh)
                expandState<true>(*sh);
        }
        else {
            for (std::vector<StateHypothesis>::const_iterator sh = instance.rootStateHypotheses.begin();
                 sh != instance.rootStateHypotheses.end(); ++sh)
                expandState<false>(*sh);
        }

        if (earlyBackoff_ && instance.rootStateHypotheses.size()) {
            if (!instance.backOffInstance)
                getBackOffInstance(&instance);

            if (instance.backOffInstance) {
                for (u32 i = 0; i < instance.rootStateHypotheses.size(); ++i)
                    instance.rootStateHypotheses[i].score += instance.backOffScore;
                instance.backOffInstance->rootStateHypotheses.swap(instance.rootStateHypotheses);
            }
        }

        instance.rootStateHypotheses.clear();

        // Expand old state hypotheses
        if (allowSkips_) {
            for (StateHypothesesList::const_iterator sh = stateHypotheses.begin() + oldStart;
                 sh != stateHypotheses.begin() + oldEnd; ++sh)
                expandState<true>(*sh);
        }
        else {
            for (StateHypothesesList::const_iterator sh = stateHypotheses.begin() + oldStart;
                 sh != stateHypotheses.begin() + oldEnd; ++sh)
                expandState<false>(*sh);
        }

        // List of state hypotheses that should be transferred into this tree.
        // Filled by applyLookaheadInTree(Internal) (see below in same method) if the sparseLookAhead fails.
        // The hypotheses are pushed to the backOffInstance.
        if (!instance.transfer.empty()) {
            // Transfer state hypotheses from other instances
            for (std::vector<StateHypothesisIndex>::const_iterator transferIt = instance.transfer.begin();
                 transferIt != instance.transfer.end();
                 ++transferIt)
                activateOrUpdateStateHypothesisDirectly(newStateHypotheses[*transferIt]);

            // Make sure we don't need to re-allocate at later timeframes
            instance.transfer.reserve(instance.transfer.capacity());
            instance.transfer.clear();
        }

        instance.states.end = newStateHypotheses.size();

        expandPerf.stop();

        // Calculates (sparse) look-up tables if necessary.
        // For every hypothesis: sh->prospect = sh->score + lmscore/lookahead + acousticlookahead + backoffscore?
        // If sparseLookAhead is used, we transfer the newStateHypotheses into the back-off tree by adding the backoff score.
        // The assumption is that all backoff trees are processed later in some topological order?
        applyLookaheadInInstance(&instance);

        expandPerf.start();
    }

    stateHypotheses.swap(newStateHypotheses);
    newStateHypotheses.clear();

    // By computing this here, we always use the state-threshold regarding the previous timeframe. This shouldn't matter though.
    currentLookaheadInstanceStateThreshold_ = std::max(fullLookAheadStateMinimum_, (u32)(fullLookAheadDominanceMinimum_ * stateHypotheses.size()));

    applyLookaheadPerf_->stopAndYield();
    applyLookaheadSparsePerf_->stopAndYield();
    computeLookaheadPerf_->stopAndYield();
    applyLookaheadSparsePrePerf_->stopAndYield();
    applyLookaheadStandardPerf_->stopAndYield();
    extendedPerf_->stopAndYield();
}

template<bool sparseLookAhead, bool useBackOffOffset, class AcousticLookAhead, class Pruning>
void SearchSpace::applyLookaheadInInstanceInternal(Instance* _instance, AcousticLookAhead& acousticLookAhead, Pruning& pruning) {
    Instance& instance(*_instance);

    pruning.startInstance(instance.key);

    verify(instance.states.empty() || instance.states.end <= newStateHypotheses.size());

    if (instance.states.empty())
        return;

    StateHypothesesList::iterator sh_begin = newStateHypotheses.begin() + instance.states.begin;
    StateHypothesesList::iterator sh_end   = newStateHypotheses.begin() + instance.states.end;
    StateHypothesesList::iterator sh       = sh_begin;

    if (!lmLookahead_) {
        if (acousticLookAhead_->isEnabled()) {
            for (; sh != sh_end; ++sh) {
                sh->prospect = sh->score + acousticLookAhead(acousticLookAhead_->getLookaheadId(sh->state), sh->state);
                pruning.prepare(*sh);
            }
        }
        else {
            for (; sh != sh_end; ++sh) {
                pruning.prepare(*sh);
            }
        }
        return;
    }

    // Check if we can activate the LM lookahead for free
    activateLmLookahead(instance, false);

    applyLookaheadPerf_->start();

    f32 backOffOffset = 0.0f;

    if (!instance.lookahead.get()) {
        if (useBackOffOffset)
            backOffOffset = static_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get())->getAccumulatedBackOffScore(instance.lookaheadHistory, 1) * unigramLookaheadBackoffFactor_ * lookaheadLm_->scale();

        bool shouldIncreaseLookAheadOrder = false;
        {
            u32 combinedTreeStateCount = instance.states.size();
            if (sparseLookAhead)
                combinedTreeStateCount = instance.backOffChainStates();

            shouldIncreaseLookAheadOrder = combinedTreeStateCount >= currentLookaheadInstanceStateThreshold_;
        }

        if (shouldIncreaseLookAheadOrder) {
            // The state-count based conditions to increase the lookahead order are satisfied
            if (fullLookaheadAfterId_ != Core::Type<AdvancedTreeSearch::LanguageModelLookahead::LookaheadId>::max) {
                // Reduced unigram LM lookahead, with check to eventually activate the lookahead based on depth

                applyLookaheadPerf_->start();
                applyLookaheadSparsePrePerf_->start();

                for (; sh != sh_end; ++sh) {
                    std::pair<uint, uint> ids = automaton_->lookAheadIds[sh->state];

                    if (ids.first <= fullLookaheadAfterId_ && (!sparseLookaheadSlowPropagation_ || sh->prospect != F32_MAX)) {
                        // Activate the full lookahead, as the active state is deeper than our depth threshold
                        verify(!instance.key.isTimeKey());
                        applyLookaheadPerf_->stop();
                        applyLookaheadSparsePrePerf_->stop();
                        activateLmLookahead(instance, true);  // Why always instance and no sh/ids?
                        applyLookaheadPerf_->start();
                        break;  // We will continue in the optimized loop without this check
                    }

                    sh->prospect = sh->score +
                                   unigramLookAhead_->scoreForLookAheadIdNormal(ids.first) +
                                   acousticLookAhead(ids.second, sh->state) +
                                   (useBackOffOffset ? backOffOffset : 0);
                    pruning.prepare(*sh);
                }

                applyLookaheadSparsePrePerf_->stop();
            }
            else {
                applyLookaheadPerf_->stop();
                activateLmLookahead(instance, true);
                applyLookaheadPerf_->start();
            }
        }
    }

    const AdvancedTreeSearch::LanguageModelLookahead::ContextLookahead* la(instance.lookahead.get());
    if (!la)
        la = unigramLookAhead_.get();
    else
        backOffOffset = 0;

    if (la->isSparse()) {
        applyLookaheadSparsePerf_->start();
        // Sparse LM lookahead

        if (!instance.backOffInstance && sh != sh_end) {
            instance.backOffInstance = getBackOffInstance(&instance);
            verify(instance.backOffInstance);
        }
        Score offset = instance.backOffScore;

        for (; sh != sh_end; ++sh) {
            std::pair<u32, u32> const& ids = automaton_->lookAheadIdAndHash[sh->state];

            Score lmScore = 0;
            bool  fail    = false;

            fail = !la->getScoreForLookAheadHashSparse(ids.first, lmScore);

            if (fail) {
                //This state needs to transfer into the back-off network

                // Set the prospect to max, so this state will be pruned away from this network
                sh->prospect = F32_MAX;
                // Add the back-off to the score
                if (earlyBackoff_) {
                    sh->score = F32_MAX;
                }
                else {
                    sh->score += offset;
                    // Transfer the state into the back-off network
                    instance.backOffInstance->transfer.push_back(sh - newStateHypotheses.begin());
                }
            }
            else {
                sh->prospect = sh->score + lmScore + acousticLookAhead(ids.second, sh->state);
                pruning.prepare(*sh);
            }
        }

        applyLookaheadSparsePerf_->stop();
    }
    else {
        applyLookaheadStandardPerf_->start();
        // Standard, non-sparse LM lookahead (use scoreForLookAheadIdNormal)
        for (; sh != sh_end; ++sh) {
            std::pair<uint, uint> ids = automaton_->lookAheadIds[sh->state];

            sh->prospect = sh->score + la->scoreForLookAheadIdNormal(ids.first) + acousticLookAhead(ids.second, sh->state) + (useBackOffOffset ? backOffOffset : 0);
            pruning.prepare(*sh);
        }
        applyLookaheadStandardPerf_->stop();
    }

    applyLookaheadPerf_->stop();
}

template<class Pruning, class AcousticLookAhead>
void SearchSpace::applyLookaheadInInstanceWithAcoustic(Instance* network, AcousticLookAhead& acousticLookAhead, Pruning& pruning) {
    if (sparseLookahead_) {
        if (unigramLookaheadBackoffFactor_ != 0)
            applyLookaheadInInstanceInternal<true, true, AcousticLookAhead, Pruning>(network, acousticLookAhead, pruning);
        else
            applyLookaheadInInstanceInternal<true, false, AcousticLookAhead, Pruning>(network, acousticLookAhead, pruning);
    }
    else {
        if (unigramLookaheadBackoffFactor_ != 0)
            applyLookaheadInInstanceInternal<false, true>(network, acousticLookAhead, pruning);
        else
            applyLookaheadInInstanceInternal<false, false>(network, acousticLookAhead, pruning);
    }
}

void SearchSpace::applyLookaheadInInstance(Instance* network) {
    if (perInstanceAcousticPruningScale_ < 1.0) {
        RecordMinimumPerInstance recordProspect(*this);

        if (acousticLookAhead_->isEnabled()) {
            AcousticLookAhead::ApplyPreCachedLookAheadForId lookahead(*acousticLookAhead_);
            applyLookaheadInInstanceWithAcoustic(network, lookahead, recordProspect);
        }
        else {
            AcousticLookAhead::ApplyNoLookahead nolookahead(*acousticLookAhead_);
            applyLookaheadInInstanceWithAcoustic(network, nolookahead, recordProspect);
        }
    }
    else {
        RecordMinimum recordProspect(*this);

        if (acousticLookAhead_->isEnabled()) {
            AcousticLookAhead::ApplyPreCachedLookAheadForId lookahead(*acousticLookAhead_);
            applyLookaheadInInstanceWithAcoustic(network, lookahead, recordProspect);
        }
        else {
            AcousticLookAhead::ApplyNoLookahead nolookahead(*acousticLookAhead_);
            applyLookaheadInInstanceWithAcoustic(network, nolookahead, recordProspect);
        }
    }
}

template<class Pruning>
void SearchSpace::addAcousticScoresInternal(Instance const& instance, Pruning& pruning, u32 from, u32 to) {
    StateHypothesesList::iterator sh     = stateHypotheses.begin() + from;
    StateHypothesesList::iterator sh_end = stateHypotheses.begin() + to;

    const Mm::CachedFeatureScorer::CachedContextScorerOverlay* scorerCache(dynamic_cast<const Mm::CachedFeatureScorer::CachedContextScorerOverlay*>(scorer_.get()));

    pruning.startInstance(instance.key);

    if (scorerCache) {
        // Omit overhead of a virtual function-call for cached scores by calling the score function directly with qualification
        for (; sh != sh_end; ++sh) {
            if (sh->prospect == F32_MAX)
                continue;  //This state will be pruned

            const HMMState&  state = network().structure.state(sh->state);
            Mm::MixtureIndex mix   = state.stateDesc.acousticModel;
            verify_(mix != StateTree::invalidAcousticModel);

            // Non-virtual call
            Score s = scorerCache->Mm::CachedFeatureScorer::CachedContextScorerOverlay::score(mix);

            sh->score += s;
            sh->prospect += s * acousticProspectFactor_;

            pruning.prepare(*sh);
        }
    }
    else {
        for (; sh != sh_end; ++sh) {
            if (sh->prospect == F32_MAX)
                continue;  //This state will be pruned

            const HMMState&  state = network().structure.state(sh->state);
            Mm::MixtureIndex mix   = state.stateDesc.acousticModel;
            verify_(mix != StateTree::invalidAcousticModel);

            // For segment-based models, we cannot recombine until we know the segment length, therefore, we cannot compute the acoustic score here,
            // but instead it should be done in activateOrUpdateStateHypothesisTransition() activateOrUpdateStateHypothesisTransitionLookup().
            Score s = scorer_->score(mix);

            sh->score += s;
            sh->prospect += s * acousticProspectFactor_;

            pruning.prepare(*sh);
        }
    }
}

template<class Pruning>
void SearchSpace::addAcousticScores() {
    verify(newStateHypotheses.empty());

    PerformanceCounter perf(*statistics, "addAcousticScores");

    bestProspect_ = Core::Type<Score>::max;
    bestScore_    = Core::Type<Score>::max;

    {
        Pruning pruning(*this);

        for (auto instance : activeInstances) {
            addAcousticScoresInternal(*instance, pruning, instance->states.begin, instance->states.end);
        }
    }

    verify(bestProspect_ != Core::Type<Score>::max || stateHypotheses.empty());
}

void SearchSpace::activateLmLookahead(Search::Instance& instance, bool compute) {
    if (instance.lookahead.get())
        return;

    if (instance.key.isTimeKey()) {
        instance.lookahead = unigramLookAhead_;
    }
    else {
        Instance& wt(static_cast<Instance&>(instance));

        if (wt.backOffParent) {
            // Compute the total back-off offset
            wt.totalBackOffOffset = wt.backOffParent->totalBackOffOffset + wt.backOffParent->backOffScore;
        }

        if (compute) {
            computeLookaheadPerf_->start();

            if (!wt.lookahead && (wt.lookaheadHistory.isValid() || wt.key.history.isValid())) {
                Lm::History h = wt.lookaheadHistory;
                if (!h.isValid())
                    h = wt.key.history;

                if (h == unigramHistory_) {
                    wt.lookahead = unigramLookAhead_;
                }
                else {
                    wt.lookahead = lmLookahead_->getLookahead(h, false);
                    lmLookahead_->fill(wt.lookahead, sparseLookahead_);
                }
            }
            computeLookaheadPerf_->stop();
        }
        else {
            if (wt.lookaheadHistory == unigramHistory_)
                instance.lookahead = unigramLookAhead_;
            else
                instance.lookahead = lmLookahead_->tryToGetLookahead(wt.lookaheadHistory.isValid() ? wt.lookaheadHistory : wt.key.history);
        }
    }
}

Score SearchSpace::bestProspect() const {
    if (bestProspect_ == Core::Type<Score>::max) {
        StateHypothesesList::const_iterator hyp = bestProspectStateHypothesis();
        if (hyp != stateHypotheses.end())
            bestProspect_ = (*hyp).prospect;
    }
    return bestProspect_;
}

Score SearchSpace::bestScore() const {
    if (bestScore_ == Core::Type<Score>::max) {
        StateHypothesesList::const_iterator hyp = bestScoreStateHypothesis();
        if (hyp != stateHypotheses.end())
            bestScore_ = (*hyp).score;
    }
    return bestScore_;
}

SearchSpace::StateHypothesesList::const_iterator
        SearchSpace::bestScoreStateHypothesis() const {
    StateHypothesesList::const_iterator ret       = stateHypotheses.begin();
    Score                               bestScore = Core::Type<Score>::max;
    for (StateHypothesesList::const_iterator sh = stateHypotheses.begin(); sh != stateHypotheses.end(); ++sh)
        if (bestScore > sh->score)
            bestScore = (ret = sh)->score;

    return ret;
}

SearchSpace::StateHypothesesList::const_iterator
        SearchSpace::bestProspectStateHypothesis() const {
    StateHypothesesList::const_iterator ret       = stateHypotheses.begin();
    Score                               bestScore = Core::Type<Score>::max;
    for (StateHypothesesList::const_iterator sh = stateHypotheses.begin(); sh != stateHypotheses.end(); ++sh) {
        if (bestScore > sh->prospect)
            bestScore = (ret = sh)->prospect;
    }

    return ret;
}

Score SearchSpace::quantileStateScore(Score minScore, Score maxScore, u32 nHyps) const {
    stateHistogram_.clear();
    stateHistogram_.setLimits(minScore, maxScore);

    for (StateHypothesesList::const_iterator sh = stateHypotheses.begin(); sh != stateHypotheses.end(); ++sh)
        stateHistogram_ += sh->prospect;

    return stateHistogram_.quantile(nHyps);
}

/// LM State pruning: based on the prospect score grouped by StateId state
void SearchSpace::pruneStatesPerLmState() {
    if (lmStatePruning_ >= acousticPruning_ ||
        lmStatePruning_ >= Core::Type<f32>::max)
        return;

    const u32 stateHypothesesSize = stateHypotheses.size();

    // Mark the best state hypothesis for each state using the recombination array
    // First pass: Find the best state hypothesis for each state using the recombination array (which is usually only used instance internally).
    for (u32 a = 0; a < stateHypotheses.size(); ++a) {
        const StateHypothesis& hyp(stateHypotheses[a]);

        StateHypothesisIndex& recombination(stateHypothesisRecombinationArray[hyp.state]);  // reference to best per-state token

        // We intentionally overflow here by using u32: If the value was would be negative, it will be > recombinationEnd
        u32 correctedRecombination = recombination - stateHypothesesSize;

        if (correctedRecombination >= stateHypothesesSize ||
            stateHypotheses[correctedRecombination].state != hyp.state ||
            stateHypotheses[correctedRecombination].prospect > hyp.prospect)
            recombination = stateHypothesesSize + a;  // Remember the best hypothesis per state as a new entry,
        // whose index also encodes the information, which StateHypothesis a was the best.
    }

    // Second pass: prune hypotheses below lmStatePruning_
    {
        u32 instIn, instOut;  // the index of the current active instance before and after pruning (some will be deactivated)

        StateHypothesesList::iterator hypIn, hypOut, hypBegin, instHypEnd;
        hypIn = hypOut = hypBegin = stateHypotheses.begin();                     // the old index and the index of the remaining hypotheses
        for (instIn = instOut = 0; instIn < activeInstances.size(); ++instIn) {  // go through all active instances
            Instance* at(activeInstances[instIn]);
            verify(hypIn == hypBegin + at->states.begin);
            at->states.begin = hypOut - hypBegin;  // first index in new, compressed array

            for (instHypEnd = hypBegin + at->states.end; hypIn < instHypEnd; ++hypIn) {  // go through all hypotheses in the instance
                verify_(hypIn < stateHypotheses.end());

                StateHypothesisIndex bestHypIndex = stateHypothesisRecombinationArray[hypIn->state] - stateHypothesesSize;  // get index of the best per-state hypothesis

                if (bestHypIndex == hypIn - stateHypotheses.begin())  // compare to the encoded number (a, the best per-state hypothesis)
                {
                    // This is the best hypothesis. Update the index so it points to the moved (compressed) state hypothesis
                    stateHypothesisRecombinationArray[hypIn->state] = (hypOut - stateHypotheses.begin()) + stateHypothesesSize;  // is this updated index only needed for the third pass?
                    *(hypOut++)                                     = *hypIn;
                }
                else {  // prune all other hypotheses
                    if (hypIn->prospect <= stateHypotheses[bestHypIndex].prospect + lmStatePruning_)
                        *(hypOut++) = *hypIn;
                }
            }

            at->states.end = hypOut - hypBegin;
            if (!eventuallyDeactivateTree(at, true)) {
                activeInstances[instOut++] = at;
            }
        }

        stateHypotheses.erase(hypOut, stateHypotheses.end());

        activeInstances.resize(instOut);
    }

    if (PathTrace::Enabled) {
        for (std::vector<StateHypothesis>::iterator it = stateHypotheses.begin(); it != stateHypotheses.end(); ++it) {
            StateHypothesisIndex index = stateHypothesisRecombinationArray[it->state] - stateHypothesesSize;
            it->pathTrace.maximizeOffset("lm-state-pruning", it->prospect - stateHypotheses[index].prospect);
        }
    }
}

/// Standard pruning:
template<class Pruning>
void SearchSpace::pruneStates(Pruning& pruning) {
    u32 instIn, instOut;

    StateHypothesesList::iterator hypIn, hypOut, hypBegin, instHypEnd;
    hypIn = hypOut = hypBegin = stateHypotheses.begin();
    for (instIn = instOut = 0; instIn < activeInstances.size(); ++instIn) {
        Instance* at(activeInstances[instIn]);
        pruning.startInstance(at->key);
        verify(hypIn == hypBegin + at->states.begin);
        at->states.begin = hypOut - hypBegin;

        for (instHypEnd = hypBegin + at->states.end; hypIn < instHypEnd; ++hypIn) {
            verify_(hypIn < stateHypotheses.end());
            if (!pruning.prune(trace_manager_, *hypIn))
                *(hypOut++) = *hypIn;
        }

        at->states.end = hypOut - hypBegin;
        if (!eventuallyDeactivateTree(at, true)) {
            activeInstances[instOut++] = at;
        }
    }

    stateHypotheses.erase(hypOut, stateHypotheses.end());

    activeInstances.resize(instOut);
}


void SearchSpace::updateSsaLm() {
    if (!ssaLm_) {
        return;
    }

    for (Instance* inst : activeInstances) {
        Lm::SearchSpaceInformation info;
        info.minLabelDistance = std::numeric_limits<unsigned>::max();
        info.bestScore        = std::numeric_limits<Score>::max();
        for (auto hypIter = stateHypotheses.begin() + inst->states.begin; hypIter < stateHypotheses.begin() + inst->states.end; ++hypIter) {
            info.minLabelDistance = std::min(info.minLabelDistance, automaton_->labelDistance[hypIter->state]);
            info.bestScore        = std::min(info.bestScore, hypIter->score);
        }
        info.bestScoreOffset = info.bestScore - bestScore();
        info.numStates       = inst->states.size() + inst->rootStateHypotheses.size();
        ssaLm_->setInfo(inst->scoreHistory, info);
    }
}

void SearchSpace::filterStates() {
    if (!automaton_->prefixFilter) {
        return;
    }

    PerformanceCounter perf(*statistics, "filter states");

    pruneStates(*automaton_->prefixFilter);
}

void SearchSpace::enforceCommonPrefix() {
    if (maximumMutableSuffixPruningInterval_ <= 0 or timeFrame_ % maximumMutableSuffixPruningInterval_ != maximumMutableSuffixPruningInterval_ - 1) {
        return;
    }

    PerformanceCounter perf(*statistics, "enforce common prefix");

    // find best trace
    Score   best_prospect = Core::Type<Score>::max;
    TraceId best_trace    = invalidTraceId;
    for (StateHypothesis const& hyp : stateHypotheses) {
        if (hyp.prospect < best_prospect) {
            best_prospect = hyp.prospect;
            best_trace    = hyp.trace;
        }
    }
    if (best_trace == invalidTraceId) {
        return;
    }

    // find root trace for all surviving
    int              remaining_lemmas = maximumMutableSuffixLength_;
    Core::Ref<Trace> root             = trace_manager_.traceItem(best_trace).trace;
    while (root) {
        size_t                           max_length = 0;
        Bliss::LemmaPronunciation const* pron       = root->pronunciation;
        if (pron and reinterpret_cast<uintptr_t>(pron) != 1) {
            Bliss::Lemma const* lemma = pron->lemma();
            if (lemma->hasEvaluationTokenSequence()) {
                for (auto eval_seqs = lemma->evaluationTokenSequences(); eval_seqs.first != eval_seqs.second; ++eval_seqs.first) {
                    max_length = std::max(max_length, eval_seqs.first->size());
                }
            }
        }
        remaining_lemmas -= max_length;
        if (remaining_lemmas > 0 or max_length == 0) {
            root = root->predecessor;
        }
        else {
            break;
        }
    }

    if (not root) {
        return;  // nothing to do, utterance shorter than maximumMutableSuffixLength_
    }

    PerformanceCounter perf2(*statistics, "enforce common prefix - pruning");
    BestTracePruning   pruning(root);
    pruneStates<BestTracePruning>(pruning);
}

// early acoustic pruning
void SearchSpace::pruneStatesEarly() {
    if (!earlyBeamPruning_)
        return;

    PerformanceCounter perf(*statistics, "early acoustic pruning");

    verify(bestProspect_ != Core::Type<Score>::max || stateHypotheses.empty());

    AcousticPruning pruning(*this, acousticPruning_);
    pruneStates(pruning);
}

void SearchSpace::pruneAndAddScores() {
    statistics->treesBeforePruning += nActiveTrees();
    statistics->statesBeforePruning += nStateHypotheses();

    doStateStatisticsBeforePruning();

    filterStates();  // pruneStates using automaton_->prefixFilter -> necessary for incremental decoding of partial segments, where we initialize the prefix
    enforceCommonPrefix();
    // works with pruneStates() on stateHypotheses
    pruneStatesEarly();  // before applying acoustic scores, uses pruneStates() with score AcousticPruning_

    statistics->treesAfterPrePruning += nActiveTrees();
    statistics->statesAfterPrePruning += nStateHypotheses();

    if (perInstanceAcousticPruningScale_ < 1.0) {
        addAcousticScores<RecordMinimumPerInstance>();
        PerformanceCounter         perf(*statistics, "acoustic pruning");
        PerInstanceAcousticPruning pruning(*this);
        pruneStates(pruning);
    }
    else {
        addAcousticScores<RecordMinimum>();
        PerformanceCounter perf(*statistics, "acoustic pruning");
        AcousticPruning    pruning(*this);
        pruneStates(pruning);
    }

    {
        PerformanceCounter perf(*statistics, "other pruning");

        pruneStatesPerLmState();

        // Histogram pruning
        if ((nStateHypotheses() > acousticPruningLimit_) && acousticPruning_) {
            Score acuThreshold = quantileStateScore(bestProspect_, bestProspect_ + acousticPruning_, acousticPruningLimit_);
            statistics->acousticHistogramPruningThreshold += acuThreshold - bestProspect_;
            AcousticPruning pruning(*this, acuThreshold - bestProspect_);
            pruneStates(pruning);

            currentAcousticPruningSaturation += 1.0;
            statistics->customStatistics("acoustic pruning saturation") += 1.0;
        }
        else {
            currentAcousticPruningSaturation += 0.0;
            statistics->customStatistics("acoustic pruning saturation") += 0.0;
        }
    }

    // now that pruning is done we can update the lm (if necessary)
    updateSsaLm();

    // Append time/score modifications to state traces to obtain correct word timings.
    correctPushedTransitions();

    statistics->treesAfterPruning += nActiveTrees();
    statistics->statesAfterPruning += nStateHypotheses();
    currentStatesAfterPruning += nStateHypotheses();

    doStateStatistics();
}

void SearchSpace::correctPushedTransitions() {
    if (!correctPushedBoundaryTimes_ || !automaton_->minimized)
        return;

    PersistentStateTree const& net = network();
    PerformanceCounter         perf(*statistics, "correct pushed boundaries");

    int alreadyCorrect = 0, corrected = 0, candidates = 0;

    bool encodeState = this->encodeState();

    s32 rootDepth = automaton_->truncatedStateDepths[net.rootState];

    for (StateHypothesesList::iterator it = stateHypotheses.begin(); it != stateHypotheses.end(); ++it) {
        TraceId& trace((*it).trace);
        if (automaton_->truncatedStateDepths[it->state] == rootDepth) {  // after fanout
            ++corrected;
            const Search::Trace& traceItem(*trace_manager_.traceItem(trace).trace);
            int                  timeDifference  = 1 + (int)timeFrame_ - (int)traceItem.time;
            u32                  scoreDifference = 0;

            if (correctPushedAcousticScores_) {
                Score d         = (*it).score + globalScoreOffset_ - traceItem.score;
                scoreDifference = reinterpret_cast<u32&>(d);
            }

            trace = trace_manager_.modify(trace_manager_.getUnmodified(trace), timeDifference, scoreDifference, encodeState ? it->state : 0);
        }
        else if (!trace_manager_.isModified(trace)) {
            if (automaton_->truncatedStateDepths[it->state] >= rootDepth)  // after fanout
            {
                ++corrected;
                const Search::Trace& traceItem(*trace_manager_.traceItem(trace).trace);
                int                  timeDifference = (int)timeFrame_ - (int)traceItem.time;
                verify(timeDifference >= 0);

                u32 scoreDifference = 0;

                if (correctPushedAcousticScores_ && timeDifference > 0) {
                    // We need to subtract the acoustic score of this timeframe, as that one should be accounted to this word already
                    Score currentAcousticScore = scorer_->score(net.structure.state(it->state).stateDesc.acousticModel);
                    Score d                    = (*it).score + globalScoreOffset_ - currentAcousticScore - traceItem.score;
                    scoreDifference            = reinterpret_cast<u32&>(d);
                }

                trace = trace_manager_.modify(trace, timeDifference, scoreDifference, encodeState ? it->state : 0);
            }
            else {  // still in fanout
                ++candidates;
            }
        }
        else {
            ++alreadyCorrect;
        }
    }

    statistics->customStatistics("state traces behind fanout already correct") += alreadyCorrect;
    statistics->customStatistics("state traces behind fanout corrected") += corrected;
    statistics->customStatistics("state traces still in fan-out") += candidates;
}

void SearchSpace::rescale(Score offset, bool ignoreWordEnds) {
    require(ignoreWordEnds || wordEndHypotheses.size() == 0);
    require(earlyWordEndHypotheses.size() == 0);
    for (StateHypothesesList::iterator sh = stateHypotheses.begin(); sh != stateHypotheses.end(); ++sh) {
        sh->score -= offset;
        sh->prospect -= offset;
    }

    minWordEndScore_ -= offset;

    verify(newStateHypotheses.empty());

    globalScoreOffset_ += offset;
    if (bestProspect_ != Core::Type<Score>::max)
        bestProspect_ -= offset;
    if (bestScore_ != Core::Type<Score>::max)
        bestScore_ -= offset;
}

Score SearchSpace::minimumWordEndScore() const {
    return minWordEndScore_;
}

Score SearchSpace::quantileWordEndScore(Score minScore, Score maxScore, u32 nHyps) const {
    wordEndHistogram_.clear();
    wordEndHistogram_.setLimits(minScore, maxScore);
    for (WordEndHypothesisList::const_iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh)
        wordEndHistogram_ += weh->score;

    return wordEndHistogram_.quantile(nHyps);
}

inline Core::Ref<Trace> SearchSpace::getModifiedTrace(TraceId traceId, const Bliss::Phoneme::Id initial) const {
    TraceItem const& item(trace_manager_.traceItem(traceId));
    Core::Ref<Trace> trace = item.trace;

    if (trace_manager_.isModified(traceId)) {
        bool encodeState = this->encodeState();

        verify(trace.get());
        SearchAlgorithm::TracebackItem::Transit transit;
        TraceManager::Modification              offsets = trace_manager_.getModification(traceId);
        if (offsets.first || offsets.second || offsets.third) {
            // Add an epsilon traceback entry which corrects the time, score and transit-entry
            TimeframeIndex time = trace->time + offsets.first;
            verify(time <= timeFrame_);
            SearchAlgorithm::ScoreVector score = trace->score;
            if (offsets.second)
                score.acoustic += reinterpret_cast<Search::Score&>(offsets.second);

            if (encodeState) {
                transit = describeRootState(offsets.third);
            }
            else if (!encodeState && trace->pronunciation && trace->pronunciation->pronunciation()->length() && initial != Bliss::Phoneme::term) {
                Bliss::Phoneme::Id final = trace->pronunciation->pronunciation()->phonemes()[trace->pronunciation->pronunciation()->length() - 1];
                if (lexicon_->phonemeInventory()->phoneme(final)->isContextDependent() && lexicon_->phonemeInventory()->phoneme(initial)->isContextDependent()) {
                    transit.final   = final;
                    transit.initial = initial;
                }
            }
            trace = Core::Ref<Trace>(new Trace(trace, epsilonLemmaPronunciation(), time, score, transit));
        }
    }
    return trace;
}

void SearchSpace::pruneEarlyWordEnds() {
    Score absoluteProspectThreshold = minWordEndScore_ + std::min(acousticPruning_, wordEndPruning_);

    PersistentStateTree const& net         = network();
    std::vector<int> const&    stateDepths = automaton_->stateDepths;

    PerformanceCounter perf(*statistics, "prune early word ends");

    bool               doPhonemePruning = wordEndPhonemePruningThreshold_ < wordEndPruning_;
    u32                nPhonemes        = lexicon_->phonemeInventory()->nPhonemes();
    std::vector<Score> thresholdsPerGroup(nPhonemes + 1, Core::Type<Score>::max);
    std::vector<u32>   groupCount(nPhonemes + 1, 0);
    std::vector<s32>   groups;

    // Expand surviving EarlyWordEndHypotheses to WordEndHypotheses.
    for (EarlyWordEndHypothesisList::iterator in = earlyWordEndHypotheses.begin(); in != earlyWordEndHypotheses.end(); ++in) {
        if (in->score <= absoluteProspectThreshold) {
            // Expand the _fat_ word end hypotheses
            const PersistentStateTree::Exit* we   = &net.exits[in->exit];
            const Bliss::LemmaPronunciation* pron = (we->pronunciation == Bliss::LemmaPronunciation::invalidId) ? 0 : lexicon_->lemmaPronunciation(we->pronunciation);

            TraceItem const&  traceItem = trace_manager_.traceItem(in->trace);
            WordEndHypothesis end(traceItem.recombinationHistory,
                                  traceItem.lookaheadHistory,
                                  traceItem.scoreHistory,
                                  we->transitState,
                                  pron,
                                  in->score,
                                  getModifiedTrace(in->trace, (pron && pron->pronunciation()->length()) ? pron->pronunciation()->phonemes()[0] : Bliss::Phoneme::term),
                                  in->exit,
                                  in->pathTrace);

            if (end.pronunciation) {
                extendHistoryByLemma(end, end.pronunciation->lemma());
            }

            if (doPhonemePruning) {
                StateId transit  = net.exits[in->exit].transitState;
                bool    isPushed = stateDepths[transit] < stateDepths[net.rootState];

                Bliss::Phoneme::Id group = nPhonemes;

                if (!isPushed) {
                    std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::const_iterator it = net.rootTransitDescriptions.find(transit);
                    verify(it != net.rootTransitDescriptions.end());
                    group = it->second.second;
                }
                else {
                    group = nPhonemes;
                }
                groupCount[group] += 1;
                groups.push_back(group);
                verify(group >= 0 && group < thresholdsPerGroup.size());
                if (thresholdsPerGroup[group] > in->score)
                    thresholdsPerGroup[group] = in->score;
            }

            wordEndHypotheses.push_back(end);
        }
    }

    if (doPhonemePruning) {
        // Record the best score per first-phoneme
        verify(!net.unpushedCoarticulatedRootStates.empty());
        verify(groups.size() == wordEndHypotheses.size());

        u32 phoneSum  = 0;
        u32 phoneMost = 0;
        for (u32 i = 0; i < nPhonemes; ++i) {
            u32 count = groupCount[i];
            phoneSum += count;
            if (count > phoneMost)
                phoneMost = count;
        }

        statistics->customStatistics("unpushed word-ends before first-phoneme pruning") += phoneSum;
        statistics->customStatistics("pushed word-ends before first-phoneme pruning") += groupCount[nPhonemes];

        if (phoneSum) {
            f32 dominance = ((f32)phoneMost) / ((f32)phoneSum);
            statistics->customStatistics("unpushed word-end phoneme dominace") += dominance;
        }

        for (u32 i = 0; i < nPhonemes; ++i) {
            if (thresholdsPerGroup[i] != Core::Type<Score>::max) {
                if (wordEndPhonemePruningThreshold_ < wordEndPruning_)
                    thresholdsPerGroup[i] += wordEndPhonemePruningThreshold_;
                else
                    thresholdsPerGroup[i] = Core::Type<Score>::max;
            }
        }

        thresholdsPerGroup[nPhonemes] = Core::Type<Score>::max;

        WordEndHypothesisList::iterator out = wordEndHypotheses.begin();
        for (WordEndHypothesisList::iterator in = wordEndHypotheses.begin(); in != wordEndHypotheses.end(); ++in) {
            int group = groups[in - wordEndHypotheses.begin()];
            if (in->score < thresholdsPerGroup[group])
                *(out++) = *in;
        }

        statistics->customStatistics("word-ends removed by first-phoneme pruning") += (wordEndHypotheses.end() - out);

        wordEndHypotheses.erase(out, wordEndHypotheses.end());
    }

    // Histogram word end pruning
    if (nWordEndHypotheses() > wordEndPruningLimit_) {
        Score minWordEndScore = minimumWordEndScore();
        Score threshold       = quantileWordEndScore(minWordEndScore, minWordEndScore + wordEndPruning_, wordEndPruningLimit_);
        statistics->lmHistogramPruningThreshold += threshold - minWordEndScore;
        pruneWordEnds(threshold);
    }

    earlyWordEndHypotheses.reserve(earlyWordEndHypotheses.capacity());
    earlyWordEndHypotheses.clear();

    statistics->wordEndsAfterPruning += nWordEndHypotheses();
    currentWordEndsAfterPruning += nWordEndHypotheses();
}

void SearchSpace::pruneWordEnds(Score absoluteScoreThreshold) {
    WordEndHypothesisList::iterator in, out;
    for (in = out = wordEndHypotheses.begin(); in != wordEndHypotheses.end(); ++in)
        if (in->score <= absoluteScoreThreshold)
            *(out++) = *in;

    wordEndHypotheses.erase(out, wordEndHypotheses.end());
}

void SearchSpace::createTraces(TimeframeIndex time) {
    // Eventually extend the lm states too
    for (WordEndHypothesisList::iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh) {
        if (weh->pronunciation) {
            weh->trace = Core::ref(new Trace(weh->trace, weh->pronunciation, time, weh->score, describeRootState(weh->transitState)));
            weh->trace->score.acoustic += globalScoreOffset_;

            // Don't allow negative per-word LM scores (may happen in conjunction with negative pronunciation scores)
            Score ownLmScore = weh->trace->score.lm;
            Score preLmScore = weh->trace->predecessor->score.lm;
            if (ownLmScore < preLmScore) {
                weh->score.lm = weh->trace->score.lm = preLmScore;

                if (overflowLmScoreToAm_) {
                    // We don't want negative scores, so only overflow if we can according to the global score offset
                    Score offset = preLmScore - ownLmScore;
                    if (offset < weh->score.acoustic) {
                        weh->trace->score.acoustic -= offset;
                        weh->score.acoustic -= offset;
                        if (weh->trace->score.acoustic < weh->trace->predecessor->score.acoustic) {
                            // Don't allow negative per-word scores (may happen in conjunction with negative pronunciation scores)
                            weh->trace->score.acoustic = weh->trace->predecessor->score.acoustic;
                            weh->score.acoustic        = weh->trace->score.acoustic - globalScoreOffset_;
                        }
                    }
                }
            }
            weh->trace->pathTrace = weh->pathTrace;
        }
    }
}

void SearchSpace::hypothesizeEpsilonPronunciations(Score bestScore) {
    PersistentStateTree const& net               = network();
    std::vector<int> const&    singleLabels      = automaton_->singleLabels;
    std::vector<u32> const&    quickLabelBatches = automaton_->quickLabelBatches;
    std::vector<s32> const&    slowLabelBatches  = automaton_->slowLabelBatches;

    u32 nWordEnds  = wordEndHypotheses.size();
    u32 considered = 0;

    Score threshold = bestScore + wordEndPruning_;

    PerformanceCounter perf(*statistics, "hypothesize epsilon pronunciations");

    for (u32 w = 0; w < nWordEnds; ++w) {
        StateId transit = wordEndHypotheses[w].transitState;
        if (singleLabels[transit] == -1)  // There are no outputs on the state.
            continue;

        // Get all outputs of the transit state.
        u32 exitsStart, exitsEnd;

        if (singleLabels[transit] >= 0)  // There is a single output on the state.
        {
            exitsStart = singleLabels[transit];
            exitsEnd   = exitsStart + 1;
        }
        else if (singleLabels[transit] == -2)  // There are multiple outputs on the state, on fast batches.
        {
            exitsStart = quickLabelBatches[net.rootState];
            exitsEnd   = quickLabelBatches[net.rootState + 1];
        }
        else {  // Negative number: There are multiple outputs on the state, on slow batches. (List with terminator -1).
            for (s32 current = -(singleLabels[transit] + 3); slowLabelBatches[current] != -1; ++current) {
                u32 exit = slowLabelBatches[current];

                /// @todo Un-copy this code
                const PersistentStateTree::Exit& wordEnd       = net.exits[exit];
                const Bliss::LemmaPronunciation* pronunciation = lexicon_->lemmaPronunciation(wordEnd.pronunciation);
                if (!pronunciation)
                    continue;

                WordEndHypothesis weh(wordEndHypotheses[w]);
                weh.pronunciation = pronunciation;
                weh.transitState  = wordEnd.transitState;

                std::unordered_map<InstanceKey, Instance*, InstanceKey::Hash>::iterator instIt = activeInstanceMap.find(InstanceKey(weh.recombinationHistory));
                if (instIt != activeInstanceMap.end()) {
                    // Use the network's cache to extend the LM score
                    static_cast<Instance&>(*instIt->second).addLmScore(weh, pronunciation->id(), lm_, lexicon_, wpScale_);
                }
                else {
                    // Go on without a cache
                    Lm::addLemmaPronunciationScoreOmitExtension(lm_, weh.pronunciation, wpScale_, lm_->scale(), weh.scoreHistory, weh.score.lm);
                }

                weh.score.acoustic += (*transitionModel(net.structure.state(transit).stateDesc))[Am::StateTransitionModel::exit];
                ++considered;
                if (weh.score <= threshold) {
                    extendHistoryByLemma(weh, weh.pronunciation->lemma());

                    weh.trace = Core::ref(new Trace(weh.trace, weh.pronunciation, weh.trace->time, weh.score, describeRootState(wordEnd.transitState)));
                    weh.trace->score.acoustic += globalScoreOffset_;
                    wordEndHypotheses.push_back(weh);
                }
            }
            continue;
        }

        // Consecutive outputs between exitsStart and exitsEnd
        for (u32 exit = exitsStart; exit != exitsEnd; ++exit) {
            const PersistentStateTree::Exit& wordEnd       = net.exits[exit];
            const Bliss::LemmaPronunciation* pronunciation = lexicon_->lemmaPronunciation(wordEnd.pronunciation);
            if (!pronunciation)
                continue;

            WordEndHypothesis weh(wordEndHypotheses[w]);
            weh.pronunciation = pronunciation;
            weh.transitState  = wordEnd.transitState;

            std::unordered_map<InstanceKey, Instance*, InstanceKey::Hash>::iterator instIt = activeInstanceMap.find(InstanceKey(weh.recombinationHistory));
            if (instIt != activeInstanceMap.end()) {
                // Use the network's cache to extend the LM score
                static_cast<Instance&>(*instIt->second).addLmScore(weh, pronunciation->id(), lm_, lexicon_, wpScale_);
            }
            else {
                // Go on without a cache
                Lm::addLemmaPronunciationScoreOmitExtension(lm_, weh.pronunciation, wpScale_, lm_->scale(), weh.scoreHistory, weh.score.lm);
            }

            weh.score.acoustic += (*transitionModel(net.structure.state(transit).stateDesc))[Am::StateTransitionModel::exit];
            ++considered;
            if (weh.score <= threshold) {
                extendHistoryByLemma(weh, weh.pronunciation->lemma());

                weh.trace = Core::ref(new Trace(weh.trace, weh.pronunciation, weh.trace->time, weh.score, describeRootState(wordEnd.transitState)));
                weh.trace->score.acoustic += globalScoreOffset_;
                wordEndHypotheses.push_back(weh);
            }
        }
    }

    statistics->epsilonWordEndsAdded += wordEndHypotheses.size() - nWordEnds;
    statistics->customStatistics("epsilon word ends considered") += considered;
}

/**
 * Remove sibling traces that are silence.
 */
void SearchSpace::pruneSilenceSiblingTraces(
        Core::Ref<Trace> trace, const Bliss::Lemma* silence) {
    for (Core::Ref<Trace> tr = trace; tr->sibling;) {
        if (tr->sibling->pronunciation->lemma() == silence)
            tr->sibling = tr->sibling->sibling;
        else
            tr = tr->sibling;
    }
}

/**
 * This is the simple lattice optimization selected by optimize-lattice=simple.
 * The effect is this: All partial sentence hypotheses ending with
 * silence are suppressed from the lattice - except that the best
 * scoring hypothesis is always preserved, even if it ends with
 * silence.
 */
void SearchSpace::optimizeSilenceInWordLattice(
        const Bliss::Lemma* silence) {
    for (WordEndHypothesisList::iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh)
        pruneSilenceSiblingTraces(weh->trace, silence);
}

StateId SearchSpace::rootForCoarticulation(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id> coarticulation) const {
    PersistentStateTree const& net = automaton_->network;  // TODO(beck): change to network() once this function is const

    if (coarticulation.first == Bliss::Phoneme::term && coarticulation.second == Bliss::Phoneme::term)
        return net.rootState;

    //   std::cout << "coarticulation " << coarticulation.first << " " << coarticulation.second << std::endl;

    bool encodeState = this->encodeState();

    if (encodeState) {
        union {
            struct {
                Bliss::Phoneme::Id first, second;
            } coart;
            StateId rootState;
        };
        coart.first  = coarticulation.first;
        coart.second = coarticulation.second;

        verify(rootState & (1 << 31));
        rootState &= ((1u << 31) - 1);
        assert(rootState != 0 && rootState < net.structure.stateCount());
        return rootState;
    }

    StateId rootState = 0;
    for (PersistentStateTree::RootTransitDescriptions::const_iterator it = net.rootTransitDescriptions.begin(); it != net.rootTransitDescriptions.end(); ++it) {
        if (it->second == coarticulation) {
            if (rootState) {
                Core::Application::us()->criticalError() << "root coarticulation is ambiguous: " << (coarticulation.first == Bliss::Phoneme::term ? "#" : lexicon_->phonemeInventory()->phoneme(coarticulation.first)->symbol().str()) << ":" << (coarticulation.second == Bliss::Phoneme::term ? "#" : lexicon_->phonemeInventory()->phoneme(coarticulation.second)->symbol().str());
            }
            rootState = it->first;
        }
    }
    if (!rootState) {
        Core::Application::us()->criticalError() << "found no root state for coarticulation: " << (coarticulation.first == Bliss::Phoneme::term ? "#" : lexicon_->phonemeInventory()->phoneme(coarticulation.first)->symbol().str()) << ":" << (coarticulation.second == Bliss::Phoneme::term ? "#" : lexicon_->phonemeInventory()->phoneme(coarticulation.second)->symbol().str());
    }
    return rootState;
}

void SearchSpace::addStartupWordEndHypothesis(TimeframeIndex time) {
    Lm::History rch = recombinationLm_->startHistory();
    Lm::History lah = lookaheadLm_->startHistory();
    Lm::History sch = lm_->startHistory();
    for (std::vector<const Bliss::Lemma*>::const_iterator it = recognitionContext_.prefix.begin(); it != recognitionContext_.prefix.end(); ++it) {
        const Bliss::SyntacticTokenSequence tokenSequence((*it)->syntacticTokenSequence());
        for (u32 ti = 0; ti < tokenSequence.length(); ++ti) {
            const Bliss::SyntacticToken* st = tokenSequence[ti];
            rch                             = recombinationLm_->extendedHistory(rch, st);
            lah                             = lookaheadLm_->extendedHistory(lah, st);
            sch                             = lm_->extendedHistory(sch, st);
        }
    }

    StateId rootState = rootForCoarticulation(recognitionContext_.coarticulation);

    if (rootState == 0)
        Core::Application::us()->error() << "failed finding coarticulated root-state for coarticulation";

    verify(rch.isValid());
    verify(lah.isValid());
    verify(sch.isValid());
    SearchAlgorithm::ScoreVector score(0.0, 0.0);
    Core::Ref<Trace>             t(new Trace(time, score, describeRootState(rootState)));
    t->score.acoustic += globalScoreOffset_;
    wordEndHypotheses.push_back(WordEndHypothesis(rch, lah, sch, rootState, 0, score, t, Core::Type<u32>::max, PathTrace()));
}

void SearchSpace::dumpWordEnds(
        std::ostream& os, Core::Ref<const Bliss::PhonemeInventory> phi) const {
    for (WordEndHypothesisList::const_iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh) {
        os << "trace:" << std::endl;
        weh->trace->write(os, phi);
        os << "recombination history: " << weh->recombinationHistory.format() << std::endl
           << "lookahead history:     " << weh->lookaheadHistory.format() << std::endl
           << "score history:         " << weh->scoreHistory.format() << std::endl
           << "transit entry:         " << weh->transitState << std::endl
           << std::endl;
    }
}

std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id> SearchSpace::describeRootState(StateId state) const {
    PersistentStateTree const& net = automaton_->network;  // TODO(beck): change to network() once this function is const

    if (encodeState()) {
        union {
            struct {
                Bliss::Phoneme::Id first, second;
            } coart;
            StateId rootState;
        };
        rootState = state | (1 << 31);
        return std::make_pair(coart.first, coart.second);
    }
    std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::const_iterator it = net.rootTransitDescriptions.find(state);
    if (it != net.rootTransitDescriptions.end())
        return (*it).second;
    else
        return std::make_pair(Bliss::Phoneme::term, Bliss::Phoneme::term);
}

/**
 * Find the best sentence end hypothesis.
 * Returns a back-trace reference which can be used to determine the
 * word sequence.  The returned reference immediately points a
 * trace-back item which includes the language model sentence end
 * score.  Its predecessor indicates the final word of the sentence.
 * In case no word-end hypothesis is active, a null reference is returned.
 * If creation of the word lattice is requested, the returned trace
 * has siblings corresponding to sub-optimal sentence ends.
 */

Core::Ref<Trace> SearchSpace::getSentenceEnd(TimeframeIndex time, bool shallCreateLattice) {
    PersistentStateTree const& net = network();

    if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::No)
        shallCreateLattice = false;
    else if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::Yes)
        shallCreateLattice = true;

    Core::Ref<Trace> best;
    Score            bestScore = Core::Type<Score>::max;

    StateId forceRoot = 0;
    if (recognitionContext_.finalCoarticulation.first != Bliss::Phoneme::term || recognitionContext_.finalCoarticulation.second != Bliss::Phoneme::term)
        forceRoot = rootForCoarticulation(recognitionContext_.finalCoarticulation);

    for (WordEndHypothesisList::const_iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh) {
        if (forceRoot)  // Either we force all hypotheses to be in a certain state
        {
            if (weh->transitState != forceRoot)
                continue;  // do not allow mismatching sentence end
        }
        else {
            if (weh->transitState != net.rootState && weh->transitState != net.ciRootState &&
                !net.uncoarticulatedWordEndStates.count(weh->transitState))
                continue;  // do not allow coarticulated sentence end
        }
        Core::Ref<Trace> t(new Trace(weh->trace, 0, time, weh->score, describeRootState(weh->transitState)));

        t->score.acoustic += globalScoreOffset_;

        Lm::History h(weh->scoreHistory);
        verify(h.isValid());

        for (std::vector<const Bliss::Lemma*>::const_iterator it = recognitionContext_.suffix.begin(); it != recognitionContext_.suffix.end(); ++it) {
            Lm::addLemmaScore(lm_, *it, lm_->scale(), h, t->score.lm);
        }

        t->score.lm += lm_->sentenceEndScore(h);
        t->pathTrace = weh->pathTrace;

        if (!best || best->score > t->score) {  // Make the selection deterministic if scores are equal
            if (shallCreateLattice)
                t->sibling = best;
            best      = t;
            bestScore = t->score;
        }
        else {
            if (shallCreateLattice) {
                if (not onTheFlyRescoring_) {
                    t->sibling    = best->sibling;
                    best->sibling = t;
                }
                else {  // sorted siblings for on the fly rescoring
                    Core::Ref<Trace> trace = best;
                    while (trace->sibling and trace->sibling->score < t->score) {
                        trace = trace->sibling;
                    }
                    t->sibling     = trace->sibling;
                    trace->sibling = t;
                }
            }
        }
    }

    verify(!forceRoot || net.uncoarticulatedWordEndStates.size() || net.coarticulatedRootStates.count(forceRoot));

    u32 activeUncoartic = 0;

    if (net.uncoarticulatedWordEndStates.size()) {
        bool encodeState = this->encodeState();
        // We expect early-recombination to be used when this is enabled.
        // Take the trace from the best word-end states

        for (std::vector<Instance*>::iterator instIt = activeInstances.begin(); instIt != activeInstances.end(); ++instIt) {
            Instance& at(static_cast<Instance&>(**instIt));

            for (StateHypothesesList::iterator it = stateHypotheses.begin() + at.states.begin; it != stateHypotheses.begin() + at.states.end; ++it) {
                if (forceRoot) {
                    if (it->state != forceRoot)
                        continue;
                }
                else {
                    if (!net.uncoarticulatedWordEndStates.count((*it).state))
                        continue;
                    else
                        ++activeUncoartic;
                }
                Score                        score = (*it).score + globalScoreOffset_;
                SearchAlgorithm::ScoreVector scores(trace_manager_.traceItem((*it).trace).trace->score);
                scores.acoustic = score - scores.lm - at.totalBackOffOffset;

                // Append score- and time correcting epsilon item
                Core::Ref<Trace> t(new Trace(trace_manager_.traceItem((*it).trace).trace,
                                             epsilonLemmaPronunciation(),
                                             time - 1,
                                             scores,
                                             encodeState ? describeRootState(it->state) : SearchAlgorithm::TracebackItem::Transit()));
                // Append sentence-end epsilon arc
                t = Core::Ref<Trace>(new Trace(t, 0, time, t->score, describeRootState(net.rootState)));

                Lm::History h(trace_manager_.traceItem((*it).trace).scoreHistory);
                verify(h.isValid());

                for (std::vector<const Bliss::Lemma*>::const_iterator it = recognitionContext_.suffix.begin(); it != recognitionContext_.suffix.end(); ++it)
                    Lm::addLemmaScore(lm_, *it, lm_->scale(), h, t->score.lm);

                t->score.lm += lm_->sentenceEndScore(h);

                if (t->score < bestScore) {
                    if (shallCreateLattice)
                        t->sibling = best;
                    // Create sentence-end trace item, and store it as best
                    bestScore = t->score;
                    best      = t;
                }
                else if (shallCreateLattice) {
                    if (not onTheFlyRescoring_) {
                        t->sibling    = best->sibling;
                        best->sibling = t;
                    }
                    else {  // sorted siblings for on the fly rescoring
                        Core::Ref<Trace> trace = best;
                        while (trace->sibling and trace->sibling->score < t->score) {
                            trace = trace->sibling;
                        }
                        t->sibling     = trace->sibling;
                        trace->sibling = t;
                    }
                }
            }
        }
    }

    hadWordEnd_ = (bool)best.get();

    return best;
}

/**
 * Fall back strategy for finding the best sentence hypothesis when
 * there is no active word end hypothesis.  Typically used when
 * getBestSentenceEnd() fails.  This problem can occur when the
 * recording is truncated in the middle of a word and pruning is
 * tight.  The strategy is to select the best state from each active
 * network and consider it as a "word end" without a pronunciation.
 */

Core::Ref<Trace> SearchSpace::getSentenceEndFallBack(TimeframeIndex time, bool shallCreateLattice) {
    PersistentStateTree const& net = network();

    Core::Ref<Trace> best;

    if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::No)
        shallCreateLattice = false;
    else if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::Yes)
        shallCreateLattice = true;

    if (shallCreateLattice) {
        Core::Application::us()->warning() << "Lattice requested, but not creating it";
    }

    Core::Application::us()->log() << "Using sentence-end fallback";

    ///@todo This does not create a lattice for the last word. However a lattice structure is given for the previous words.
    std::vector<StateHypothesis>::const_iterator bestHyp = bestScoreStateHypothesis();
    if (bestHyp == stateHypotheses.end()) {
        Core::Application::us()->warning() << "Found no best state hypotheses, total number of hypotheses: " << stateHypotheses.size();
        return best;
    }

    u32 bestHypIndex = bestHyp - stateHypotheses.begin();

    for (InstanceList::const_iterator t = activeInstances.begin(); t != activeInstances.end(); ++t) {
        const Instance* at(*t);
        TraceId         activeTrace = bestHyp->trace;
        if (bestHypIndex >= at->states.begin && bestHypIndex < at->states.end) {
            Score score = bestHyp->score;

            Core::Ref<Trace> pre(trace_manager_.traceItem(activeTrace).trace);
            best                 = Core::Ref<Trace>(new Trace(pre, 0, time, pre->score, describeRootState(net.rootState)));
            best->score.acoustic = globalScoreOffset_ + score - pre->score.lm;

            Lm::History h = trace_manager_.traceItem(bestHyp->trace).scoreHistory;

            verify(h.isValid());
            for (std::vector<const Bliss::Lemma*>::const_iterator it = recognitionContext_.suffix.begin(); it != recognitionContext_.suffix.end(); ++it)
                Lm::addLemmaScore(lm_, *it, lm_->scale(), h, best->score.lm);

            best->score.lm += lm_->sentenceEndScore(h);
        }
    }

    verify(best);

    return best;
}

class RootTraceSearcher {
public:
    RootTraceSearcher(std::vector<Core::Ref<Trace>> traces)
            : rootTrace_(0) {
        for (std::vector<Core::Ref<Trace>>::const_iterator it = traces.begin(); it != traces.end(); ++it)
            addTrace(it->get(), 0, true);

        for (std::map<Trace*, TraceDesc>::iterator it = traces_.begin(); it != traces_.end(); ++it) {
            if ((*it).second.length == 1) {
                // This is "the" root trace
                verify(rootTrace_ == 0);
                rootTrace_      = it->first;
                TraceDesc& desc = traces_[rootTrace_];
                while (desc.followers.size() == 1) {  // can not be sure if current root trace still have active states
                    if (desc.isEndTrace)
                        break;
                    rootTrace_ = desc.followers.front();
                    desc       = traces_[rootTrace_];
                }
            }
        }
    }

    Trace* rootTrace() const {
        return rootTrace_;
    }

private:
    int addTrace(Trace* trace, Trace* follower, bool isEndTrace = false) {
        std::map<Trace*, TraceDesc>::iterator it = traces_.find(trace);

        if (it != traces_.end()) {
            // Already there, just add follower
            TraceDesc& desc((*it).second);
            if (follower)
                desc.followers.push_back(follower);
            return desc.length;
        }
        else {
            // Add the predecessors, compute the length, and add the new trace + follower
            int length = 1;
            if (trace->predecessor)
                length += addTrace(trace->predecessor.get(), trace);
            TraceDesc desc;
            desc.length     = length;
            desc.isEndTrace = isEndTrace;
            if (follower)
                desc.followers.push_back(follower);
            traces_.insert(std::make_pair(trace, desc));
            return length;
        }
    }

    struct TraceDesc {
        int                 length;
        std::vector<Trace*> followers;
        bool                isEndTrace;
    };

    std::map<Trace*, TraceDesc> traces_;
    Trace*                      rootTrace_;
};

Core::Ref<Trace>
        SearchSpace::getCommonPrefix() const {
    std::set<TraceId> considerTraceIds;
    for (std::vector<StateHypothesis>::const_iterator it = stateHypotheses.begin(); it != stateHypotheses.end(); ++it)
        considerTraceIds.insert(it->trace);

    // Find the trace where all traces merge

    std::vector<Core::Ref<Trace>> traces;
    for (std::set<TraceId>::iterator it = considerTraceIds.begin(); it != considerTraceIds.end(); ++it) {
        Core::Ref<Trace> trace = trace_manager_.traceItem(*it).trace;
        traces.push_back(trace);
    }

    for (WordEndHypothesisList::const_iterator it = wordEndHypotheses.begin(); it != wordEndHypotheses.end(); ++it)
        traces.push_back(it->trace);

    RootTraceSearcher searcher(traces);
    verify(searcher.rootTrace());
    return Core::Ref<Trace>(searcher.rootTrace());
}

class InitialTraceChanger {
public:
    InitialTraceChanger(Core::Ref<Trace> initialTrace)
            : kept(0), killed(0), initialTrace_(initialTrace), baseScore_(initialTrace->score) {
    }

    bool check(const Core::Ref<Trace>& trace) {
        if (!trace)
            return false;

        std::stack<Core::Ref<Trace>> stack;
        stack.push(trace);

        while (stack.size()) {
            Core::Ref<Trace> current = stack.top();

            if (!keepTraces_.count(current.get())) {
                if (current->sibling && !keepTraces_.count(current->sibling.get())) {
                    stack.push(current->sibling);
                    continue;
                }
                else if (current->predecessor && !keepTraces_.count(current->predecessor.get())) {
                    stack.push(current->predecessor);
                    continue;
                }
                else {
                    verify(!current->predecessor || keepTraces_.count(current->predecessor.get()));
                    verify(!current->sibling || keepTraces_.count(current->sibling.get()));

                    current->score.acoustic -= baseScore_.acoustic;
                    current->score.lm -= baseScore_.lm;

                    bool keep                  = (current == initialTrace_) || (current->predecessor && keepTraces_[current->predecessor.get()]);
                    keepTraces_[current.get()] = keep;
                    if (keep) {
                        verify(current->score.acoustic >= -0.01);
                        verify(current->score.lm >= -0.01);
                    }

                    if (current->sibling && !keepTraces_[current->sibling.get()])
                        current->sibling = current->sibling->sibling;

                    verify(!current->sibling || keepTraces_[current->sibling.get()]);

                    if (keep)
                        ++kept;
                    else
                        ++killed;
                }
            }

            stack.pop();
        }

        std::map<Trace*, bool>::const_iterator it = keepTraces_.find(trace.get());
        verify(it != keepTraces_.end());
        return it->second;
    }

    u32 kept, killed;

private:
    std::map<Trace*, bool>       keepTraces_;
    Core::Ref<Trace>             initialTrace_;
    SearchAlgorithm::ScoreVector baseScore_;
};

void SearchSpace::changeInitialTrace(Core::Ref<Trace> trace) {
    verify(trace);
    trace->sibling.reset();
    trace->predecessor.reset();
    trace->pronunciation = 0;

    // Also rescale word-end hypotheses
    for (WordEndHypothesisList::iterator it = wordEndHypotheses.begin(); it != wordEndHypotheses.end(); ++it) {
        it->score.acoustic -= trace->score.acoustic - globalScoreOffset_;
        it->score.lm -= trace->score.lm;
        verify(it->score.acoustic > -0.01);
        verify(it->score.lm > -0.01);
    }

    // Re-scale state hypotheses relative to the new base score
    rescale(trace->score - globalScoreOffset_, true);

    globalScoreOffset_ = 0;

    // Truncate and scale all traces

    InitialTraceChanger changer(trace);

    for (StateHypothesesList::iterator sh = stateHypotheses.begin(); sh != stateHypotheses.end(); ++sh) {
        verify(sh->score > -0.01);
        Core::Ref<Trace> trace = trace_manager_.traceItem(sh->trace).trace;
        bool             ok    = changer.check(trace);
        verify(ok);
    }

    for (WordEndHypothesisList::const_iterator it = wordEndHypotheses.begin(); it != wordEndHypotheses.end(); ++it) {
        bool ok = changer.check(it->trace);
        verify(ok);
    }

    verify(trace->score.acoustic == 0.0);
    verify(trace->score.lm == 0.0);

    std::cout << "changed initial trace, removed " << changer.killed << ", preserved " << changer.kept << " traces" << std::endl;
}

u32 SearchSpace::nStateHypotheses() const {
    return stateHypotheses.size();
}

u32 SearchSpace::nEarlyWordEndHypotheses() const {
    return earlyWordEndHypotheses.size();
}

u32 SearchSpace::nWordEndHypotheses() const {
    return wordEndHypotheses.size();
}

u32 SearchSpace::nActiveTrees() const {
    return activeInstances.size();
}

void SearchSpace::doStateStatisticsBeforePruning() {
    if (!extendStatistics_)
        return;

    const Lm::BackingOffLm* backOffLm = dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get());

    u32              statesInTreesWithLookAhead = 0, statesInTreesWithoutLookAhead = 0;
    std::vector<u32> statesInTreesWithLookAheadHistory;

    for (InstanceList::reverse_iterator it = activeInstances.rbegin(); it != activeInstances.rend(); ++it) {
        if (backOffLm) {
            //Do statistics over the count of states in back-off instances
            Instance& mt = dynamic_cast<Instance&>(**it);

            if (mt.lookahead.get())
                statesInTreesWithLookAhead += mt.states.size();
            else
                statesInTreesWithoutLookAhead += mt.states.size();
        }
    }

    statistics->customStatistics("states before pruning in trees with lookahead") += statesInTreesWithLookAhead;
    statistics->customStatistics("states before pruning in trees without lookahead") += statesInTreesWithoutLookAhead;
}

void SearchSpace::doStateStatistics() {
    if (PathTrace::Enabled) {
        Score best = bestProspect();
        for (std::vector<StateHypothesis>::iterator it = stateHypotheses.begin(); it != stateHypotheses.end(); ++it)
            (*it).pathTrace.maximizeOffset("acoustic-pruning", (*it).prospect - best);
    }

    if (!extendStatistics_)
        return;

    if (!automaton_->stateDepths.empty()) {
        {
            std::vector<u32> perDepth;

            for (StateHypothesisIndex idx = 0; idx < stateHypotheses.size(); ++idx) {
                u32 depth = automaton_->stateDepths[stateHypotheses[idx].state];
                if (depth >= perDepth.size())
                    perDepth.resize(depth + 1, 0);
                ++perDepth[depth];
            }

            for (u32 a = 0; a < perDepth.size(); ++a)
                statesOnDepth_.addValue(a, perDepth[a]);
        }
        {
            std::vector<u32> perDepth;

            for (InstanceList::iterator it = activeInstances.begin(); it != activeInstances.end(); ++it) {
                if (!static_cast<Instance&>(**it).lookahead.get())
                    continue;
                for (StateHypothesisIndex idx = (*it)->states.begin; idx < (*it)->states.end; ++idx) {
                    u32 depth = automaton_->stateDepths[stateHypotheses[idx].state];
                    if (depth >= perDepth.size())
                        perDepth.resize(depth + 1, 0);
                    ++perDepth[depth];
                }
            }
        }
    }

    if (!automaton_->invertedStateDepths.empty()) {
        {
            std::vector<u32> perDepth;

            for (StateHypothesisIndex idx = 0; idx < stateHypotheses.size(); ++idx) {
                u32 depth = automaton_->invertedStateDepths[stateHypotheses[idx].state];
                if (depth >= perDepth.size())
                    perDepth.resize(depth + 1, 0);
                ++perDepth[depth];
            }

            for (u32 a = 0; a < perDepth.size(); ++a)
                statesOnInvertedDepth_.addValue(a, perDepth[a]);
        }
        {
            std::vector<u32> perDepth;

            for (InstanceList::iterator it = activeInstances.begin(); it != activeInstances.end(); ++it) {
                if (!static_cast<Instance&>(**it).lookahead.get())
                    continue;
                for (StateHypothesisIndex idx = (*it)->states.begin; idx < (*it)->states.end; ++idx) {
                    u32 depth = automaton_->invertedStateDepths[stateHypotheses[idx].state];
                    if (depth >= perDepth.size())
                        perDepth.resize(depth + 1, 0);
                    ++perDepth[depth];
                }
            }
        }
    }

    const Lm::BackingOffLm* backOffLm = dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get());

    u32              statesInTreesWithLookAhead = 0, statesInTreesWithoutLookAhead = 0;
    std::vector<u32> statesInTreesWithLookAheadHistory;

    for (InstanceList::reverse_iterator it = activeInstances.rbegin(); it != activeInstances.rend(); ++it) {
        if (backOffLm) {
            //Do statistics over the count of states in back-off instances
            Instance& mt = dynamic_cast<Instance&>(**it);

            Lm::History h = mt.lookaheadHistory;

            int len = 0;

            if (h.isValid())
                len = backOffLm->historyLenght(h);

            if (mt.lookahead.get())
                statesInTreesWithLookAhead += mt.states.size();
            else
                statesInTreesWithoutLookAhead += mt.states.size();

            if (len >= statesInTreesWithLookAheadHistory.size())
                statesInTreesWithLookAheadHistory.resize(len + 1, 0);

            statesInTreesWithLookAheadHistory[len] += mt.states.size();
        }
    }

    for (u32 len = 0; len < statesInTreesWithLookAheadHistory.size(); ++len) {
        std::ostringstream os;
        os << "states in trees with lookahead history length " << len;
        statistics->customStatistics(os.str()) += statesInTreesWithLookAheadHistory[len];
    }

    statistics->customStatistics("states in trees with lookahead") += statesInTreesWithLookAhead;
    statistics->customStatistics("states in trees without lookahead") += statesInTreesWithoutLookAhead;
}

template<bool onTheFlyRescoring>
inline void SearchSpace::recombineTwoHypotheses(WordEndHypothesis& a, WordEndHypothesis& b, bool shallCreateLattice) {
    if (b.score > a.score || (b.score == a.score && b.pronunciation->id() > a.pronunciation->id())) {  // Make the order deterministic if the scores are equal
        // The incoming hypothesis a is better than the stored hypothesis b,
        // just store the values from the incoming a over the old b.

        if (onTheFlyRescoring) {
            auto offset = b.score - a.score;
            for (AlternativeHistory& h : b.trace->alternativeHistories.container()) {
                h.offset += offset;
            }
            while (not b.trace->alternativeHistories.empty()) {
                a.trace->alternativeHistories.push(b.trace->alternativeHistories.top());
                b.trace->alternativeHistories.pop();
            }
            a.trace->alternativeHistories.push(AlternativeHistory{b.scoreHistory, offset, b.trace});
            while (a.trace->alternativeHistories.size() > onTheFlyRescoringMaxHistories_) {
                a.trace->alternativeHistories.pop();
            }
            if (not a.trace->mark) {
                altHistTraces_.push_back(a.trace);
                a.trace->mark = true;
            }
        }

        b.recombinationHistory = a.recombinationHistory;  // just remember the history of the better path (relevant for mesh decoding)
        b.lookaheadHistory     = a.lookaheadHistory;
        b.scoreHistory         = a.scoreHistory;
        b.pronunciation        = a.pronunciation;
        b.endExit              = a.endExit;
        b.score                = a.score;
        if (shallCreateLattice) {
            verify(!a.trace->sibling);
            a.trace->sibling = b.trace;
        }
        b.trace = a.trace;
    }
    else {
        if (shallCreateLattice) {
            verify(!a.trace->sibling);
            if (not onTheFlyRescoring) {
                a.trace->sibling = b.trace->sibling;
                b.trace->sibling = a.trace;
            }
            else {  // for on the fly-rescoring we require sorted siblings
                Core::Ref<Trace> t = b.trace;
                while (t->sibling and t->sibling->score < a.trace->score) {
                    t = t->sibling;
                }
                a.trace->sibling = t->sibling;
                t->sibling       = a.trace;

                b.trace->alternativeHistories.push(AlternativeHistory{a.scoreHistory, a.score - b.score, a.trace});
                if (not b.trace->mark) {
                    altHistTraces_.push_back(b.trace);
                    b.trace->mark = true;
                }
                while (b.trace->alternativeHistories.size() > onTheFlyRescoringMaxHistories_) {
                    b.trace->alternativeHistories.pop();
                }
            }
        }
    }
}

void SearchSpace::recombineWordEnds(bool shallCreateLattice) {
    if (onTheFlyRescoring_) {
        recombineWordEndsInternal<true>(shallCreateLattice);
    }
    else {
        recombineWordEndsInternal<false>(shallCreateLattice);
    }
}

template<bool onTheFlyRescoring>
void SearchSpace::recombineWordEndsInternal(bool shallCreateLattice) {
    PerformanceCounter perf(*statistics, "recombine word-ends");

    if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::No)
        shallCreateLattice = false;
    else if (recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::Yes)
        shallCreateLattice = true;

    WordEndHypothesisList::iterator in, out;

    if (decodeMesh_ && shallCreateLattice) {
        typedef std::unordered_set<WordEndHypothesisList::iterator, WordEndHypothesis::MeshHash, WordEndHypothesis::MeshEquality> MeshWordEndHypothesisRecombinationMap;
        MeshWordEndHypothesisRecombinationMap                                                                                     wordEndHypothesisMap;  // Map used for recombining word end hypotheses

        for (in = out = wordEndHypotheses.begin(); in != wordEndHypotheses.end(); ++in) {
            MeshWordEndHypothesisRecombinationMap::iterator i = wordEndHypothesisMap.find(in);  // equality based on equal transit and a shared pronunciation suffix of meshHistoryPhones
            if (i != wordEndHypothesisMap.end()) {
                WordEndHypothesis& a(*in);
                WordEndHypothesis& b(**i);
                verify_(b.transitState == a.transitState);
                recombineTwoHypotheses<onTheFlyRescoring>(a, b, shallCreateLattice);
            }
            else {
                *out = *in;
                wordEndHypothesisMap.insert(out);
                ++out;
            }
        }
    }
    else if (reducedContextWordRecombination_) {
        ReducedContextRecombinationMap wordEndHypothesisMap;

        for (in = out = wordEndHypotheses.begin(); in != wordEndHypotheses.end(); ++in) {
            auto                                     key = std::make_pair(recombinationLm_->reducedHistory(in->recombinationHistory,
                                                                       reducedContextWordRecombinationLimit_),
                                      in->transitState);
            ReducedContextRecombinationMap::iterator i   = wordEndHypothesisMap.find(key);
            if (i != wordEndHypothesisMap.end()) {
                WordEndHypothesis& a(*in);
                WordEndHypothesis& b(*(i->second));
                verify_(b.transitState == a.transitState);
                recombineTwoHypotheses<onTheFlyRescoring>(a, b, shallCreateLattice);
            }
            else {
                *out = *in;
                wordEndHypothesisMap.insert(std::make_pair(key, out));
                ++out;
            }
        }
    }
    else {
        wordEndHypothesisMap.clear();

        for (in = out = wordEndHypotheses.begin(); in != wordEndHypotheses.end(); ++in) {
            WordEndHypothesisRecombinationMap::iterator i = wordEndHypothesisMap.find(in);
            if (i != wordEndHypothesisMap.end()) {
                WordEndHypothesis& a(*in);
                WordEndHypothesis& b(**i);
                verify_(b.recombinationHistory == a.recombinationHistory);  // found another hypothesis with equal transit and history
                verify_(b.transitState == a.transitState);
                recombineTwoHypotheses<onTheFlyRescoring>(a, b, shallCreateLattice);
            }
            else {
                *out = *in;
                wordEndHypothesisMap.insert(out);
                ++out;
            }
        }
    }
    wordEndHypotheses.erase(out, wordEndHypotheses.end());

    doWordEndStatistics();
}

void SearchSpace::doWordEndStatistics() {
    PersistentStateTree const& net = network();

    if (lmLookahead_)
        lmLookahead_->collectStatistics();

    {
        std::unordered_map<Bliss::Lemma::Id, bool> wordEndLemmas;

        for (std::vector<WordEndHypothesis>::iterator it = wordEndHypotheses.begin(); it != wordEndHypotheses.end(); ++it)
            if (it->pronunciation && it->pronunciation->lemma() && it->pronunciation->lemma()->syntacticTokenSequence().size())
                wordEndLemmas.insert(std::make_pair(it->pronunciation->lemma()->id(), true));
        currentWordLemmasAfterRecombination += wordEndLemmas.size();
        statistics->customStatistics("word lemmas after recombination") += wordEndLemmas.size();
    }

    if (PathTrace::Enabled)
        for (std::vector<WordEndHypothesis>::iterator it = wordEndHypotheses.begin(); it != wordEndHypotheses.end(); ++it)
            (*it).trace->pathTrace.maximizeOffset("word-end-pruning", (*it).score - minWordEndScore_);

    if (!extendStatistics_)
        return;

    typedef std::unordered_map<Lm::History, Score, Lm::History::Hash> BestHash;
    BestHash                                                          bestWordEnds;

    u32 coarticulatedWordEnds = 0, rootWordEnds = 0, ciWordEnds = 0, specialWordEnds = 0;

    for (WordEndHypothesisList::const_iterator weh = wordEndHypotheses.begin();
         weh != wordEndHypotheses.end(); ++weh) {
        if (weh->pronunciation->lemma() == 0 || !weh->pronunciation->lemma()->hasSyntacticTokenSequence())
            ++specialWordEnds;
        if (weh->transitState == net.rootState)
            ++rootWordEnds;
        else if (weh->transitState == net.ciRootState)
            ++ciWordEnds;
        else
            ++coarticulatedWordEnds;
    }

    statistics->customStatistics("coarticulated word ends") += coarticulatedWordEnds;
    statistics->customStatistics("root word-ends") += rootWordEnds;
    statistics->customStatistics("ci word ends") += ciWordEnds;
    statistics->customStatistics("special word ends") += specialWordEnds;

    if (activeInstances.size() > 1) {
        f32 dominance         = 0;
        int maxTreeStateCount = 0;

        for (std::vector<Instance*>::const_iterator it = activeInstances.begin(); it != activeInstances.end(); ++it)
            if ((*it)->states.size() > maxTreeStateCount)
                maxTreeStateCount = (*it)->states.size();

        if (maxTreeStateCount) {
            for (std::vector<Instance*>::const_iterator it = activeInstances.begin(); it != activeInstances.end(); ++it)
                dominance += (*it)->states.size();

            dominance = maxTreeStateCount / dominance;
        }

        statistics->customStatistics("network dominance") += dominance;
    }
}

void SearchSpace::setCurrentTimeFrame(TimeframeIndex timeFrame, const Mm::FeatureScorer::Scorer& scorer) {
    timeFrame_ = timeFrame;
    scorer_    = scorer;

    if (currentPruning_.get() && currentPruning_->haveTimeDependentPruning())
        setMasterBeam(currentPruning_->beamForTime(timeFrame) * lm_->scale());

    PerformanceCounter perf(*statistics, "initialize acoustic lookahead");

    acousticLookAhead_->startLookAhead(timeFrame_, true);

    if (ssaLm_) {
        ssaLm_->startFrame(timeFrame);
    }
}

Instance* SearchSpace::createTreeInstance(const Search::InstanceKey& key) {
    Instance* ret = new Instance(key, 0);
    return ret;
}

Instance* SearchSpace::instanceForKey(bool create, const InstanceKey& key, Lm::History const& lookaheadHistory, Lm::History const& scoreHistory) {
    std::unordered_map<InstanceKey, Instance*, InstanceKey::Hash>::iterator it = activeInstanceMap.find(key);
    if (it != activeInstanceMap.end())
        return it->second;

    if (!create)
        return 0;

    Instance* t         = createTreeInstance(key);
    t->lookaheadHistory = lookaheadHistory;
    t->scoreHistory     = scoreHistory;
    activeInstances.push_back(t);
    verify(activeInstanceMap.find(key) == activeInstanceMap.end());
    activeInstanceMap[key] = t;

    return t;
}

void SearchSpace::cleanup() {
    // Cleanup the traces
    PerformanceCounter perf(*statistics, "cleanup");

    TraceManager::Cleaner cleaner = trace_manager_.getCleaner();
    for (std::vector<StateHypothesis>::const_iterator it = stateHypotheses.begin(); it != stateHypotheses.end(); ++it) {
        cleaner.visit((*it).trace);
    }

    for (std::vector<Instance*>::iterator instIt = activeInstances.begin(); instIt != activeInstances.end(); ++instIt) {
        for (std::vector<StateHypothesis>::const_iterator it = (*instIt)->rootStateHypotheses.begin(); it != (*instIt)->rootStateHypotheses.end(); ++it) {
            cleaner.visit(it->trace);
        }
    }

    cleaner.clean();
}

int SearchSpace::lookAheadLength() const {
    return acousticLookAhead_->length();
}

Search::SearchAlgorithm::RecognitionContext SearchSpace::setContext(Search::SearchAlgorithm::RecognitionContext context) {
    Search::SearchAlgorithm::RecognitionContext ret = recognitionContext_;
    recognitionContext_                             = context;
    return ret;
}

void SearchSpace::setLookAhead(const std::vector<Mm::FeatureVector>& lookahead) {
    acousticLookAhead_->setLookAhead(lookahead);
}

void SearchSpace::logStatistics(Core::XmlChannel& channel) const {
    statistics->write(channel);

    if (lmLookahead_)
        lmLookahead_->logStatistics();

    if (extendStatistics_) {
        channel << "states on hmm-depth: " << statesOnDepth_.print();
        channel << "states on inverted hmm-depth: " << statesOnInvertedDepth_.print();
    }
}

void SearchSpace::resetStatistics() {
    statistics->clear();
}

void SearchSpace::extendHistoryByLemma(WordEndHypothesis& weh, const Bliss::Lemma* lemma) const {
    const Bliss::SyntacticTokenSequence tokenSequence(lemma->syntacticTokenSequence());
    for (u32 ti = 0; ti < tokenSequence.length(); ++ti) {
        const Bliss::SyntacticToken* st = tokenSequence[ti];
        weh.recombinationHistory        = recombinationLm_->extendedHistory(weh.recombinationHistory, st);
        weh.lookaheadHistory            = lookaheadLm_->extendedHistory(weh.lookaheadHistory, st);
        weh.scoreHistory                = lm_->extendedHistory(weh.scoreHistory, st);
    }
}

bool SearchSpace::relaxPruning(f32 factor, f32 offset) {
    if (histogramPruningIsMasterPruning_) {
        if (acousticPruningLimit_ * factor + offset <= minimumAcousticPruningLimit_) {
            std::cout << "FAILED tightening pruning, minimum beam pruning limit of " << acousticPruningLimit_ << std::endl;
            return false;
        }
        if (acousticPruningLimit_ >= maximumAcousticPruningLimit_) {
            std::cout << "FAILED relaxing pruning, maximum beam pruning limit of " << acousticPruningLimit_ << std::endl;
            return false;
        }
        u32 newLimit = acousticPruningLimit_ * factor + offset;
        if (newLimit > maximumAcousticPruningLimit_)
            newLimit = maximumAcousticPruningLimit_;
        setMasterBeam(newLimit * lm_->scale());
        return true;
    }
    if (beamPruning() >= maximumBeamPruning_) {
        std::cout << "FAILED relaxing pruning, maximum beam pruning is already hit: " << beamPruning() << " >= " << maximumBeamPruning_ << std::endl;
        return false;
    }

    if (beamPruning() < Core::Type<f32>::max && (factor < 1.0 || offset < 0) && beamPruning() * factor + offset < minimumBeamPruning_) {
        std::cout << "FAILED tightening pruning, minimum beam pruning is already hit: "
                  << (beamPruning() * factor + offset) << " < " << minimumBeamPruning_ << std::endl;
        return false;
    }

    if ((factor > 1.0 || offset > 0)) {
        if (currentStatesAfterPruning.average() > maximumStatesAfterPruning_) {
            std::cout << "FAILED relaxing pruning, maximum states-after-pruning already hit: "
                      << currentStatesAfterPruning.average() << " > " << maximumStatesAfterPruning_ << std::endl;
            return false;
        }

        if (currentWordEndsAfterPruning.average() > maximumWordEndsAfterPruning_) {
            std::cout << "FAILED relaxing pruning, maximum word-ends-after-pruning already hit: "
                      << currentWordEndsAfterPruning.average() << " > " << maximumWordEndsAfterPruning_ << std::endl;
            return false;
        }

        if (currentAcousticPruningSaturation.average() > maximumAcousticPruningSaturation_) {
            std::cout << "FAILED relaxing pruning, maximum acoustic-pruning-saturation already hit: "
                      << currentAcousticPruningSaturation.average() << " > " << maximumAcousticPruningSaturation_ << std::endl;
            return false;
        }
    }

    setMasterBeam(acousticPruning_ * factor + offset * lm_->scale());

    return true;
}

void SearchSpace::setMasterBeam(Score value) {
    if (histogramPruningIsMasterPruning_) {
        f32 oldAcousticPruningLimit = acousticPruningLimit_;

        acousticPruningLimit_ = value / lm_->scale();

        if (oldAcousticPruningLimit != acousticPruningLimit_) {
            std::cout << "t=" << timeFrame_ << " hp -> " << acousticPruningLimit_ << std::endl;
            if (wordEndPruningLimit_ < oldAcousticPruningLimit)
                wordEndPruningLimit_ = wordEndPruningLimit_ * (acousticPruningLimit_ / oldAcousticPruningLimit);
        }
    }
    else {
        f32 oldAcousticPruning       = acousticPruning_;
        f32 oldWordEndPruning        = wordEndPruning_;
        f32 oldLmStatePruning        = lmStatePruning_;
        f32 oldWordEndPhonemePruning = wordEndPhonemePruningThreshold_;

        verify(acousticPruning_ < Core::Type<f32>::max);

        acousticPruning_ = value;
        /*if (oldAcousticPruning != acousticPruning_)
            std::cout << "t=" << timeFrame_ << ": bp -> " << acousticPruning_ / lm_->scale() << " (previous " << oldAcousticPruning / lm_->scale() << ")" << std::endl;*/
        verify(acousticPruning_ != 0);

        if (wordEndPruning_ < Core::Type<f32>::max)
            wordEndPruning_ = oldWordEndPruning * (acousticPruning_ / oldAcousticPruning);

        if (lmStatePruning_ < Core::Type<f32>::max)
            lmStatePruning_ = oldLmStatePruning * (acousticPruning_ / oldAcousticPruning);

        if (wordEndPhonemePruningThreshold_ < Core::Type<f32>::max)
            wordEndPhonemePruningThreshold_ = oldWordEndPhonemePruning * (acousticPruning_ / oldAcousticPruning);
    }
}

SearchAlgorithm::PruningRef SearchSpace::describePruning() {
    PruningDesc* oldPruning = new PruningDesc;
    if (histogramPruningIsMasterPruning_) {
        oldPruning->beam = acousticPruningLimit_;
    }
    else {
        oldPruning->beam = acousticPruning_ / lm_->scale();
    }

    if (currentStatesAfterPruning.nObservations()) {
        if (!hadWordEnd_) {
            oldPruning->searchSpaceOK = false;
            log() << "had no word-end";
        }

        if (!histogramPruningIsMasterPruning_) {
            if (currentStatesAfterPruning.average() < minimumStatesAfterPruning_) {
                oldPruning->searchSpaceOK = false;
                log() << "too few average states: " << currentStatesAfterPruning.average() << " < " << minimumStatesAfterPruning_;
            }
            if (currentWordEndsAfterPruning.average() < minimumWordEndsAfterPruning_) {
                oldPruning->searchSpaceOK = false;
                log() << "too few average word-ends: " << currentWordEndsAfterPruning.average() << " < " << minimumWordEndsAfterPruning_;
            }
            if (currentWordLemmasAfterRecombination.average() < minimumWordLemmasAfterRecombination_) {
                oldPruning->searchSpaceOK = false;
                log() << "too few word lemmas after recombination: " << currentWordLemmasAfterRecombination.average() << " < " << minimumWordLemmasAfterRecombination_;
            }
        }
    }

    return SearchAlgorithm::PruningRef(oldPruning);
}

void SearchSpace::resetPruning(SearchAlgorithm::PruningRef pruning) {
    PruningDesc* newPruning(dynamic_cast<PruningDesc*>(pruning.get()));
    verify(newPruning->beam != Core::Type<Score>::max);
    setMasterBeam(newPruning->beam * lm_->scale());
    currentPruning_ = Core::Ref<PruningDesc>(newPruning);
    if (!currentPruning_->haveTimeDependentPruning())
        currentPruning_.reset();  // No reason to keep it around
}

void SearchSpace::startNewTrees() {
    std::set<Instance*> allEnteredTrees;

    PerformanceCounter perf(*statistics, "start new trees");

    for (WordEndHypothesisList::const_iterator weh = wordEndHypotheses.begin(); weh != wordEndHypotheses.end(); ++weh) {
        Instance* instance = activateOrUpdateTree(weh->trace, weh->recombinationHistory, weh->lookaheadHistory, weh->scoreHistory, weh->transitState, weh->score);
        verify(instance);
        allEnteredTrees.insert(instance);
        if (lmLookahead_)
            instance->lookaheadHistory = lmLookahead_->getReducedHistory(weh->lookaheadHistory);
    }

    wordEndHypotheses.reserve(wordEndHypotheses.size());
    wordEndHypotheses.clear();
}

Instance* SearchSpace::activateOrUpdateTree(const Core::Ref<Trace>& trace,
                                            Lm::History             recombinationHistory,
                                            Lm::History             lookaheadHistory,
                                            Lm::History             scoreHistory,
                                            StateId                 entry,
                                            Score                   score) {
    /// TODO (Nolden): getLastSyntacticToken is inefficient for long sequences. A simple rule would be better: Stay in same instance, or follow most recent pron.
    InstanceKey key(recombinationHistory, conditionPredecessorWord_ ? getLastSyntacticToken(trace) : Bliss::LemmaPronunciation::invalidId);
    Instance* instance = instanceForKey(true, key, lookaheadHistory, scoreHistory);

    if (!instance)
        return 0;
    verify(dynamic_cast<Instance*>(instance));

    Instance* at = static_cast<Instance*>(instance);

    // still keep the full history in the trace
    at->enter(trace_manager_, trace, entry, score);

    return at;
}

template<bool earlyWordEndPruning, bool onTheFlyRescoring>
void SearchSpace::processOneWordEnd(Instance const& at, StateHypothesis const& hyp, s32 exit, Score exitPenalty, Score relativePruning, Score& bestWordEndPruning) {
    PersistentStateTree::Exit const* we   = &network().exits[exit];
    TraceItem const&                 item = trace_manager_.traceItem(hyp.trace);

    //We can do a more efficient word end handling if there is only one item in the trace, which is the standard case
    verify_(item.scoreHistory.isValid());

    EarlyWordEndHypothesis weh(hyp.trace,
                               SearchAlgorithm::ScoreVector(hyp.score - item.trace->score.lm - at.totalBackOffOffset, item.trace->score.lm),
                               exit,
                               hyp.pathTrace);
    weh.score.acoustic += exitPenalty;
    auto oldScore = weh.score;
    at.addLmScore(weh, we->pronunciation, lm_, lexicon_, wpScale_);

    if (weh.score < minWordEndScore_) {
        minWordEndScore_ = weh.score;
        if (earlyWordEndPruning) {
            bestWordEndPruning = weh.score + relativePruning;
        }
    }

    if (not earlyWordEndPruning or weh.score <= bestWordEndPruning) {
        earlyWordEndHypotheses.push_back(weh);
    }

    if (onTheFlyRescoring) {
        Core::Ref<Trace> trace = item.trace;
        for (AlternativeHistory const& h : trace->alternativeHistories.container()) {
            SearchAlgorithm::ScoreVector newScore = oldScore + h.offset;
            if (we->pronunciation != Bliss::LemmaPronunciation::invalidId) {
                Lm::addLemmaPronunciationScoreOmitExtension(lm_, lexicon_->lemmaPronunciation(we->pronunciation), wpScale_, lm_->scale(), h.hist, newScore.lm);
            }

            if (newScore < minWordEndScore_) {
                minWordEndScore_ = newScore;
                if (earlyWordEndPruning) {
                    bestWordEndPruning = newScore + relativePruning;
                }
            }

            if (not earlyWordEndPruning or newScore <= bestWordEndPruning) {
                TraceItem const& item    = trace_manager_.traceItem(hyp.trace);  // item might be relocated by call to getTrace
                TraceId          traceId = trace_manager_.getTrace(TraceItem(h.trace, item.recombinationHistory, item.lookaheadHistory, h.hist));
                if (trace_manager_.isModified(hyp.trace)) {
                    TraceManager::Modification mod = trace_manager_.getModification(hyp.trace);
                    traceId                        = trace_manager_.modify(trace_manager_.getUnmodified(traceId), mod.first, mod.second, mod.third);
                }
                earlyWordEndHypotheses.emplace_back(traceId, newScore, exit, hyp.pathTrace);
            }
        }
    }
}

template<bool earlyWordEndPruning, bool onTheFlyRescoring>
void SearchSpace::findWordEndsInternal() {
    PerformanceCounter perf(*statistics, "find word ends");

    PersistentStateTree const& net               = network();
    std::vector<int> const&    singleLabels      = automaton_->singleLabels;
    std::vector<u32> const&    quickLabelBatches = automaton_->quickLabelBatches;
    std::vector<s32> const&    slowLabelBatches  = automaton_->slowLabelBatches;

    Score relativePruning    = std::min(acousticPruning_, wordEndPruning_);
    Score bestWordEndPruning = Core::Type<Score>::max;
    minWordEndScore_         = Core::Type<Score>::max;

    verify(earlyWordEndHypotheses.empty());

    for (Instance* inst : activeInstances) {
        for (StateHypothesesList::iterator sh = stateHypotheses.begin() + inst->states.begin; sh != stateHypotheses.begin() + inst->states.end; ++sh) {
            StateHypothesis& hyp(*sh);

            s32 exit = singleLabels[hyp.state];
            if (exit == -1) {
                continue;  // No labels
            }

            const HMMState& state       = net.structure.state(hyp.state);
            Score           exitPenalty = (*transitionModel(state.stateDesc))[Am::StateTransitionModel::exit];

            if (earlyWordEndPruning && hyp.score + exitPenalty + earlyWordEndPruningAnticipatedLmScore_ > bestWordEndPruning) {
                continue;  //Apply early word-end pruning (If the best score can not be reached, do not even try)
            }

            ///With pushing, ca. 80% of all label-lists are single-labels, so optimize for this case
            if (exit >= 0) {
                // There is 1 label
                processOneWordEnd<earlyWordEndPruning, onTheFlyRescoring>(*inst, hyp, exit, exitPenalty, relativePruning, bestWordEndPruning);
            }
            else if (exit == -2) {
                // There are multiple labels, with a nice regular structure, use quickLabelBatches_
                u32 exitsStart = quickLabelBatches[hyp.state];
                u32 exitsEnd   = quickLabelBatches[hyp.state + 1];

                for (exit = exitsStart; exit != exitsEnd; ++exit) {
                    processOneWordEnd<earlyWordEndPruning, onTheFlyRescoring>(*inst, hyp, exit, exitPenalty, relativePruning, bestWordEndPruning);
                }
            }
            else {
                // There are multiple labels, however we cannot use quickLabelBatches_.
                for (s32 current = -(exit + 3); slowLabelBatches[current] != -1; ++current) {
                    u32 exit = slowLabelBatches[current];
                    processOneWordEnd<earlyWordEndPruning, onTheFlyRescoring>(*inst, hyp, exit, exitPenalty, relativePruning, bestWordEndPruning);
                }
            }
        }
    }
}

void SearchSpace::findWordEnds() {
    //std::cerr << "best hyp: " << bestScore() << std::endl;
    if (earlyWordEndPruning_) {
        if (onTheFlyRescoring_) {
            findWordEndsInternal<true, true>();
        }
        else {
            findWordEndsInternal<true, false>();
        }
    }
    else {
        if (onTheFlyRescoring_) {
            findWordEndsInternal<false, true>();
        }
        else {
            findWordEndsInternal<false, false>();
        }
    }
}

Instance* SearchSpace::getBackOffInstance(Instance* instance) {
    if (instance->backOffInstance || !lmLookahead_)
        return instance->backOffInstance;

    const Lm::BackingOffLm* lm = dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get());
    verify(lm);

    Lm::History useHistory = instance->lookaheadHistory;

    int length = lm->historyLenght(useHistory);

    if (length == 0)
        return 0;

    // Create a back-off network for history-length length-1
    Lm::History reduced = lm->reducedHistory(useHistory, length - 1);

    verify(lm->historyLenght(reduced) == length - 1);

    verify(reduced.isValid());

    activeInstances.push_back(new Instance(instance->key, instance));
    verify(instance->backOffInstance == activeInstances.back());  // beck: how can this ever be true?
    /// @todo Möglicherweise wird der falsche backoff-score angewandt. Es sollte der score für die verkürzte history benutzt werden.
    instance->backOffScore = lm->getBackOffScores(useHistory).backOffScore;

    instance->backOffInstance->scoreHistory     = instance->scoreHistory;
    instance->backOffInstance->lookaheadHistory = reduced;

    verify(instance->backOffInstance);
    verify(instance->backOffInstance->backOffParent == instance);

    return instance->backOffInstance;
}

}  // namespace Search
