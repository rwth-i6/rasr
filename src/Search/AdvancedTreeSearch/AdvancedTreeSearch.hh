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
// $Id: ConditionedTreeSearch.hh 7047 2009-03-02 12:38:56Z nolden $

#ifndef _SEARCH_CONDITIONED_TREE_SEARCH_HH
#define _SEARCH_CONDITIONED_TREE_SEARCH_HH

#include <chrono>

#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Search/Histogram.hh>
#include <Search/LatticeAdaptor.hh>
#include <Search/Search.hh>
#include <Speech/ModelCombination.hh>
#include "DynamicBeamPruningStrategy.hh"
#include "Trace.hh"

namespace Speech {
class StateTying;
}

namespace Search {
struct SearchSpaceStatistics;
class SearchSpace;

class LanguageModelLookahead;
class StateTree;

class AdvancedTreeSearchManager : public SearchAlgorithm {
    class ReverseWordLattice;
    friend class ReverseWordLattice;

public:
    enum LatticeOptimizationMethod {
        noLatticeOptimization,
        simpleSilenceLatticeOptimization
    };
    enum SearchVariant {
        WordConditioned,
        TimeConditioned
    };

private:
    Bliss::LexiconRef                        lexicon_;
    const Bliss::Lemma*                      silence_;
    Core::Ref<const Am::AcousticModel>       acousticModel_;
    Core::Ref<const Lm::ScaledLanguageModel> lm_;

    Score wpScale_;
    bool  shallCreateLattice_;
    bool  allowSentenceEndFallBack_;

    LatticeOptimizationMethod shallOptimizeLattice_;
    u32                       startTreesInterval_, cleanupInterval_;
    f32                       onlineSegmentationLength_, onlineSegmentationMargin_, onlineSegmentationTolerance_;
    bool                      onlineSegmentationIncludeGap_;
    TimeframeIndex            time_;
    TimeframeIndex            currentSegmentStart_;

    f32                                                frameShift_;
    std::chrono::time_point<std::chrono::steady_clock> segmentStartTime_;
    std::unique_ptr<DynamicBeamPruningStrategy>        dynamicBeamPruningStrategy_;

    SearchSpace* ss_;

    mutable Core::XmlChannel statisticsChannel_;

private:
    mutable Core::Ref<Trace> sentenceEnd_;

    Core::Ref<Trace> lastPartialTrace_;

    Core::Ref<Trace> sentenceEnd() const;

    void mergeEpsilonTraces(Core::Ref<Trace> trace) const;

    bool shouldComputeWordEnds();

    Core::Ref<Trace> getCorrectedCommonPrefix();

    Core::Ref<const LatticeAdaptor> buildLatticeForTrace(Core::Ref<Trace> trace) const;

    void traceback(Core::Ref<Trace>, Traceback& result, Core::Ref<Trace> boundary = Core::Ref<Trace>()) const;

public:
    AdvancedTreeSearchManager(const Core::Configuration&);
    virtual ~AdvancedTreeSearchManager();
    AdvancedTreeSearchManager(const AdvancedTreeSearchManager& rhs);

    virtual void                            setAllowHmmSkips(bool allow);
    virtual bool                            setModelCombination(const Speech::ModelCombination& modelCombination);
    virtual void                            setGrammar(Fsa::ConstAutomatonRef);
    virtual void                            restart();
    virtual void                            setSegment(Bliss::SpeechSegment const* segment);
    virtual void                            feed(const Mm::FeatureScorer::Scorer&);
    virtual void                            getPartialSentence(Traceback& result);
    virtual void                            getCurrentBestSentencePartial(Traceback& result) const;
    virtual void                            getCurrentBestSentence(Traceback& result) const;
    virtual Core::Ref<const LatticeAdaptor> getCurrentWordLattice() const;
    virtual Core::Ref<const LatticeAdaptor> getPartialWordLattice();
    virtual void                            resetStatistics();
    virtual void                            logStatistics() const;

    virtual bool       relaxPruning(f32 factor, f32 offset);
    virtual void       resetPruning(PruningRef pruning);
    virtual PruningRef describePruning();

    virtual u32                lookAheadLength();
    virtual void               setLookAhead(const std::vector<Mm::FeatureVector>& lookahead);
    virtual RecognitionContext setContext(RecognitionContext context);
};
}  // namespace Search

#endif
