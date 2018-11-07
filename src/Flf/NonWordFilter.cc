/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Core/Vector.hh>
#include <Fsa/hSort.hh>
#include <Fsa/Hash.hh>

#include "FlfCore/Basic.hh"
#include "FlfCore/LatticeInternal.hh"
#include "FlfCore/TopologicalOrderQueue.hh"
#include "FlfCore/Utility.hh"
#include "Cache.hh"
#include "Copy.hh"
#include "EpsilonRemoval.hh"
#include "Lexicon.hh"
#include "Map.hh"
#include "NonWordFilter.hh"


namespace Flf {

    // -------------------------------------------------------------------------
    struct Hyp {
        bool visited;
        Score score;
        Fsa::StateId bptr;
        State::const_iterator a;
        Hyp() : visited(false), score(Semiring::Max), bptr(Fsa::InvalidStateId) {}
    };
    typedef Core::Vector<Hyp> HypList;
    typedef Core::Vector<Fsa::StateId> StateIdList;
    // -------------------------------------------------------------------------

    // No guarantee that each path is covered only once. However, only arcs are kept which are on the best path for at least one word sequence. Thus, the lattice is the smallest possible sublattice which still covers all word sequences.
    //
    // Pruning:
    // * When the overall effort becomes too high, or a maximum depth is reached, just cut off and select the currently best path (normalized by timeframes)

    struct ArcTraceback {
    public:
        typedef s32 Index;
        enum {
            InvalidIndex = -1
        };
        struct BackPointer {
            BackPointer(Index _pre = InvalidIndex, Fsa::StateId _state = Fsa::InvalidStateId, u32 _arc = Core::Type<u32>::max, Fsa::LabelId _label = Fsa::InvalidLabelId, Score _score = Core::Type<Score>::max) :
                pre(_pre),
                state(_state),
                arc(_arc),
                label(_label),
                score(_score) {}
            Index pre;     // Best predecessor backpos32er for this arc
            Fsa::StateId state;     // Origin state of this arc
            u32 arc;     // Index of this arc relative to the origin state
            Fsa::LabelId label;     // Label of the arc
            Score score;     // Accumulated score (subtract predecessor score to make it relative)
        };

        Index append(Index predecessor, Fsa::StateId state, u32 arc, Fsa::LabelId label, Score score) {
            BackPointer bp(predecessor, state, arc, label, score);
            backPointers_.push_back(bp);
            return backPointers_.size() - 1;
        }

        const BackPointer& get(u32 index) const {
            return backPointers_[index];
        }

        // Returns the longest prefix backtrace which is common to both backtraces
        // InvalidIndex means no common prefix
        Index intersect(Index i1, Index i2) const {
            while (true)
            {
                if (i1 == i2)
                    return i1;
                if (i1 == InvalidIndex || i2 == InvalidIndex)
                    return InvalidIndex;
                if (i1 > i2)
                    i1 = backPointers_[i1].pre;
                else
                    i2 = backPointers_[i2].pre;
            }
        }

        u32 length(Index i) const {
            u32 ret = 0;
            while(i != InvalidIndex)
            {
                ++ret;
                i = backPointers_[i].pre;
            }
            return ret;
        }

        void clear() {
            backPointers_.clear();
        }

    private:
        std::vector<BackPointer> backPointers_;
    };

    struct WordTraceback {
    public:
        typedef s32 Index;
        enum {
            InvalidIndex = -1
        };

        struct BackPointer {
            BackPointer(Index _pre = InvalidIndex, Fsa::LabelId _label = Fsa::InvalidLabelId) :
                pre(_pre),
                label(_label) {}

            Index pre;     // Best predecessor backpos32er for this arc
            Fsa::LabelId label;     // Label of the arc
        };

        Index append(Index predecessor, Fsa::LabelId label) {
            BackPointer bp(predecessor, label);
            backPointers_.push_back(bp);
            return backPointers_.size() - 1;
        }

        const BackPointer& get(u32 index) const {
            return backPointers_[index];
        }

        void clear() {
            backPointers_.clear();
        }

        Fsa::LabelId label(u32 index, u32 offset = 0) const {
            while(offset) {
                index = backPointers_[index].pre;
                --offset;
            }
            return backPointers_[index].label;
        }

        u32 length(Index i) const {
            u32 ret = 0;
            while(i != InvalidIndex)
            {
                ++ret;
                i = backPointers_[i].pre;
            }
            return ret;
        }

    private:
        std::vector<BackPointer> backPointers_;
    };

    struct LatticeClosure {
        LatticeClosure(ConstLatticeRef l, const std::set<Fsa::LabelId>& nonWordLabels, int maxClosureLength) :
            l_(l),
            nonWordLabels_(nonWordLabels),
            maxLabel_(0),
            maxClosureLength_(maxClosureLength) {
            verify(l->initialStateId() == 0); // must be topological
            queue_.insert(l->initialStateId());
            while (!queue_.empty()) {
                Fsa::StateId stateId = *queue_.begin();
                queue_.erase(queue_.begin());
                process(stateId);
            }
        }

        void process(Fsa::StateId stateId) {
            if (stateId >= closureForState_.size())
                closureForState_.resize(stateId + 1, std::make_pair(-1, -1));
            if (closureForState_[stateId].first != -1)
                return;

            Speech::TimeframeIndex startTime = l_->boundary(stateId).time();

            {
                ArcTraceback traceback;

                const ScoreList &scales = l_->semiring()->scales();

                std::map<Fsa::StateId, s32> closureHypotheses;
                closureHypotheses[stateId] = ArcTraceback::InvalidIndex;

                std::map<Fsa::LabelId, std::set<ArcTraceback::Index> > words;

                std::set<Fsa::StateId> had;

                while (!closureHypotheses.empty())
                {
                    Fsa::StateId currentStateId = closureHypotheses.begin()->first;
                    ArcTraceback::Index preBp = closureHypotheses.begin()->second;
                    closureHypotheses.erase(closureHypotheses.begin());
                    ConstStateRef currentState = l_->getState(currentStateId);

                    verify(had.count(currentStateId) == 0);
                    had.insert(currentStateId);

                    if (currentState->isFinal()) {
                        words[Fsa::Epsilon].insert(preBp);
                        continue;
                    }

                    for (u32 a = 0; a < currentState->nArcs(); ++a)
                    {
                        const Arc* arc = currentState->getArc(a);
                        Score score = arc->weight()->project(scales);
                        if (preBp != ArcTraceback::InvalidIndex)
                            score += traceback.get(preBp).score;

                        if (arc->input() > maxLabel_)
                            maxLabel_ = arc->input();

                        bool shortenClosure = false;
                        if (arc->input() != Fsa::Epsilon) {
                            shortenClosure = shortenClosureStates_.count(arc->target());
                            if (!shortenClosure && l_->boundary(arc->target()).time() - startTime > maxClosureLength_ && preBp != ArcTraceback::InvalidIndex) {
                                shortenClosure = true;
                                shortenClosureStates_.insert(arc->target());
                            }
                        }

                        if (nonWordLabels_.count(arc->input()) && !shortenClosure)
                        {
                            ArcTraceback::Index newBp = traceback.append(preBp, currentStateId, a, Fsa::Epsilon, score);

                            std::map<Fsa::StateId, s32>::iterator newClosureHyp = closureHypotheses.find(arc->target());
                            if (newClosureHyp != closureHypotheses.end())
                            {
                                Score otherScore = traceback.get(newClosureHyp->second).score;
                                if (score < otherScore)
                                    newClosureHyp->second = newBp;
                            }else{
                                closureHypotheses.insert(std::make_pair(arc->target(), newBp));
                            }
                        }else{
                            words[arc->input()].insert(traceback.append(preBp, currentStateId, a, arc->input(), score));
                            queue_.insert(arc->target());
                        }
                    }
                }

                s32 closureBegin = closures_.size();

                for (std::map<Fsa::LabelId, std::set<ArcTraceback::Index> >::iterator it = words.begin(); it != words.end(); ++it)
                {
                    for (std::set<ArcTraceback::Index>::iterator bpIt = it->second.begin(); bpIt != it->second.end(); ++bpIt)
                    {
                        Closure closure;
                        closure.word = it->first;
                        closure.score = traceback.get(*bpIt).score;
                        closure.arcs = arcs_.size();
                        closure.target = l_->getState(traceback.get(*bpIt).state)->getArc(traceback.get(*bpIt).arc)->target();

                        closures_.push_back(closure);

                        s32 bp = *bpIt;
                        while (bp != -1)
                        {
                            arcs_.push_back(traceback.get(bp).arc);
                            bp = traceback.get(bp).pre;
                        }

                        std::reverse(arcs_.begin() + closure.arcs, arcs_.end());
                    }
                }

                closureForState_[stateId] = std::make_pair(closureBegin, (s32)closures_.size());
            }
        }

        struct Closure {
            Fsa::LabelId word;     // Epsilon for a path leading through epsilons to the final state
            Fsa::StateId target;
            Score score;
            s32 arcs;

            struct WordCompare {
                bool operator()(const Closure& lhs, const Closure& rhs) const {
                    return lhs.word < rhs.word;
                }
            };
        };

        std::pair<s32, s32> getClosures(Fsa::StateId state) const {
            if (state < closureForState_.size())
                return closureForState_[state];
            else
                return std::make_pair(-1, -1);
        }

        std::vector<std::pair<s32, s32> > closureForState_;
        std::vector<Closure> closures_;
        std::vector<s32> arcs_;
        std::set<Fsa::StateId> queue_;

        ConstLatticeRef l_;
        std::set<Fsa::LabelId> nonWordLabels_;
        std::set<Fsa::StateId> shortenClosureStates_;
        Fsa::LabelId maxLabel_;
        int maxClosureLength_;
    };

    void expandFilterStates(std::vector<std::vector<s32> >& expanded, const std::vector<s32>& labels, const std::map<std::pair<s32, s32>, s32>& initialFilters, const std::multimap<s32, s32>& initialFilterSuccessors, s32 filterState) {
        if (filterState == -1 || initialFilters.count(std::make_pair(filterState, Fsa::Epsilon))) {
            expanded.push_back(labels);
            return;
        }
        std::pair<std::multimap<s32, s32>::const_iterator, std::multimap<s32, s32>::const_iterator> range = initialFilterSuccessors.equal_range(filterState);
        verify(range.first != range.second);
        for (; range.first != range.second; ++range.first) {
            s32 label = range.first->second;
            verify(label != Fsa::Epsilon);
            std::vector<s32> newLabels(labels);
            newLabels.push_back(label);
            std::map<std::pair<s32, s32>, s32>::const_iterator nextFilterStateIt = initialFilters.find(std::make_pair(filterState, label));
            verify(nextFilterStateIt != initialFilters.end());
            expandFilterStates(expanded, newLabels, initialFilters, initialFilterSuccessors, nextFilterStateIt->second);
        }
    }

    StaticLatticeRef uniqueSentenceAlignmentFilter(ConstLatticeRef l, u32 maxWidth, u32 maxDepth, int maxClosureLength) {
        verify_(l->hasProperty(Fsa::PropertyAcyclic));
        l = sortByTopologicalOrder(l);

        const bool exact = true, filterFinalEpsilon = true;

        std::map<int, std::set<std::vector<int> > > activeStateWords;

        std::set<Fsa::LabelId> nonWordLabels;

        {
            // Collect non-word labels
            nonWordLabels.insert(Fsa::Epsilon);
            Flf::LabelMapRef nonWordEpsilonMap = LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet()));
            for (Fsa::Alphabet::const_iterator label = l->getInputAlphabet()->begin(); label != l->getInputAlphabet()->end(); ++label)
            {
                const LabelMap::Mapping &mapping = (*nonWordEpsilonMap)[label];
                verify(mapping.size() <= 1);
                if (mapping.size() == 1 && mapping.front().label == Fsa::Epsilon)
                    nonWordLabels.insert(label);
            }
        }

        std::cout << "computing closure for unique sentence alignment filter. number of nonword labels: " << nonWordLabels.size() << std::endl;

        LatticeClosure latticeClosure(l, nonWordLabels, maxClosureLength);

        std::cout << "closure ready, have " << latticeClosure.closures_.size() << " closures (" << latticeClosure.shortenClosureStates_.size() << " shortened)" << std::endl;

        Fsa::LabelId positiveEpsilon = latticeClosure.maxLabel_ + 1;

        activeStateWords[0].insert(std::vector<int>());

        // Initialize the new lattice

        StaticLattice *s = new StaticLattice(l->type());
        s->setProperties(l->knownProperties(), l->properties());
        s->setInputAlphabet(l->getInputAlphabet());
        if (s->type() != Fsa::TypeAcceptor)
            s->setOutputAlphabet(l->getOutputAlphabet());
        s->setSemiring(l->semiring());
        s->setInitialStateId(l->initialStateId());
        s->setBoundaries(l->getBoundaries());
        s->setProperties(Fsa::PropertySortedByInputAndTarget, Fsa::PropertyAll);
        s->setDescription(Core::form("unique-sentence-alignment-filter(%s)",
                l->describe().c_str()));

        ArcTraceback traceback;
        WordTraceback wordSequences;

        std::vector<u32> stateRecombinaton;
        std::vector<u32> labelRecombination(positiveEpsilon + 1, -1);

        Fsa::StateId finalStateId = Fsa::InvalidStateId;

        u32 prunedStates = 0, visitedStates = 0;

        // Process the active states and next-words in topological order
        while (!activeStateWords.empty())
        {
            ++visitedStates;
            traceback.clear();
            Fsa::StateId initialStateId = activeStateWords.begin()->first;
            std::set<std::vector<int> > initialWords(activeStateWords.begin()->second);
            verify(!initialWords.empty());
            activeStateWords.erase(activeStateWords.begin());

            if (!s->hasState(initialStateId)) {
                finalStateId = initialStateId;
                ConstStateRef state = l->getState(initialStateId);
                s->setState(new State(state->id(), state->tags(), state->weight()));
            }

            if (l->getState(initialStateId)->isFinal())
            {
                verify(s->getState(initialStateId)->isFinal());
                std::cout << "reached final state" << std::endl;
                continue;
            }

            std::map<std::pair<s32, s32>, s32> initialFilters;
            std::multimap<s32, s32> initialFilterSuccessors;
            s32 filterStateCount = 1;

            for (std::set<std::vector<s32> >::iterator initialWordsIt = initialWords.begin(); initialWordsIt != initialWords.end(); ++initialWordsIt)
            {
                std::vector<s32> words(*initialWordsIt);

                s32 filterState = 0;
                for (std::vector<s32>::const_iterator it = words.begin(); it != words.end(); ++it)
                {
                    std::map<std::pair<s32, s32>, s32>::iterator initialIt = initialFilters.find(std::make_pair(filterState, *it));
                    if (initialIt == initialFilters.end())
                    {
                        initialFilterSuccessors.insert(std::make_pair(filterState, *it));
                        initialFilters.insert(std::make_pair(std::make_pair(filterState, *it), filterStateCount));
                        filterState = filterStateCount;
                        ++filterStateCount;
                    }else{
                        filterState = initialIt->second;
                    }
                }
                std::map<std::pair<s32, s32>, s32>::iterator initialIt = initialFilters.find(std::make_pair(filterState, Fsa::Epsilon));
                if (initialIt == initialFilters.end())
                {
                    initialFilterSuccessors.insert(std::make_pair(filterState, Fsa::Epsilon));
                    initialFilters.insert(std::make_pair(std::make_pair(filterState, Fsa::Epsilon), -1));
                }
            }

            // < < wordTraceback, filterState>, <hypBegin, hypEnd> >
            typedef std::vector<std::pair<std::pair<WordTraceback::Index, s32>, std::pair<u32, u32> > > SequenceHypotheses;
            SequenceHypotheses sequenceHypotheses;
            typedef std::vector<std::pair<Fsa::StateId, ArcTraceback::Index> > Hypotheses;
            Hypotheses hypotheses;

            std::vector<std::pair<std::pair<Fsa::LabelId, s32>, std::pair<s32, s32> > > wordExtensions;       // < <word, link>, <hyp, closure> >
            std::vector<Fsa::LabelId> labels;

            sequenceHypotheses.push_back(std::make_pair(std::make_pair((WordTraceback::Index)WordTraceback::InvalidIndex, 0), std::make_pair(0u, 1u)));
            hypotheses.push_back(std::make_pair(initialStateId, (ArcTraceback::Index)ArcTraceback::InvalidIndex));

            u32 depth = 0;

            while (hypotheses.size())
            {
                SequenceHypotheses oldSequenceHypotheses;
                oldSequenceHypotheses.swap(sequenceHypotheses);
                Hypotheses oldHypotheses;
                oldHypotheses.swap(hypotheses);

                // Step 1: Extend all predecessor hypotheses
                for (SequenceHypotheses::const_iterator wordSequenceIt = oldSequenceHypotheses.begin(); wordSequenceIt != oldSequenceHypotheses.end(); ++wordSequenceIt)
                {
                    WordTraceback::Index wordSequence = wordSequenceIt->first.first;
                    int filterState = wordSequenceIt->first.second;
                    Hypotheses::const_iterator hypBegin = oldHypotheses.begin() + wordSequenceIt->second.first;
                    Hypotheses::const_iterator hypEnd = oldHypotheses.begin() + wordSequenceIt->second.second;

                    wordExtensions.reserve(std::max(wordExtensions.capacity(), wordExtensions.size()));
                    wordExtensions.clear();
                    labels.reserve(std::max(labels.capacity(), labels.size()));
                    labels.clear();

                    for (Hypotheses::const_iterator stateHypIt = hypBegin; stateHypIt != hypEnd; ++stateHypIt)
                    {
                        std::pair<s32, s32> closures = latticeClosure.getClosures(stateHypIt->first);
                        verify(closures.first != -1);

                        for (s32 closureI = closures.first; closureI != closures.second; ++closureI)
                        {
                            const LatticeClosure::Closure& closure(latticeClosure.closures_[closureI]);
                            Fsa::LabelId label = closure.word != Fsa::Epsilon ? closure.word : positiveEpsilon;
                            u32& labelRecomb(labelRecombination[label]);
                            if (labelRecomb >= wordExtensions.size() || wordExtensions[labelRecomb].first.first != label)
                            {
                                wordExtensions.push_back(std::make_pair(std::make_pair(label, -1), std::make_pair(s32(stateHypIt - hypBegin), closureI)));
                                labels.push_back(label);
                            }else{
                                wordExtensions.push_back(std::make_pair(std::make_pair(label, labelRecomb), std::make_pair(s32(stateHypIt - hypBegin), closureI)));
                            }
                            labelRecomb = wordExtensions.size() - 1;
                        }
                    }

                    for (std::vector<Fsa::LabelId>::const_iterator labelIt = labels.begin(); labelIt != labels.end(); ++labelIt)
                    {
                        Fsa::LabelId label = *labelIt;
                        int newFilterState = -1;
                        if (filterState != -1 && (label != positiveEpsilon || filterFinalEpsilon))
                        {
                            if (initialFilters.count(std::make_pair(filterState, Fsa::Epsilon))) {
                                // Ok, newFilterState is -1, the search is free now
                            }else{
                                std::map<std::pair<int, int>, int>::iterator initialIt = initialFilters.find(std::make_pair(filterState, label));
                                if (initialIt == initialFilters.end())
                                    continue;  // Skip
                                newFilterState = initialIt->second;
                            }
                        }

                        u32 firstHyp = hypotheses.size();

                        s32 currentExtension = labelRecombination[label];
                        verify(currentExtension != -1);

                        while (currentExtension != -1)
                        {
                            verify(currentExtension < wordExtensions.size());
                            const std::pair<std::pair<Fsa::LabelId, s32>, std::pair<s32, s32> >& extension(wordExtensions[currentExtension]);
                            verify(extension.first.first == label);

                            std::pair <Fsa::StateId, Flf::WordTraceback::Index> oldHypothesis(*(hypBegin + extension.second.first));
                            s32 closureIndex = extension.second.second;
                            verify(closureIndex < latticeClosure.closures_.size());
                            const LatticeClosure::Closure& closure(latticeClosure.closures_[closureIndex]);
                            Score newScore = closure.score;
                            if(oldHypothesis.second != -1)
                                newScore += traceback.get(oldHypothesis.second).score;

                            if (closure.target >= stateRecombinaton.size())
                                stateRecombinaton.resize(closure.target + 1, -1);

                            verify(closure.target < stateRecombinaton.size());
                            u32& recomb = stateRecombinaton[closure.target];

                            if (recomb < firstHyp || recomb >= hypotheses.size() || hypotheses[recomb].first != closure.target)
                            {
                                recomb = hypotheses.size();
                                hypotheses.push_back(std::make_pair(
                                        closure.target,
                                        traceback.append(oldHypothesis.second, oldHypothesis.first, closureIndex, closure.word, newScore)));
                            }else if(newScore < traceback.get(hypotheses[recomb].second).score)
                            {
                                hypotheses[recomb].second =
                                    traceback.append(oldHypothesis.second, oldHypothesis.first, closureIndex, closure.word, newScore);
                            }
                            currentExtension = extension.first.second;
                        }

                        sequenceHypotheses.push_back(std::make_pair(std::make_pair(wordSequences.append(wordSequence, label), newFilterState), std::make_pair(firstHyp, (u32)hypotheses.size())));
                    }
                }

                if (hypotheses.empty())
                    break;

                depth += 1;

                if (hypotheses.size() > maxWidth || depth > maxDepth)
                {
                    ++prunedStates;
                    // If we've crossed our beam limits, just keep the best path for each hypothesis word sequence
                    Speech::TimeframeIndex startTime = l->boundary(initialStateId).time();

                    for (SequenceHypotheses::iterator wordSequenceIt = sequenceHypotheses.begin(); wordSequenceIt != sequenceHypotheses.end(); ++wordSequenceIt)
                    {
                        WordTraceback::Index wordSequence = wordSequenceIt->first.first;
                        Hypotheses::iterator hypBegin = hypotheses.begin() + wordSequenceIt->second.first;
                        Hypotheses::iterator hypEnd = hypotheses.begin() + wordSequenceIt->second.second;

                        Score bestNormalizedScore = Core::Type<Score>::max;
                        std::pair<Fsa::StateId, s32> bestHypothesis;

                        for (Hypotheses::const_iterator stateIt = hypBegin; stateIt != hypEnd; ++stateIt)
                        {
                            const ArcTraceback::BackPointer& bp(traceback.get(stateIt->second));
                            Fsa::StateId nextState = latticeClosure.closures_[bp.arc].target;
                            Speech::TimeframeIndex nextTime = l->boundary(nextState).time();
                            if (nextTime == startTime)
                                nextTime += 1;
                            Score normalizedScore = bp.score / (nextTime - startTime);
                            if (normalizedScore < bestNormalizedScore)
                            {
                                bestNormalizedScore = normalizedScore;
                                bestHypothesis = *stateIt;
                            }
                        }
                        if (bestNormalizedScore != Core::Type<Score>::max)
                        {
                            // Keep only the best hypothesis
                            *hypBegin = bestHypothesis;
                            wordSequenceIt->second.second = wordSequenceIt->second.first + 1;
                        }
                    }
                }

                // Step 2: Check for intersections, eventually activate successor states
                for (SequenceHypotheses::iterator wordSequenceIt = sequenceHypotheses.begin(); wordSequenceIt != sequenceHypotheses.end(); ++wordSequenceIt)
                {
                    WordTraceback::Index wordSequence = wordSequenceIt->first.first;
                    int filterState = wordSequenceIt->first.second;
                    Hypotheses::const_iterator hypBegin = hypotheses.begin() + wordSequenceIt->second.first;
                    Hypotheses::const_iterator hypEnd = hypotheses.begin() + wordSequenceIt->second.second;

                    s32 intersection = hypBegin->second;

                    for (Hypotheses::const_iterator stateIt = hypBegin; stateIt != hypEnd; ++stateIt)
                    {
                        intersection = traceback.intersect(intersection, stateIt->second);
                        if (intersection == ArcTraceback::InvalidIndex)
                            break;
                    }

                    if (intersection != ArcTraceback::InvalidIndex)
                    {
                        const ArcTraceback::BackPointer& bp(traceback.get(intersection));
                        Fsa::StateId nextState = latticeClosure.closures_[bp.arc].target;
                        // Ready! Add the intersection.
                        s32 traceLength = traceback.length(intersection);
                        s32 seqLength = wordSequences.length(wordSequence);
                        verify(traceLength <= seqLength);

                        std::vector<Fsa::LabelId> labels;
                        if (traceLength == seqLength) {
                            // We've matched exactly at the end of the sequence, so we need _all_ successor words. An empty sequence will do.
                        } else {
                            // Add successor words to follow
                            WordTraceback::Index wordTail = wordSequence;
                            int skip = seqLength - traceLength - 1;
                            while (skip) {
                                if (exact)
                                    labels.push_back(wordSequences.get(wordTail).label);
                                wordTail = wordSequences.get(wordTail).pre;
                                --skip;
                            }
                            labels.push_back(wordSequences.get(wordTail).label);
                            std::reverse(labels.begin(), labels.end());
                        }

                        if (filterState == -1 || initialFilters.count(std::make_pair(filterState, Fsa::Epsilon)))
                        {
                            activeStateWords[nextState].insert(labels);
                        }else if(!activeStateWords[nextState].count(std::vector<int>())) {
                            std::vector<std::vector<int> > expanded;
                            expandFilterStates(expanded, labels, initialFilters, initialFilterSuccessors, filterState);
                            std::cout << "expanded " << expanded.size() << " sequences" << std::endl;
                            for (std::vector<std::vector<int> >::const_iterator labelsIt = expanded.begin(); labelsIt != expanded.end(); ++labelsIt)
                                activeStateWords[nextState].insert(*labelsIt);
                        }

                        s32 keepBpI = intersection;
                        while (keepBpI != ArcTraceback::InvalidIndex)
                        {
                            const ArcTraceback::BackPointer& keepBp(traceback.get(keepBpI));

                            std::vector <s32>::const_iterator arcsBegin = latticeClosure.arcs_.begin() + latticeClosure.closures_[keepBp.arc].arcs;
                            std::vector <s32>::const_iterator arcsEnd =
                                (keepBp.arc == latticeClosure.closures_.size() - 1) ?
                                latticeClosure.arcs_.end() :  latticeClosure.arcs_.begin() + latticeClosure.closures_[keepBp.arc + 1].arcs;

                            verify(arcsEnd <= latticeClosure.arcs_.end());

                            Fsa::StateId stateId = keepBp.state;
                            for (std::vector <s32>::const_iterator arcIt = arcsBegin; arcIt != arcsEnd; ++arcIt)
                            {
                                ConstStateRef state = l->getState(stateId);
                                if (!s->hasState(stateId))
                                    s->setState(new State(state->id(), state->tags(), state->weight()));
                                verify(*arcIt < state->nArcs());
                                const Flf::Arc& a = *state->getArc(*arcIt);

                                State *sp = s->fastState(stateId);

                                State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                                if ((pos == sp->end())
                                    || (a.target() != pos->target()) || (a.input() != pos->input()))
                                    sp->insert(pos, a);

                                stateId = a.target();
                            }

                            keepBpI = keepBp.pre;
                        }

                        // Don't follow these hypotheses any more
                        wordSequenceIt->second.second = wordSequenceIt->second.first = 0;
                    }
                }
            }
        }

        verify(s->boundary(s->initialStateId()) == l->boundary(l->initialStateId()));
        verify(s->boundary(finalStateId) == l->boundary(finalStateId));
        std::cout << "total lattice states: " << finalStateId + 1 << ", visited " << visitedStates << ", approximated " << prunedStates << std::endl;

        return StaticLatticeRef(s);
    }

    static const Core::ParameterInt paramMaxWidth(
        "max-width",
        "maximum number of concurrent hypotheses allowed during unique alignment filtering",
        100000);
    static const Core::ParameterInt paramMaxDepth(
        "max-depth",
        "maximum local expansion depth allowed during unique alignment filtering",
        10);

    static const Core::ParameterInt paramMaxClosureLength(
        "max-closure-length",
        "maximum length of non-word closures (in timeframes)",
        1000);

    // -------------------------------------------------------------------------
    class UniqueSentenceAlignmentFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = uniqueSentenceAlignmentFilter(l, maxWidth_, maxDepth_, maxClosureLength_);
            verify(l->type() == Fsa::TypeAcceptor);
            return l;
        }
        s32 maxWidth_, maxDepth_, maxClosureLength_;
    public:
        UniqueSentenceAlignmentFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config),
            maxWidth_(100000),
            maxDepth_(10),
            maxClosureLength_(200) {}
        ~UniqueSentenceAlignmentFilterNode() {}
        virtual void init(const std::vector<std::string> &arguments) {
            maxWidth_ = paramMaxWidth(config);
            maxDepth_ = paramMaxDepth(config);
            maxClosureLength_ = paramMaxClosureLength(config);
            log() << "max width " << maxWidth_ << " max depth " << maxDepth_ << " max closure length " << maxClosureLength_;
        }
    };

    NodeRef createUniqueSentenceAlignmentFilterNode(
        const std::string &name, const Core::Configuration &config) {
        return NodeRef(new UniqueSentenceAlignmentFilterNode(name, config));
    }

    // -------------------------------------------------------------------------
    StaticLatticeRef applyEpsClosureFilter(ConstLatticeRef l) {
        verify_(l->hasProperty(Fsa::PropertyAcyclic));
        l = sort(l, Fsa::SortTypeByInputAndTarget);
        l = persistent(l);
        ConstStateMapRef topologicalOrderMap = findTopologicalOrder(l);
        require(topologicalOrderMap);
        TopologicalOrderQueueRef nonEpsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &Q = *nonEpsQueue;
        TopologicalOrderQueueRef epsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &epsQ = *epsQueue;
        StateIdList epsClosure, epsHull;
        const ScoreList &scales = l->semiring()->scales();

        StaticLattice *s = new StaticLattice(l->type());
        s->setProperties(l->knownProperties(), l->properties());
        s->setInputAlphabet(l->getInputAlphabet());
        if (s->type() != Fsa::TypeAcceptor)
            s->setOutputAlphabet(l->getOutputAlphabet());
        s->setSemiring(l->semiring());
        s->setInitialStateId(l->initialStateId());
        s->setBoundaries(l->getBoundaries());
        s->setProperties(Fsa::PropertySortedByInputAndTarget, Fsa::PropertyAll);
        s->setDescription(Core::form("eps-closure-filter(%s)",
                                     l->describe().c_str()));

        HypList hyps(topologicalOrderMap->maxSid + 1);
        Core::Vector<bool> visited(topologicalOrderMap->maxSid + 1, false);
        Q.insert(l->initialStateId());
        while (!Q.empty()) {
            Fsa::StateId sid = Q.top(); Q.pop();
            if (visited[sid])
                continue;
            visited[sid] = true;
            verify((hyps[sid].score == Semiring::Max) && (hyps[sid].bptr == Fsa::InvalidStateId));
            // filter epsilon closure
            hyps[sid].score = Semiring::One;
            epsQ.insert(sid);
            while (!epsQ.empty()) {
                Fsa::StateId epsSid = epsQ.top(); epsQ.pop();
                ConstStateRef epsSr = l->getState(epsSid);
                Score fwdScore = hyps[epsSid].score;
                State::const_iterator epsA = epsSr->begin(), epsEnd = epsSr->end();
                for (; (epsA != epsEnd) && (epsA->input() == Fsa::Epsilon); ++epsA) {
                    Score score = fwdScore + epsA->weight()->project(scales);
                    Hyp &hyp = hyps[epsA->target()];
                    if (!hyp.visited || (score < hyp.score)) {
                        if (!hyp.visited) {
                            hyp.visited = true;
                            epsQ.insert(epsA->target());
                        }
                        hyp.score = score;
                        hyp.bptr = epsSid;
                        hyp.a = epsA;
                    }
                }
                if (epsA != epsEnd) {
                    Q.insert(epsSid);
                    epsHull.push_back(epsSid);
                } else if (epsSr->isFinal()){
                    epsHull.push_back(epsSid);
                }
                epsClosure.push_back(epsSid);
            }
            // add epsilon arcs; reset traceback arrays
            for (StateIdList::const_iterator itSid = epsHull.begin(), endSid = epsHull.end(); itSid != endSid; ++itSid) {
                Fsa::StateId epsSid = *itSid;
                if (!s->hasState(epsSid)) {
                    ConstStateRef epsSr = l->getState(epsSid);
                    s->setState(new State(epsSr->id(), epsSr->tags(), epsSr->weight()));
                }
                for (;;) {
                    Hyp &hyp = hyps[epsSid];
                    if (hyp.bptr == Fsa::InvalidStateId)
                        break;
                    if (!s->hasState(hyp.bptr)) {
                        ConstStateRef bptrSr = l->getState(hyp.bptr);
                        s->setState(new State(bptrSr->id(), bptrSr->tags(), bptrSr->weight()));
                    }
                    State *sp = s->fastState(hyp.bptr);
                    const Arc &a = *hyp.a;
                    State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                    if ((pos == sp->end())
                        || (a.target() != pos->target()) || (a.input() != pos->input()))
                        sp->insert(pos, a);
                    epsSid = hyp.bptr;
                    hyp.bptr = Fsa::InvalidStateId;
                }
            }
            epsHull.clear();
            for (StateIdList::const_iterator itSid = epsClosure.begin(), endSid = epsClosure.end(); itSid != endSid; ++itSid) {
                Hyp &hyp = hyps[*itSid];
                hyp.visited = false;
                hyp.score = Semiring::Max;
                hyp.bptr = Fsa::InvalidStateId;
            }
            epsClosure.clear();
            // add non-epsilon arcs
            ConstStateRef sr = l->getState(sid);
            State::const_iterator a = sr->begin(), a_end = sr->end();
            verify(s->hasState(sid));
            State *sp = s->fastState(sid);
            for (; (a != a_end) && (a->input() == Fsa::Epsilon); ++a);
            for (; a != a_end; ++a) {
                *sp->newArc() = *a;
                Q.insert(a->target());
            }
            verify(!hyps[sid].visited && (hyps[sid].bptr == Fsa::InvalidStateId));
            // hyps[sid].score = Semiring::Max; // should not be necessary
            verify(s->hasState(sid));
        }

        return StaticLatticeRef(s);
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    StaticLatticeRef applyEpsClosureWeakDeterminizationFilter(ConstLatticeRef l) {
        verify_(l->hasProperty(Fsa::PropertyAcyclic));
        l = sort(l, Fsa::SortTypeByInputAndTarget);
        l = persistent(l);
        ConstStateMapRef topologicalOrderMap = findTopologicalOrder(l);
        require(topologicalOrderMap);
        TopologicalOrderQueueRef nonEpsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &Q = *nonEpsQueue;
        TopologicalOrderQueueRef epsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &epsQ = *epsQueue;
        StateIdList epsClosure, epsHull;
        const ScoreList &scales = l->semiring()->scales();

        StaticLattice *s = new StaticLattice(l->type());
        s->setProperties(l->knownProperties(), l->properties());
        s->setInputAlphabet(l->getInputAlphabet());
        if (s->type() != Fsa::TypeAcceptor)
            s->setOutputAlphabet(l->getOutputAlphabet());
        s->setSemiring(l->semiring());
        s->setInitialStateId(l->initialStateId());
        s->setBoundaries(l->getBoundaries());
        s->setProperties(Fsa::PropertySortedByInputAndTarget, Fsa::PropertyAll);
        s->setDescription(Core::form("eps-closure-weak-determinization-filter(%s)",
                                     l->describe().c_str()));

        HypList hyps(topologicalOrderMap->maxSid + 1);
        Core::Vector<bool> visited(topologicalOrderMap->maxSid + 1, false);
        Q.insert(l->initialStateId());
        while (!Q.empty()) {
            Fsa::StateId sid = Q.top(); Q.pop();
            if (visited[sid])
                continue;
            visited[sid] = true;
            // initialize source state
            verify((hyps[sid].score == Semiring::Max) && (hyps[sid].bptr == Fsa::InvalidStateId));
            hyps[sid].score = Semiring::One;
            ConstStateRef sr = l->getState(sid);
            State::const_iterator a = sr->begin(), a_end = sr->end();
            if (sid != l->initialStateId())
                for (; (a != a_end) && (a->input() == Fsa::Epsilon); ++a);
            while (a != a_end) {
                // initialize label-epsilon closure
                for(Fsa::LabelId label = a->input(); (a != a_end) && (a->input() == label); ++a) {
                    Score score = a->weight()->project(scales);
                    Hyp &hyp = hyps[a->target()];
                    if (!hyp.visited || (score < hyp.score)) {
                        if (!hyp.visited) {
                            hyp.visited = true;
                            epsQ.insert(a->target());
                        }
                        hyp.score = score;
                        hyp.bptr = sid;
                        hyp.a = a;
                    }
                }
                verify_(!epsQ.empty());
                // filter label-epsilon closure
                while (!epsQ.empty()) {
                    Fsa::StateId epsSid = epsQ.top(); epsQ.pop();
                    ConstStateRef epsSr = l->getState(epsSid);
                    Score fwdScore = hyps[epsSid].score;
                    State::const_iterator epsA = epsSr->begin(), epsEnd = epsSr->end();
                    for (; (epsA != epsEnd) && (epsA->input() == Fsa::Epsilon); ++epsA) {
                        Score score = fwdScore + epsA->weight()->project(scales);
                        Hyp &hyp = hyps[epsA->target()];
                        if (!hyp.visited || (score < hyp.score)) {
                            if (!hyp.visited) {
                                hyp.visited = true;
                                epsQ.insert(epsA->target());
                            }
                            hyp.score = score;
                            hyp.bptr = epsSid;
                            hyp.a = epsA;
                        }
                    }
                    if (epsA != epsEnd) {
                        Q.insert(epsSid);
                        epsHull.push_back(epsSid);
                    } else if (epsSr->isFinal()){
                        epsHull.push_back(epsSid);
                    }
                    epsClosure.push_back(epsSid);
                }
                // add label and epsilon arcs; reset traceback arrays
                for (StateIdList::const_iterator itSid = epsHull.begin(), endSid = epsHull.end(); itSid != endSid; ++itSid) {
                    Fsa::StateId epsSid = *itSid;
                    if (!s->hasState(epsSid)) {
                        ConstStateRef epsSr = l->getState(epsSid);
                        s->setState(new State(epsSr->id(), epsSr->tags(), epsSr->weight()));
                    }
                    for (;;) {
                        Hyp &hyp = hyps[epsSid];
                        if (hyp.bptr == Fsa::InvalidStateId)
                            break;
                        if (!s->hasState(hyp.bptr)) {
                            ConstStateRef bptrSr = l->getState(hyp.bptr);
                            s->setState(new State(bptrSr->id(), bptrSr->tags(), bptrSr->weight()));
                        }
                        State *sp = s->fastState(hyp.bptr);
                        const Arc &a = *hyp.a;
                        State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                        if ((pos == sp->end())
                            || (a.target() != pos->target()) || (a.input() != pos->input()))
                            sp->insert(pos, a);
                        epsSid = hyp.bptr;
                        hyp.bptr = Fsa::InvalidStateId;
                    }
                }
                verify_(s->hasState(sid) && (s->fastState(sid)->hasArcs()));
                epsHull.clear();
                for (StateIdList::const_iterator itSid = epsClosure.begin(), endSid = epsClosure.end(); itSid != endSid; ++itSid) {
                    Hyp &hyp = hyps[*itSid];
                    hyp.visited = false;
                    hyp.score = Semiring::Max;
                    hyp.bptr = Fsa::InvalidStateId;
                }
                epsClosure.clear();
            }
            if (!s->hasState(sid)) {
                ConstStateRef sr = l->getState(sid);
                s->setState(new State(sr->id(), sr->tags(), sr->weight()));
            }
            verify(!hyps[sid].visited && (hyps[sid].bptr == Fsa::InvalidStateId));
            hyps[sid].score = Semiring::Max;
            verify(s->hasState(sid));
        }

        return StaticLatticeRef(s);
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    StaticLatticeRef applyEpsClosureStrongDeterminizationFilter(ConstLatticeRef l) {
        verify_(l->hasProperty(Fsa::PropertyAcyclic));
        l = sort(l, Fsa::SortTypeByInputAndTarget);
        l = persistent(l);
        ConstStateMapRef topologicalOrderMap = findTopologicalOrder(l);
        require(topologicalOrderMap);
        TopologicalOrderQueueRef nonEpsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &Q = *nonEpsQueue;
        TopologicalOrderQueueRef epsQueue =
            createTopologicalOrderQueue(l, topologicalOrderMap);
        TopologicalOrderQueue &epsQ = *epsQueue;
        StateIdList leftEpsClosure, leftEpsFinalsHull, rightEpsClosure, rightEpsHull;
        typedef Core::Vector<std::pair<Fsa::StateId, std::pair<State::const_iterator, State::const_iterator> > > StateArcRangeList;
        StateArcRangeList leftEpsExtendedHull;
        const ScoreList &scales = l->semiring()->scales();

        StaticLattice *s = new StaticLattice(l->type());
        s->setProperties(l->knownProperties(), l->properties());
        s->setInputAlphabet(l->getInputAlphabet());
        if (s->type() != Fsa::TypeAcceptor)
            s->setOutputAlphabet(l->getOutputAlphabet());
        s->setSemiring(l->semiring());
        s->setInitialStateId(l->initialStateId());
        s->setBoundaries(l->getBoundaries());
        s->setProperties(Fsa::PropertySortedByInputAndTarget, Fsa::PropertyAll);
        s->setDescription(Core::form("eps-closure-strong-determinization-filter(%s)",
                                     l->describe().c_str()));

        HypList
            leftHyps(topologicalOrderMap->maxSid + 1),
            rightHyps(topologicalOrderMap->maxSid + 1);
        Core::Vector<bool> visited(topologicalOrderMap->maxSid + 1, false);
        Q.insert(l->initialStateId());
        while (!Q.empty()) {
            Fsa::StateId sid = Q.top(); Q.pop();
            if (visited[sid])
                continue;
            visited[sid] = true;
            // initialize left epsilon closure
            verify((leftHyps[sid].score == Semiring::Max) && (leftHyps[sid].bptr == Fsa::InvalidStateId));
            leftHyps[sid].score = Semiring::One;
            Fsa::LabelId nextLabel = Core::Type<Fsa::LabelId>::max;
            epsQ.insert(sid);
            while (!epsQ.empty()) {
                Fsa::StateId epsSid = epsQ.top(); epsQ.pop();
                ConstStateRef epsSr = l->getState(epsSid);
                Score fwdScore = leftHyps[epsSid].score;
                State::const_iterator epsA = epsSr->begin(), epsEnd = epsSr->end();
                for (; (epsA != epsEnd) && (epsA->input() == Fsa::Epsilon); ++epsA) {
                    Score score = fwdScore + epsA->weight()->project(scales);
                    Hyp &leftHyp = leftHyps[epsA->target()];
                    if (!leftHyp.visited || (score < leftHyp.score)) {
                        if (!leftHyp.visited) {
                            leftHyp.visited = true;
                            epsQ.insert(epsA->target());
                        }
                        leftHyp.score = score;
                        leftHyp.bptr = epsSid;
                        leftHyp.a = epsA;
                    }
                }
                if (epsA != epsEnd) {
                    leftEpsExtendedHull.push_back(std::make_pair(epsSid, std::make_pair(epsA, epsEnd)));
                    if (epsA->input() < nextLabel) nextLabel = epsA->input();
                }
                if (epsSr->isFinal())
                    leftEpsFinalsHull.push_back(epsSid);
                leftEpsClosure.push_back(epsSid);
            }
            // add left finals
            for (StateIdList::const_iterator itSid = leftEpsFinalsHull.begin(), endSid = leftEpsFinalsHull.end(); itSid != endSid; ++itSid) {
                Fsa::StateId epsSid = *itSid;
                if (!s->hasState(epsSid)) {
                    ConstStateRef epsSr = l->getState(epsSid);
                    s->setState(new State(epsSr->id(), epsSr->tags(), epsSr->weight()));
                }
                for (;;) {
                    Hyp &leftHyp = leftHyps[epsSid];
                    if (leftHyp.bptr == Fsa::InvalidStateId)
                        break;
                    if (!s->hasState(leftHyp.bptr)) {
                        ConstStateRef bptrSr = l->getState(leftHyp.bptr);
                        s->setState(new State(bptrSr->id(), bptrSr->tags(), bptrSr->weight()));
                    }
                    State *sp = s->fastState(leftHyp.bptr);
                    const Arc &a = *leftHyp.a;
                    State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                    if ((pos == sp->end())
                        || (a.target() != pos->target()) || (a.input() != pos->input()))
                        sp->insert(pos, a);
                    epsSid = leftHyp.bptr;
                    leftHyp.bptr = Fsa::InvalidStateId;
                }
            }
            leftEpsFinalsHull.clear();
            // filter right label-epsilon closure
            while (nextLabel != Core::Type<Fsa::LabelId>::max) {
                Fsa::LabelId label = nextLabel;
                nextLabel = Core::Type<Fsa::LabelId>::max;
                for (StateArcRangeList::iterator itSidRange = leftEpsExtendedHull.begin(), endSidRange = leftEpsExtendedHull.end(); itSidRange != endSidRange; ++itSidRange) {
                    Fsa::StateId epsSid = itSidRange->first;
                    Score fwdScore = leftHyps[epsSid].score;
                    std::pair<State::const_iterator, State::const_iterator> &epsRange = itSidRange->second;
                    for (; (epsRange.first != epsRange.second) && (epsRange.first->input() == label); ++epsRange.first) {
                        Score score = fwdScore + epsRange.first->weight()->project(scales);
                        Hyp &rightHyp = rightHyps[epsRange.first->target()];
                        if (!rightHyp.visited || (score < rightHyp.score)) {
                            if (!rightHyp.visited) {
                                rightHyp.visited = true;
                                epsQ.insert(epsRange.first->target());
                            }
                            rightHyp.score = score;
                            rightHyp.bptr = epsSid;
                            rightHyp.a = epsRange.first;
                        }
                    }
                    if ((epsRange.first != epsRange.second) && (epsRange.first->input() < nextLabel))
                        nextLabel = epsRange.first->input();
                }
                while (!epsQ.empty()) {
                    Fsa::StateId epsSid = epsQ.top(); epsQ.pop();
                    ConstStateRef epsSr = l->getState(epsSid);
                    Score fwdScore = rightHyps[epsSid].score;
                    State::const_iterator epsA = epsSr->begin(), epsEnd = epsSr->end();
                    for (; (epsA != epsEnd) && (epsA->input() == Fsa::Epsilon); ++epsA) {
                        Score score = fwdScore + epsA->weight()->project(scales);
                        Hyp &rightHyp = rightHyps[epsA->target()];
                        if (!rightHyp.visited || (score < rightHyp.score)) {
                            if (!rightHyp.visited) {
                                rightHyp.visited = true;
                                epsQ.insert(epsA->target());
                            }
                            rightHyp.score = score;
                            rightHyp.bptr = epsSid;
                            rightHyp.a = epsA;
                        }
                    }
                    if (epsA != epsEnd) {
                        Q.insert(epsSid);
                        rightEpsHull.push_back(epsSid);
                    } else if (epsSr->isFinal())
                        rightEpsHull.push_back(epsSid);
                    rightEpsClosure.push_back(epsSid);
                }
                // add label and epsilon arcs from right and left closures; reset traceback arrays
                for (StateIdList::const_iterator itSid = rightEpsHull.begin(), endSid = rightEpsHull.end(); itSid != endSid; ++itSid) {
                    // right epsilon closure
                    Fsa::StateId epsSid = *itSid;
                    if (!s->hasState(epsSid)) {
                        ConstStateRef epsSr = l->getState(epsSid);
                        s->setState(new State(epsSr->id(), epsSr->tags(), epsSr->weight()));
                    }
                    for (;;) {
                        Hyp &rightHyp = rightHyps[epsSid];
                        if (rightHyp.bptr == Fsa::InvalidStateId) {
                            // left epsilon closure
                            for (;;) {
                                Hyp &leftHyp = leftHyps[epsSid];
                                if (leftHyp.bptr == Fsa::InvalidStateId)
                                    break;
                                if (!s->hasState(leftHyp.bptr)) {
                                    ConstStateRef bptrSr = l->getState(leftHyp.bptr);
                                    s->setState(new State(bptrSr->id(), bptrSr->tags(), bptrSr->weight()));
                                }
                                State *sp = s->fastState(leftHyp.bptr);
                                const Arc &a = *leftHyp.a;
                                State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                                if ((pos == sp->end())
                                    || (a.target() != pos->target()) || (a.input() != pos->input()))
                                    sp->insert(pos, a);
                                epsSid = leftHyp.bptr;
                                leftHyp.bptr = Fsa::InvalidStateId;
                            }
                            break;
                        }
                        if (!s->hasState(rightHyp.bptr)) {
                            ConstStateRef bptrSr = l->getState(rightHyp.bptr);
                            s->setState(new State(bptrSr->id(), bptrSr->tags(), bptrSr->weight()));
                        }
                        State *sp = s->fastState(rightHyp.bptr);
                        const Arc &a = *rightHyp.a;
                        State::iterator pos = sp->lower_bound(a, Ftl::byInputAndTarget<Lattice>());
                        if ((pos == sp->end())
                            || (a.target() != pos->target()) || (a.input() != pos->input()))
                            sp->insert(pos, a);
                        epsSid = rightHyp.bptr;
                        rightHyp.bptr = Fsa::InvalidStateId;
                    }
                }
                rightEpsHull.clear();
                for (StateIdList::const_iterator itSid = rightEpsClosure.begin(), endSid = rightEpsClosure.end(); itSid != endSid; ++itSid) {
                    Hyp &rightHyp = rightHyps[*itSid];
                    rightHyp.visited = false;
                    rightHyp.score = Semiring::Max;
                    rightHyp.bptr = Fsa::InvalidStateId;
                }
                rightEpsClosure.clear();
            }
            for (StateIdList::const_iterator itSid = leftEpsClosure.begin(), endSid = leftEpsClosure.end(); itSid != endSid; ++itSid) {
                Hyp &leftHyp = leftHyps[*itSid];
                leftHyp.score = Semiring::Max;
                leftHyp.bptr = Fsa::InvalidStateId;
            }
            leftEpsClosure.clear();
            verify(!leftHyps[sid].visited && (leftHyps[sid].bptr == Fsa::InvalidStateId));
            leftHyps[sid].score = Semiring::Max;
            verify(s->hasState(sid));
        }

        return StaticLatticeRef(s);
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NonWordClosureFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = transducer(l);
            l = applyOneToOneLabelMap(l, LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet())));
            l = applyEpsClosureFilter(l);
            l = projectOutput(l);
            l->setProperties(Fsa::PropertySorted, 0);
            verify(l->type() == Fsa::TypeAcceptor);
            return l;
        }
    public:
        NonWordClosureFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config) {}
        ~NonWordClosureFilterNode() {}
    };
    NodeRef createNonWordClosureFilterNode(
        const std::string &name, const Core::Configuration &config) {
        return NodeRef(new NonWordClosureFilterNode(name, config));
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NonWordClosureWeakDeterminizationFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = transducer(l);
            l = applyOneToOneLabelMap(l, LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet())));
            l = applyEpsClosureWeakDeterminizationFilter(l);
            l = projectOutput(l);
            l->setProperties(Fsa::PropertySorted, 0);
            verify(l->type() == Fsa::TypeAcceptor);
            return l;
        }
    public:
        NonWordClosureWeakDeterminizationFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config) {}
        ~NonWordClosureWeakDeterminizationFilterNode() {}
    };
    NodeRef createNonWordClosureWeakDeterminizationFilterNode(
        const std::string &name, const Core::Configuration &config) {
        return NodeRef(new NonWordClosureWeakDeterminizationFilterNode(name, config));
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NonWordClosureStrongDeterminizationFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = transducer(l);
            l = applyOneToOneLabelMap(l, LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet())));
            l = applyEpsClosureStrongDeterminizationFilter(l);
            l = projectOutput(l);
            l->setProperties(Fsa::PropertySorted, 0);
            verify(l->type() == Fsa::TypeAcceptor);
            return l;
        }
    public:
        NonWordClosureStrongDeterminizationFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config) {}
        ~NonWordClosureStrongDeterminizationFilterNode() {}
    };
    NodeRef createNonWordClosureStrongDeterminizationFilterNode(
        const std::string &name, const Core::Configuration &config) {
        return NodeRef(new NonWordClosureStrongDeterminizationFilterNode(name, config));
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NormalizeEpsilonClosureLattice : public SlaveLattice {
    private:
        ConstSemiringRef semiring_;
        mutable TopologicalOrderQueueRef epsQeue_;
        mutable Core::Vector<ScoresRef> epsClosureScores_;

    public:
        NormalizeEpsilonClosureLattice(ConstLatticeRef l) :
            SlaveLattice(cache(sort(l, Fsa::SortTypeByInputAndOutputAndTarget))),
            semiring_(l->semiring()) {
            ConstStateMapRef topologicalOrderMap = findTopologicalOrder(l);
            verify(topologicalOrderMap && (topologicalOrderMap->maxSid != Fsa::InvalidStateId));
            epsQeue_ = createTopologicalOrderQueue(l, topologicalOrderMap);
            epsClosureScores_.grow(topologicalOrderMap->maxSid);
        }
        virtual ~NormalizeEpsilonClosureLattice() {}

        virtual ConstStateRef getState(Fsa::StateId sid) const {
            ConstStateRef sr = fsa_->getState(sid);
            State::const_iterator a = sr->begin(), a_end = sr->end();

            // Check in O(1), if eps/eps arcs exist
            if ((a == a_end)
                || (a->input() != Fsa::Epsilon)
                || (a->output() != Fsa::Epsilon))
                return sr;

            // Initialize epsilon closure
            verify(sid < epsClosureScores_.size());
            TopologicalOrderQueue &epsQ(*epsQeue_);
            State *sp = new State(sr->id());
            for (; (a != a_end) && (a->input() == Fsa::Epsilon) && (a->output() == Fsa::Epsilon); ++a)
                if (!epsClosureScores_[a->target()]) {
                    epsClosureScores_[a->target()] = a->weight();
                    epsQ.insert(a->target());
                } else
                    epsClosureScores_[a->target()] =
                        semiring_->collect(epsClosureScores_[a->target()], a->weight());
            for (; a != a_end; ++a)
                *sp->newArc() = *a;
            // Initialize (potential) final weight
            ScoresRef finalWeight;
            if (sr->isFinal()) {
                sp->addTags(Fsa::StateTagFinal);
                finalWeight = sr->weight();
            } else
                finalWeight = semiring_->zero();
            // Process epsilon closure
            while (!epsQ.empty()) {
                Fsa::StateId epsSid = epsQ.top(); epsQ.pop();
                ConstStateRef epsSr = fsa_->getState(epsSid);
                ScoresRef score = epsClosureScores_[epsSid];
                epsClosureScores_[epsSid].reset();
                State::const_iterator a = epsSr->begin(), a_end = epsSr->end();
                if ((a != a_end) && ((a_end - 1)->input() == Fsa::Epsilon) && ((a_end - 1)->output() == Fsa::Epsilon)) {
                    if (epsSr->isFinal()) {
                        sp->addTags(Fsa::StateTagFinal);
                        finalWeight = semiring_->collect(
                            finalWeight, semiring_->extend(score, epsSr->weight()));
                    }
                    for (; a != a_end; ++a) {
                        verify((a->input() == Fsa::Epsilon) && (a->output() == Fsa::Epsilon));
                        if (!epsClosureScores_[a->target()]) {
                            epsClosureScores_[a->target()] = semiring_->extend(score, a->weight());
                            epsQ.insert(a->target());
                        } else
                            epsClosureScores_[a->target()] =
                                semiring_->collect(
                                    epsClosureScores_[a->target()], semiring_->extend(
                                        score, a->weight()));
                    }
                } else
                    sp->newArc(epsSid, score, Fsa::Epsilon, Fsa::Epsilon);
            }
            if (sp->isFinal())
                sp->setWeight(finalWeight);
            return ConstStateRef(sp);
        }
        virtual std::string describe() const {
            return Core::form("normalize-epsilon-closure(%s)", fsa_->describe().c_str());
        }
    };

    ConstLatticeRef normalizeEpsClosure(ConstLatticeRef l) {
        verify_(l->hasProperty(Fsa::PropertyAcyclic));
        return ConstLatticeRef(new NormalizeEpsilonClosureLattice(l));
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NonWordClosureNormalizationFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = applyOneToOneLabelMap(
                l, LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet())));
            return normalizeEpsClosure(l);
        }
    public:
        NonWordClosureNormalizationFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config) {}
        ~NonWordClosureNormalizationFilterNode() {}
    };
    NodeRef createNonWordClosureNormalizationFilterNode(const std::string &name, const Core::Configuration &config) {
        return NodeRef(new NonWordClosureNormalizationFilterNode(name, config));
    }
    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------
    class NonWordClosureRemovalFilterNode : public FilterNode {
    protected:
        virtual ConstLatticeRef filter(ConstLatticeRef l) {
            if (!l)
                return ConstLatticeRef();
            if (l->type() != Fsa::TypeAcceptor) {
                warning("%s: \"%s\" is a transducer, but result will be an acceptor, i.e. output will be lost.",
                        name.c_str(), l->describe().c_str());
                l = projectInput(l);
            }
            l = applyOneToOneLabelMap(
                l, LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet())));
            return fastRemoveEpsilons(l);
        }
    public:
        NonWordClosureRemovalFilterNode(const std::string &name, const Core::Configuration &config) :
            FilterNode(name, config) {}
        ~NonWordClosureRemovalFilterNode() {}
    };
    NodeRef createNonWordClosureRemovalFilterNode(const std::string &name, const Core::Configuration &config) {
        return NodeRef(new NonWordClosureRemovalFilterNode(name, config));
    }

} // namespace Flf
