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
#include <Fsa/Cache.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Rational.hh>
#include <Fsa/Sort.hh>
#include <Fsa/Static.hh>
#include <Lattice/Lattice.hh>
#include <Lm/FsaLm.hh>
#include <Lm/Module.hh>
#include <Mm/Feature.hh>
#include <Search/Search.hh>
#include <Speech/DelayedRecognizer.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Module.hh>

#include "Best.hh"
#include "Convert.hh"
#include "Copy.hh"
#include "EpsilonRemoval.hh"
#include "FlfCore/Basic.hh"
#include "FlfCore/Utility.hh"
#include "FwdBwd.hh"
#include "Info.hh"
#include "LatticeHandler.hh"
#include "Lexicon.hh"
#include "Map.hh"
#include "Module.hh"
#include "NonWordFilter.hh"
#include "Prune.hh"
#include "RescoreLm.hh"
#include "SegmentwiseSpeechProcessor.hh"
#include "TimeframeConfusionNetworkBuilder.hh"
#include "Union.hh"

// On very long segments a large tolerance between the forward and backward result may be required
#define NUMERICAL_TOLERANE 0.5
#define JOIN_RANGES_TOLERANCE 5

namespace Flf {
struct ForwardBackwardAlignment {
    struct Word {
        Word()
                : amScore(Core::Type<Score>::max),
                  lmScore(Core::Type<Score>::max) {}
        const Bliss::LemmaPronunciation* pron;
        u32                              start, end;
        Fsa::StateId                     originState;
        Score                            amScore, lmScore;

        bool intersects(const Word& rhs) const {
            return end >= rhs.start && start <= rhs.end && rhs.end >= start && rhs.start <= end;
        }

        bool equals(const Word& rhs) const {
            return pron == rhs.pron && end == rhs.end && start == rhs.start;
        }
    };

    ForwardBackwardAlignment(u32                                                _segmentLength,
                             Flf::ConstLatticeRef                               _forward,
                             Flf::ConstLatticeRef                               _backward,
                             u32                                                _lmContextLength,
                             Core::Ref<const Bliss::LemmaPronunciationAlphabet> _lpAlphabet,
                             bool                                               _forceEqualAlignment,
                             bool                                               _correctForceEqualScore,
                             bool                                               _includeNoise,
                             bool                                               _verboseRefinement)
            : segmentLength_(_segmentLength),
              minimumGoodLength_(std::max(_lmContextLength + 1, 2u)),  // At least 2, because otherwise the re-recognized ranges would overlap. +1 because always re-recognize 1 of the good words, and need full context.
              forward_(_forward),
              backward_(_backward),
              forwardScore_(0),
              backwardScore_(0),
              lpAlphabet_(_lpAlphabet),
              forceEqualAlignment_(_forceEqualAlignment),
              correctForceEqualScore_(_correctForceEqualScore),
              includeNoise_(_includeNoise),
              verboseRefinement_(_verboseRefinement),
              errorRate_(0) {
        initialize();
        align();
        normalize();
        select();
    }

    std::pair<std::deque<Word>, u32> parseForwardLattice(Flf::ConstLatticeRef lattice) {
        std::deque<Word> ret;
        u32              length = Core::Type<u32>::max;
        Fsa::StateId     state  = lattice->initialStateId();
        forwardScore_           = 0;
        while (state != Fsa::InvalidStateId) {
            Ftl::Automaton<Semiring, State>::ConstStateRef s = lattice->getState(state);
            if (s->nArcs()) {
                verify(s->nArcs() == 1);  // Lattice-handling not supported ATM

                Word word;
                word.amScore = s->getArc(0)->score(lattice->semiring()->id("am")) * lattice->semiring()->scale(lattice->semiring()->id("am"));
                word.lmScore = s->getArc(0)->score(lattice->semiring()->id("lm")) * lattice->semiring()->scale(lattice->semiring()->id("lm"));

                const Bliss::LemmaPronunciation* pron = lpAlphabet_->lemmaPronunciation(s->getArc(0)->output());
                if (verboseRefinement_) {
                    if (pron)
                        std::cout << "forward pron " << pron->lemma()->name() << " score " << word.amScore << " + " << word.lmScore << " = " << word.amScore + word.lmScore << std::endl;
                    else
                        std::cout << "forward anon score " << word.amScore << " + " << word.lmScore << " = " << word.amScore + word.lmScore << std::endl;
                }
                forwardScore_ += word.amScore + word.lmScore;
                if (pron &&
                    (includeNoise_ || isRealWord(pron))) {
                    word.pron  = pron;
                    word.start = lattice->boundary(state).time();
                    word.end   = lattice->boundary(s->getArc(0)->target()).time();
                    if (word.end > word.start)
                        word.end -= 1;
                    word.originState = state;
                    ret.push_back(word);
                }
                state  = s->getArc(0)->target();
                length = lattice->boundary(state).time();
            }
            else {
                state = Fsa::InvalidStateId;
            }
        }
        if (verboseRefinement_)
            std::cout << "forward total score " << forwardScore_ << std::endl;
        verify(length != Core::Type<u32>::max);
        if (length > 0)
            length -= 1;  // The sentence-end is always hypothesized one frame after the last hypothesis. That additional frame doesn't count.
        return std::make_pair(ret, length);
    }

    bool isRealWord(const Bliss::LemmaPronunciation* pron) const {
        return pron->lemma()->evaluationTokenSequences().first != pron->lemma()->evaluationTokenSequences().second &&
               pron->lemma()->evaluationTokenSequences().first->size();
    }

    std::deque<Word> parseBackwardLattice(Flf::ConstLatticeRef lattice) {
        std::deque<Word> ret;
        Fsa::StateId     state = lattice->initialStateId();
        backwardScore_         = 0;
        while (state != Fsa::InvalidStateId) {
            Ftl::Automaton<Semiring, State>::ConstStateRef s = lattice->getState(state);
            if (s->nArcs()) {
                verify(s->nArcs() == 1);  // Lattice-handling not supported ATM

                Word word;
                word.amScore = s->getArc(0)->score(lattice->semiring()->id("am")) * lattice->semiring()->scale(lattice->semiring()->id("am"));
                word.lmScore = s->getArc(0)->score(lattice->semiring()->id("lm")) * lattice->semiring()->scale(lattice->semiring()->id("lm"));

                const Bliss::LemmaPronunciation* pron = lpAlphabet_->lemmaPronunciation(s->getArc(0)->output());
                if (pron && verboseRefinement_)
                    std::cout << "backward pron " << pron->lemma()->name() << " score " << word.amScore + word.lmScore << std::endl;

                backwardScore_ += word.amScore + word.lmScore;

                if (pron &&
                    (includeNoise_ || isRealWord(pron))) {
                    word.pron = pron;
                    word.end  = segmentLength_ - 1 - lattice->boundary(state).time();
                    s32 start = ((s32)segmentLength_) - 1 - lattice->boundary(s->getArc(0)->target()).time();
                    if (start < (s32)word.end)
                        start += 1;
                    word.start       = std::max(0, start);
                    word.originState = state;
                    ret.push_front(word);
                }
                state = s->getArc(0)->target();
            }
            else {
                state = Fsa::InvalidStateId;
            }
        }
        if (verboseRefinement_)
            std::cout << "backward total score " << backwardScore_ << std::endl;
        return ret;
    }

    void initialize() {
        verify(forward_ && backward_);
        u32 forwardLatticeLength;
        Core::tie(forWords, forwardLatticeLength) = parseForwardLattice(forward_);
        backWords                                 = parseBackwardLattice(backward_);
        if (forwardLatticeLength != segmentLength_)
            std::cout << "WARNING: Alignment length does not match the feature count!" << std::endl;
    }

    void invalidateForWord(s32 word) {
        verify(forWordAlignment.count(word));
        badForWords.insert(word);
        badBackWords.insert(forWordAlignment[word]);
        backWordAlignment.erase(forWordAlignment[word]);
        forWordAlignment.erase(word);
    }

    void invalidateBackWord(s32 word) {
        verify(backWordAlignment.count(word));
        badBackWords.insert(word);
        badForWords.insert(backWordAlignment[word]);
        forWordAlignment.erase(backWordAlignment[word]);
        backWordAlignment.erase(word);
    }

    f32 errorRate() {
        return errorRate_;
    }

    f32 updateRate() {
        return (forWords.size() + backWords.size() - 2 * forWordAlignment.size()) / (f32)(forWords.size() + backWords.size());
    }

    void align() {
        // Align forward/backward result
        {
            s32 currentForWord = 0, currentBackWord = 0;
            while (currentForWord < (s32)forWords.size() && currentBackWord < (s32)backWords.size()) {
                const Word& backWord(backWords[currentBackWord]);
                const Word& forWord(forWords[currentForWord]);
                if (backWord.pron == forWord.pron &&
                    backWord.intersects(forWord) &&
                    (!forceEqualAlignment_ || backWord.equals(forWord)) &&
                    (!correctForceEqualScore_ || std::abs(backWord.amScore - forWord.amScore) < 0.2 ||
                     ((backWord.lmScore == 0 || forWord.lmScore == 0) && backWord.lmScore != forWord.lmScore)))  // ignore AM score if the LM score was 'overflown' into the AM
                {
                    forWordAlignment.insert(std::make_pair(currentForWord, currentBackWord));
                    backWordAlignment.insert(std::make_pair(currentBackWord, currentForWord));
                    ++currentForWord;
                    ++currentBackWord;
                }
                else {
                    if (backWord.start < forWord.start) {
                        badBackWords.insert(currentBackWord);
                        ++currentBackWord;
                    }
                    else {
                        badForWords.insert(currentForWord);
                        ++currentForWord;
                    }
                }
            }

            while (currentForWord < (s32)forWords.size()) {
                badForWords.insert(currentForWord);
                ++currentForWord;
            }

            while (currentBackWord < (s32)backWords.size()) {
                badBackWords.insert(currentBackWord);
                ++currentBackWord;
            }
        }
        errorRate_ = (forWords.size() + backWords.size() - 2 * forWordAlignment.size()) / (f32)(forWords.size() + backWords.size());
    }

    void normalize() {
        bool changed = true;
        // Grow ranges for refinement: Extend the badness ranges where the boundary time of surrounding words is unequal
        while (changed) {
            changed = false;
            for (std::set<s32>::const_iterator it = badForWords.begin(); it != badForWords.end(); ++it) {
                if (forWordAlignment.count(*it - 1)) {
                    if (forWords[*it - 1].start != backWords[forWordAlignment[*it - 1]].start || !isRealWord(forWords[*it - 1].pron)) {
                        invalidateForWord(*it - 1);
                        changed = true;
                        break;
                    }
                }
                if (forWordAlignment.count(*it + 1)) {
                    if (forWords[*it + 1].end != backWords[forWordAlignment[*it + 1]].end || !isRealWord(forWords[*it + 1].pron)) {
                        invalidateForWord(*it + 1);
                        changed = true;
                        break;
                    }
                }
            }
            for (std::set<s32>::const_iterator it = badBackWords.begin(); it != badBackWords.end(); ++it) {
                if (backWordAlignment.count(*it - 1)) {
                    if (backWords[*it - 1].start != forWords[backWordAlignment[*it - 1]].start || !isRealWord(backWords[*it - 1].pron)) {
                        invalidateBackWord(*it - 1);
                        changed = true;
                        break;
                    }
                }
                if (backWordAlignment.count(*it + 1)) {
                    if (backWords[*it + 1].end != forWords[backWordAlignment[*it + 1]].end || !isRealWord(backWords[*it + 1].pron)) {
                        invalidateBackWord(*it + 1);
                        changed = true;
                        break;
                    }
                }
            }
            // Remove intermediate alignment sequences which are shorter than minimumGoodLength, because the n-gram requires this (forwards)
            {
                s32 sequenceStart = -1;
                for (s32 forWord = 0; forWord < (s32)forWords.size(); ++forWord) {
                    if (forWordAlignment.count(forWord)) {
                        if (sequenceStart == -1)
                            sequenceStart = forWord;
                    }
                    else {
                        // This intentionally doesn't trigger on a sequence which goes until the end or which starts at the beginning
                        if (sequenceStart > 0) {
                            u32 realWords = 0;
                            for (s32 q = sequenceStart; q < forWord; ++q)
                                if (isRealWord(forWords[q].pron))
                                    ++realWords;
                            if (realWords < minimumGoodLength_) {
                                // The previous sequence is not the initial sequence, and not the last sequence. But it is too short.
                                for (s32 q = sequenceStart; q < forWord; ++q)
                                    invalidateForWord(q);
                                changed = true;
                            }
                        }
                        sequenceStart = -1;
                    }
                }
            }
            {
                // Remove intermediate alignment sequences which are shorter than minimumGoodLength (backwards)
                s32 sequenceStart = -1;
                for (s32 backWord = 0; backWord < (s32)backWords.size(); ++backWord) {
                    if (backWordAlignment.count(backWord)) {
                        if (sequenceStart == -1)
                            sequenceStart = backWord;
                    }
                    else {
                        // This intentionally doesn't trigger a sequence which goes until the end
                        if (sequenceStart > 0 && backWord - sequenceStart < (s32)minimumGoodLength_) {
                            // The previous sequence is not the initial sequence, and not the last sequence. But it is too short.
                            for (s32 q = sequenceStart; q < backWord; ++q)
                                invalidateBackWord(q);
                            changed = true;
                        }
                        sequenceStart = -1;
                    }
                }
            }
        }
    }

    struct Range {
        Range()
                : startTime(-1),
                  endTime(-1),
                  firstForWord(-1),
                  lastForWord(-1),
                  firstBackWord(-1),
                  lastBackWord(-1),
                  prePhon(Bliss::Phoneme::term),
                  sufPhon(Bliss::Phoneme::term) {
        }
        bool operator<(const Range& rhs) const {
            return startTime < rhs.startTime || (startTime == rhs.startTime && endTime < rhs.endTime);
        }
        bool operator==(const Range& rhs) const {
            return startTime == rhs.startTime && endTime == rhs.endTime;
        }
        s32                              startTime, endTime;
        s32                              firstForWord, lastForWord;                         // First / last for-word which is _replaced_ by this range. -1 means boundary.
        s32                              firstBackWord, lastBackWord;                       // First / last back-word which is _replaced_ by this range -1 means boundary.
        Flf::Boundary::Transit           coarticulation, backwardCoarticulation;            // backwardCoarticulation can be used as-is in the backward decoder, it is ordered correctly.
        Flf::Boundary::Transit           finalCoarticulation, finalBackwardCoarticulation;  // backwardCoarticulation can be used as-is in the backward decoder, it is ordered correctly.
        std::vector<const Bliss::Lemma*> prefix, suffix;
        Bliss::Phoneme::Id               prePhon, sufPhon;
    };

    std::set<Range> select() {
        std::set<Range> ret;
        Range           currentRange;
        for (std::set<s32>::const_iterator it = badForWords.begin(); it != badForWords.end(); ++it) {
            if (currentRange.startTime == -1) {
                if (*it == 0) {
                    currentRange.startTime = 0;
                }
                else {
                    currentRange.firstForWord = *it - 1;
                    verify(currentRange.firstForWord >= 0 && currentRange.firstForWord < (s32)forWords.size());
                    currentRange.startTime = forWords[currentRange.firstForWord].start;
                    verify(forWordAlignment.count(currentRange.firstForWord));
                    currentRange.coarticulation = forward_->boundary(forWords[currentRange.firstForWord].originState).transit();
                    verify(forWordAlignment.count(currentRange.firstForWord));
                    currentRange.firstBackWord = forWordAlignment[currentRange.firstForWord];
                    verify(currentRange.firstBackWord >= 0 && currentRange.firstBackWord < (s32)backWords.size());
                    verify((s32)backWords[currentRange.firstBackWord].start == currentRange.startTime);
                    verify(backward_->getState(backWords[currentRange.firstBackWord].originState)->nArcs() == 1);
                    currentRange.finalBackwardCoarticulation = backward_->boundary(backward_->getState(backWords[currentRange.firstBackWord].originState)->getArc(0)->target()).transit();

                    for (s32 q = 0; q < currentRange.firstForWord; ++q)
                        currentRange.prefix.push_back(forWords[q].pron->lemma());

                    if (currentRange.firstForWord > 0) {
                        const Bliss::LemmaPronunciation* pron = forWords[currentRange.firstForWord - 1].pron;
                        if (pron && pron->pronunciation()->length())
                            currentRange.prePhon = pron->pronunciation()->operator[](pron->pronunciation()->length() - 1);
                    }
                }
            }
            if (!badForWords.count(*it + 1)) {
                if (*it == forWords.size() - 1) {
                    currentRange.endTime = segmentLength_ - 1;
                }
                else {
                    currentRange.lastForWord = *it + 1;
                    verify(currentRange.lastForWord >= 0 && currentRange.lastForWord < forWords.size());
                    currentRange.endTime      = forWords[currentRange.lastForWord].end;
                    currentRange.lastBackWord = forWordAlignment[currentRange.lastForWord];
                    verify(currentRange.lastBackWord >= 0 && currentRange.lastBackWord < backWords.size());
                    verify(currentRange.endTime == backWords[currentRange.lastBackWord].end);
                    currentRange.backwardCoarticulation = backward_->boundary(backWords[currentRange.lastBackWord].originState).transit();
                    verify(forward_->getState(forWords[currentRange.lastForWord].originState)->nArcs() == 1);
                    currentRange.finalCoarticulation = forward_->boundary(forward_->getState(forWords[currentRange.lastForWord].originState)->getArc(0)->target()).transit();
                    for (u32 q = currentRange.lastForWord + 1; q < forWords.size(); ++q)
                        currentRange.suffix.push_back(forWords[q].pron->lemma());
                    if (currentRange.lastForWord + 1 < forWords.size()) {
                        const Bliss::LemmaPronunciation* pron = forWords[currentRange.lastForWord + 1].pron;
                        if (pron && pron->pronunciation()->length())
                            currentRange.sufPhon = pron->pronunciation()->operator[](0);
                    }
                }

                if (currentRange.firstForWord == 0) {
                    // Lift unneeded constraints
                    currentRange.startTime    = 0;
                    currentRange.firstForWord = -1;
                }

                if (currentRange.lastForWord == forWords.size() - 1) {
                    // Lift unneeded constraints
                    currentRange.endTime     = segmentLength_ - 1;
                    currentRange.lastForWord = -1;
                }

                ret.insert(currentRange);
                currentRange = Range();
            }
        }
        for (std::set<s32>::const_iterator it = badBackWords.begin(); it != badBackWords.end(); ++it) {
            if (currentRange.startTime == -1) {
                if (*it == 0) {
                    currentRange.startTime = 0;
                }
                else {
                    currentRange.firstBackWord = *it - 1;
                    verify(currentRange.firstBackWord >= 0 && currentRange.firstBackWord < backWords.size());
                    currentRange.startTime = backWords[currentRange.firstBackWord].start;
                    verify(backWordAlignment.count(currentRange.firstBackWord));
                    currentRange.firstForWord = backWordAlignment[currentRange.firstBackWord];
                    verify(currentRange.firstForWord >= 0 && currentRange.firstForWord < forWords.size());
                    currentRange.coarticulation = forward_->boundary(forWords[currentRange.firstForWord].originState).transit();
                    verify(backward_->getState(backWords[currentRange.firstBackWord].originState)->nArcs() == 1);
                    currentRange.finalBackwardCoarticulation = backward_->boundary(backward_->getState(backWords[currentRange.firstBackWord].originState)->getArc(0)->target()).transit();
                    verify(forWords[currentRange.firstForWord].start == currentRange.startTime);

                    for (s32 q = 0; q < currentRange.firstBackWord; ++q)
                        currentRange.prefix.push_back(backWords[q].pron->lemma());
                    if (currentRange.firstBackWord > 0) {
                        const Bliss::LemmaPronunciation* pron = backWords[currentRange.firstBackWord - 1].pron;
                        if (pron && pron->pronunciation()->length())
                            currentRange.prePhon = pron->pronunciation()->operator[](pron->pronunciation()->length() - 1);
                    }
                }
            }
            if (!badBackWords.count(*it + 1)) {
                if (*it == backWords.size() - 1) {
                    currentRange.endTime = segmentLength_ - 1;
                }
                else {
                    currentRange.lastBackWord = *it + 1;
                    currentRange.endTime      = backWords[currentRange.lastBackWord].end;
                    verify(currentRange.lastBackWord >= 0 && currentRange.lastBackWord < backWords.size());
                    currentRange.lastForWord = backWordAlignment[*it + 1];
                    verify(currentRange.lastForWord >= 0 && currentRange.lastForWord < forWords.size());
                    verify(currentRange.endTime == forWords[currentRange.lastForWord].end);
                    currentRange.backwardCoarticulation = backward_->boundary(backWords[currentRange.lastBackWord].originState).transit();
                    verify(forward_->getState(forWords[currentRange.lastForWord].originState)->nArcs() == 1);
                    currentRange.finalCoarticulation = forward_->boundary(forward_->getState(forWords[currentRange.lastForWord].originState)->getArc(0)->target()).transit();
                    for (u32 q = currentRange.lastBackWord + 1; q < backWords.size(); ++q)
                        currentRange.suffix.push_back(backWords[q].pron->lemma());
                    if (currentRange.lastBackWord + 1 < backWords.size()) {
                        const Bliss::LemmaPronunciation* pron = backWords[currentRange.lastBackWord + 1].pron;
                        if (pron && pron->pronunciation()->length())
                            currentRange.sufPhon = pron->pronunciation()->operator[](0);
                    }
                }

                if (currentRange.firstForWord == 0) {
                    // Lift unneeded constraints
                    currentRange.startTime    = 0;
                    currentRange.firstForWord = -1;
                }

                if (currentRange.lastForWord == forWords.size() - 1) {
                    // Lift unneeded constraints
                    currentRange.endTime     = segmentLength_ - 1;
                    currentRange.lastForWord = -1;
                }

                ret.insert(currentRange);
                currentRange = Range();
            }
        }

        // Join ranges which are very close to each other
        std::vector<Range> jointRanges;
        for (std::set<Range>::const_iterator rangeIt = ret.begin(); rangeIt != ret.end(); ++rangeIt) {
            if (!jointRanges.empty()) {
                Range& previousRange(jointRanges.back());
                if (previousRange.endTime >= rangeIt->endTime)
                    continue;  // Just swallow this range, it is completeley contained in the previous

                if (previousRange.endTime + JOIN_RANGES_TOLERANCE > rangeIt->startTime) {
                    previousRange.endTime                = rangeIt->endTime;
                    previousRange.lastBackWord           = rangeIt->lastBackWord;
                    previousRange.lastForWord            = rangeIt->lastForWord;
                    previousRange.suffix                 = rangeIt->suffix;
                    previousRange.backwardCoarticulation = rangeIt->backwardCoarticulation;
                    previousRange.finalCoarticulation    = rangeIt->finalCoarticulation;
                }
                else {
                    jointRanges.push_back(*rangeIt);
                }
            }
            else {
                jointRanges.push_back(*rangeIt);
            }
        }

        return std::set<Range>(jointRanges.begin(), jointRanges.end());
    }

    u32                                                segmentLength_, minimumGoodLength_;
    std::deque<Word>                                   forWords, backWords;
    Flf::ConstLatticeRef                               forward_, backward_;
    Score                                              forwardScore_, backwardScore_;
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lpAlphabet_;
    bool                                               forceEqualAlignment_, correctForceEqualScore_, includeNoise_, verboseRefinement_;
    f32                                                errorRate_;
    std::map<s32, s32>                                 forWordAlignment, backWordAlignment;
    std::set<s32>                                      badForWords, badBackWords;
};

// -------------------------------------------------------------------------
class IncrementalRecognizer : public Speech::Recognizer {
public:
    static const Core::ParameterBool  paramMeshRescoring;
    static const Core::ParameterBool  paramExpandTransits;
    static const Core::ParameterBool  paramMeshCombination;
    static const Core::ParameterBool  paramForceForwardBackwardLattices;
    static const Core::ParameterInt   paramRescoreWordEndLimit;
    static const Core::ParameterBool  paramPronunciationScore;
    static const Core::ParameterBool  paramCorrectWholeSegment;
    static const Core::ParameterBool  paramCorrectForceEqualAlignment;
    static const Core::ParameterBool  paramCorrectForceEqualScore;
    static const Core::ParameterBool  paramCorrectStrictInitial;
    static const Core::ParameterBool  paramCorrectIncludeNoise;
    static const Core::ParameterBool  paramConfidenceScore;
    static const Core::ParameterFloat paramAlpha;
    static const Core::ParameterBool  paramApplyNonWordClosureFilter;
    static const Core::ParameterBool  paramApplyUniqueSentenceAlignmentFilter;
    static const Core::ParameterFloat paramPosteriorPruningThreshold;
    static const Core::ParameterBool  paramVerboseRefinement;
    static const Core::ParameterBool  paramConsiderSentenceBegin;
    static const Core::ParameterFloat paramScoreTolerance;
    static const Core::ParameterBool  paramOnlyEnforceMinimumSearchSpace;
    static const Core::ParameterFloat paramMaxRtf;
    static const Core::ParameterInt   paramCacheFrames;
    static const Core::ParameterBool  paramPreCache;
    static const Core::ParameterInt   paramLmContextLength;
    static const Core::ParameterFloat paramRelaxPruningFactor;
    static const Core::ParameterFloat paramRelaxPruningOffset;
    static const Core::ParameterFloat paramLatticeRelaxPruningFactor;
    static const Core::ParameterFloat paramLatticeRelaxPruningOffset;
    static const Core::ParameterFloat paramAdaptInitialUpdateRate;
    static const Core::ParameterFloat paramAdaptRelaxPruningFactor;
    static const Core::ParameterFloat paramAdaptRelaxPruningOffset;
    static const Core::ParameterInt   paramLatticeRelaxPruningInterval;
    static const Core::ParameterInt   paramAdaptCorrectionRatio;
    static const Core::ParameterFloat paramAdaptPruningFactor;
    static const Core::ParameterFloat paramMinArcsPerSecond;
    static const Core::ParameterFloat paramMaxArcsPerSecond;
    static const Core::ParameterInt   paramMaxLatticeRegenerations;

private:
    std::unique_ptr<Speech::RecognizerDelayHandler> delayedRecognition_;
    Core::Ref<Mm::ContextScorerCache>               contextScorerCache_;
    ModelCombinationRef                             mc_;
    SegmentwiseFeatureExtractorRef                  featureExtractor_;
    SegmentwiseModelAdaptorRef                      modelAdaptor_;
    Core::XmlChannel                                tracebackChannel_;
    Search::Traceback                               traceback_;
    std::vector<Flow::Timestamp>                    featureTimes_;
    std::unique_ptr<IncrementalRecognizer>          backwardRecognizer_;

    Core::Timer globalTimer_;
    u32         segmentFeatureCount_;

    f32 globalRtf() const {
        if (segmentFeatureCount_ == 0)
            return 0;
        else
            return globalTimer_.user() / (segmentFeatureCount_ / 100.0);
    }

    bool             meshRescoring_;
    bool             expandTransits_;
    bool             meshCombination_;
    bool             forceForwardBackwardLattices_;
    bool             addPronunciationScores_;
    bool             correctWholeSegment_;
    bool             correctForceEqualAlignment_;
    bool             correctForceEqualScore_;
    bool             correctIncludeNoise_;
    bool             addConfidenceScores_;
    bool             applyNonWordClosureFilter_;
    bool             applyUniqueSentenceAlignmentFilter_;
    std::deque<bool> secondOrderCorrectionHistory_;

    int rescoreWordEndLimit_;

    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lpAlphabet_;
    Fsa::LabelId                                       sentenceEndLabel_;
    LabelMapRef                                        nonWordToEpsilonMap_;
    Score                                              pronScale_, lmScale_;
    ScoreId                                            amId_, pronunciationId_, lmId_, confidenceId_;
    ConstSemiringRef                                   semiring_, posteriorSemiring_;
    Score                                              fwdBwdThreshold_;
    u32                                                lmContextLength_;

    f32  relaxPruningFactor_, relaxPruningOffset_, latticeRelaxPruningFactor_, latticeRelaxPruningOffset_, adaptInitialUpdateRate_, adaptRelaxPruningFactor_, adaptRelaxPruningOffset_;
    u32  latticeRelaxPruningInterval_;
    s32  adaptCorrectionRatio_;
    f32  scoreTolerance_;
    f32  adaptPruningFactor_;
    f32  minArcsPerSecond_;
    f32  maxArcsPerSecond_;
    s32  maxLatticeRegenerations_;
    bool onlyEnforceMinimumSearchSpace_;
    bool correctStrictInitial_;
    f32  maximumRtf_;

    const Bliss::SpeechSegment* segment_;
    // Current sub-segment index, if partial lattices were returned by the decoder
    u32           subSegment_;
    bool          verboseRefinement_;
    bool          considerSentenceBegin_;
    bool          preCacheAllFrames_;
    DataSourceRef dataSource_;

protected:
    void addPartialToTraceback(Search::Traceback& partialTraceback) {
        if (!traceback_.empty() && traceback_.back().time == partialTraceback.front().time)
            partialTraceback.erase(partialTraceback.begin());
        traceback_.insert(traceback_.end(), partialTraceback.begin(), partialTraceback.end());
    }

    void processResult() {
        Search::Traceback remainingTraceback;
        recognizer_->getCurrentBestSentence(remainingTraceback);
        addPartialToTraceback(remainingTraceback);

        Core::XmlWriter& os(clog());
        os << Core::XmlOpen("traceback");
        traceback_.write(os, lexicon_->phonemeInventory());
        os << Core::XmlClose("traceback");
        os << Core::XmlOpen("orth") + Core::XmlAttribute("source", "recognized");
        for (u32 i = 0; i < traceback_.size(); ++i)
            if (traceback_[i].pronunciation)
                os << traceback_[i].pronunciation->lemma()->preferredOrthographicForm()
                   << Core::XmlBlank();
        os << Core::XmlClose("orth");
        if (tracebackChannel_.isOpen()) {
            logTraceback(traceback_);
            featureTimes_.clear();
        }
    }

    /*
     * In case of a valid label id:
     * Before:
     * am_score = emission_scale * emission + transition_scale * transition
     * am_scale = 1.0
     * lm_score = pronunciation_scale * pronunciation + lm_scale * lm
     * lm_scale = 1.0
     * Afterwards:
     * am_score = emission_scale * emission + transition_scale * transition + pronunciation_scale * pronunciation
     * am_scale = 1.0
     * lm_score = lm
     * lm_scale = lm_scale
     */
    ScoresRef buildScore(Fsa::LabelId label, Score amRecogScore, Score lmRecogScore) {
        verify(amRecogScore != Semiring::Zero);
        verify(lmRecogScore != Semiring::Zero);

        Score amScore, pronScore, lmScore;
        if ((Fsa::FirstLabelId <= label) && (label <= Fsa::LastLabelId)) {
            amScore   = amRecogScore;
            pronScore = lpAlphabet_->lemmaPronunciation(label)->pronunciationScore();
            verify(pronScore != Semiring::Zero);
            lmScore = (lmRecogScore - pronScale_ * pronScore) / lmScale_;
        }
        else {
            amScore   = amRecogScore;
            pronScore = Semiring::One;
            lmScore   = lmRecogScore / lmScale_;
        }

        ScoresRef scores = semiring_->create();
        if (addPronunciationScores_) {
            scores->set(amId_, amScore);
            scores->set(pronunciationId_, pronScore);
        }
        else
            scores->set(amId_, amScore + pronScale_ * pronScore);
        scores->set(lmId_, lmScore);
        if (addConfidenceScores_)
            scores->set(confidenceId_, 0.0);
        verify(semiring_->project(scores) != Semiring::Zero);

        return scores;
    }

    enum Extension {
        ExtendNone  = 0,
        ExtendLeft  = 1,
        ExtendRight = 2,
        ExtendBoth  = 3
    };

    std::pair<Flf::ConstLatticeRef, Flf::ConstLatticeRef> recognizeFeatures(const std::deque<FeatureRef>& features) {
        delayedRecognition_->reset();
        recognizer_->restart();
        for (std::deque<FeatureRef>::const_iterator it = features.begin(); it != features.end(); ++it)
            delayedRecognition_->add(*it);
        while (delayedRecognition_->flush())
            ;
        Core::Ref<const Search::LatticeAdaptor> recognizerLattice = recognizer_->getCurrentWordLattice();
        Flf::ConstLatticeRef                    ret               = buildLattice(recognizerLattice, true);
        Flf::ConstLatticeRef                    meshLattice;
        if (meshRescoring_) {
            Search::SearchAlgorithm::RecognitionContext c = recognizer_->setContext(Search::SearchAlgorithm::RecognitionContext());
            recognizer_->setContext(c);
            meshLattice = mesh(ret);
            if (meshCombination_)
                ret = best(ret, BellmanFord);
            else
                ret = best(decodeRescoreLm(meshLattice,
                                           mc_->languageModel(),
                                           recognizer_->describePruning()->masterBeam(),
                                           rescoreWordEndLimit_, c.prefix, c.suffix),
                           BellmanFord);
            if (expandTransits_)
                meshLattice = expandTransits(meshLattice, c.prePhon, c.sufPhon);
        }
        return std::make_pair(ret, meshLattice);
    }

    struct RecognizedSequence {
        RecognizedSequence()
                : extension(Flf::IncrementalRecognizer::ExtendNone) {}
        Extension                           extension;
        Flf::ConstLatticeRef                singleBestLattice;
        Search::SearchAlgorithm::PruningRef pruning;
        std::vector<MeshEntry>              meshEntries;
    };

    RecognizedSequence refine(int                                     offset,
                              const std::deque<FeatureRef>&           features,
                              const Bliss::LemmaPronunciation*        forceFirstWord,
                              const Bliss::LemmaPronunciation*        forceLastWord,
                              const std::vector<const Bliss::Lemma*>& lmPrefix,
                              const std::vector<const Bliss::Lemma*>& lmSuffix,
                              bool                                    isInitial = false,
                              bool                                    isSecond  = false) {
        RecognizedSequence ret;
        ret.pruning = recognizer_->describePruning()->clone();

        if (features.empty()) {
            ret.singleBestLattice = recognizeFeatures(features).first;
            return ret;
        }

        Core::Timer timer;
        timer.start();

        std::cout << "forward-backward refinement on "
                  << offset
                  << " -> "
                  << offset + features.size()
                  << " (enforce "
                  << (forceFirstWord ? forceFirstWord->lemma()->preferredOrthographicForm().str() : "*")
                  << ":"
                  << (forceLastWord ? forceLastWord->lemma()->preferredOrthographicForm().str() : "*")
                  << ") with pruning "
                  << ret.pruning->format()
                  << std::endl;

        Flf::ConstLatticeRef forward, forwardMesh, backward, backwardMesh;
        Core::tie(forward, forwardMesh)   = recognizeFeatures(features);
        Core::tie(backward, backwardMesh) = backwardRecognizer_->recognizeFeatures(std::deque<FeatureRef>(features.rbegin(), features.rend()));

        if (forwardMesh) {
            MeshEntry e;
            e.lattice = forwardMesh;
            ret.meshEntries.push_back(e);
            e.lattice       = backwardMesh;
            e.reverseOffset = features.size();
            ret.meshEntries.push_back(e);
        }

        timer.stop();
        if (verboseRefinement_) {
            std::cout << "local needed time: "
                      << timer.user()
                      << " for duration: "
                      << (features.size() / 100.0)
                      << " local RTF: "
                      << timer.user() / (features.size() / 100.0)
                      << std::endl;
        }

        ForwardBackwardAlignment alignment(features.size(),
                                           forward, backward,
                                           lmContextLength_,
                                           lpAlphabet_,
                                           correctForceEqualAlignment_,
                                           correctForceEqualScore_ || (correctStrictInitial_ && isInitial),
                                           correctIncludeNoise_ || (correctStrictInitial_ && isInitial),
                                           verboseRefinement_);

        Score completeForwardScore  = 0;  // Complete score of the forward hypothesis, including the complete LM context
        Score completeBackwardScore = 0;  // Complete score of the backward hypothesis, including the complete LM context

        {
            Lm::History basicForwardHistory = mc_->languageModel()->startHistory();
            Score       basicForwardScore   = considerSentenceBegin_ ? mc_->languageModel()->sentenceBeginScore() : 0;

            for (std::vector<const Bliss::Lemma*>::const_iterator it = lmPrefix.begin(); it != lmPrefix.end(); ++it)
                Lm::addLemmaScore(mc_->languageModel(), *it, mc_->languageModel()->scale(), basicForwardHistory, basicForwardScore);

            if (verboseRefinement_)
                std::cout << "basic forward LM score: " << basicForwardScore << "(with " << lmPrefix.size() << " prefix words)" << std::endl;

            Lm::History basicBackwardHistory = backwardRecognizer_->mc_->languageModel()->startHistory();
            Score       basicBackwardScore   = considerSentenceBegin_ ? backwardRecognizer_->mc_->languageModel()->sentenceBeginScore() : 0;

            for (std::vector<const Bliss::Lemma*>::const_reverse_iterator it = lmSuffix.rbegin(); it != lmSuffix.rend(); ++it)
                Lm::addLemmaScore(backwardRecognizer_->mc_->languageModel(),
                                  *it,
                                  backwardRecognizer_->mc_->languageModel()->scale(),
                                  basicBackwardHistory,
                                  basicBackwardScore);

            if (verboseRefinement_)
                std::cout << "basic backward LM score: " << basicBackwardScore << "(with " << lmSuffix.size() << " suffix words)" << std::endl;

            // Step 1: Compute forward score for the forward hypothesis
            Lm::History forwardHypothesisForwardHistory = basicForwardHistory;

            Score forwardHypothesisCentralForwardLMScore = 0;

            for (std::deque<ForwardBackwardAlignment::Word>::const_iterator it = alignment.forWords.begin(); it != alignment.forWords.end(); ++it) {
                Score old = forwardHypothesisCentralForwardLMScore;
                Lm::addLemmaScore(mc_->languageModel(),
                                  it->pron->lemma(),
                                  mc_->languageModel()->scale(),
                                  forwardHypothesisForwardHistory,
                                  forwardHypothesisCentralForwardLMScore);
                if (forwardHypothesisCentralForwardLMScore - old >= 0 && std::abs((forwardHypothesisCentralForwardLMScore - old) - it->lmScore) > NUMERICAL_TOLERANE && verboseRefinement_)
                    std::cout << "WARNING: forward word score "
                              << forwardHypothesisCentralForwardLMScore - old
                              << " and alignment score alignment "
                              << it->lmScore
                              << " differ"
                              << std::endl;

                if (verboseRefinement_)
                    std::cout << "forward "
                              << " (" << it->start << " -> " << it->end << "): "
                              << it->pron->lemma()->preferredOrthographicForm() << " "
                              << it->pron->id() << " [ am " << it->amScore << ", lm " << it->lmScore << " ]"
                              << " actual " << forwardHypothesisCentralForwardLMScore - old << std::endl;
            }

            Score forwardHypothesisForwardScore = basicForwardScore + forwardHypothesisCentralForwardLMScore;

            for (std::vector<const Bliss::Lemma*>::const_iterator it = lmSuffix.begin(); it != lmSuffix.end(); ++it)
                Lm::addLemmaScore(mc_->languageModel(), *it, mc_->languageModel()->scale(), forwardHypothesisForwardHistory, forwardHypothesisForwardScore);

            forwardHypothesisForwardScore += mc_->languageModel()->sentenceEndScore(forwardHypothesisForwardHistory);

            completeForwardScore = (forwardHypothesisForwardScore - forwardHypothesisCentralForwardLMScore) + alignment.forwardScore_;

            // Step 2: Compute backward score for the forward hypothesis

            Lm::History forwardHypothesisBackwardHistory = basicBackwardHistory;
            Score       forwardHypothesisBackwardScore   = basicBackwardScore;

            for (std::deque<ForwardBackwardAlignment::Word>::const_reverse_iterator it = alignment.forWords.rbegin(); it != alignment.forWords.rend(); ++it) {
                Score old = forwardHypothesisBackwardScore;
                Lm::addLemmaScore(backwardRecognizer_->mc_->languageModel(),
                                  it->pron->lemma(),
                                  backwardRecognizer_->mc_->languageModel()->scale(),
                                  forwardHypothesisBackwardHistory,
                                  forwardHypothesisBackwardScore);
                if (verboseRefinement_)
                    std::cout << "backward LM score component for "
                              << it->pron->lemma()->preferredOrthographicForm()
                              << ": "
                              << forwardHypothesisBackwardScore - old
                              << std::endl;
            }

            for (std::vector<const Bliss::Lemma*>::const_reverse_iterator it = lmPrefix.rbegin(); it != lmPrefix.rend(); ++it) {
                Lm::addLemmaScore(backwardRecognizer_->mc_->languageModel(),
                                  *it,
                                  backwardRecognizer_->mc_->languageModel()->scale(),
                                  forwardHypothesisBackwardHistory,
                                  forwardHypothesisBackwardScore);
            }

            forwardHypothesisBackwardScore += backwardRecognizer_->mc_->languageModel()->sentenceEndScore(forwardHypothesisBackwardHistory);

            // Step 3: Compute the forward score for the backward hypothesis

            Score       backwardHypothesisForwardScore   = basicForwardScore;
            Lm::History backwardHypothesisForwardHistory = basicForwardHistory;

            for (std::deque<ForwardBackwardAlignment::Word>::const_iterator it = alignment.backWords.begin(); it != alignment.backWords.end(); ++it) {
                Score old = backwardHypothesisForwardScore;
                Lm::addLemmaScore(mc_->languageModel(),
                                  it->pron->lemma(),
                                  mc_->languageModel()->scale(),
                                  backwardHypothesisForwardHistory,
                                  backwardHypothesisForwardScore);

                if (verboseRefinement_) {
                    std::cout << "forward LM score component for "
                              << it->pron->lemma()->preferredOrthographicForm()
                              << ": "
                              << backwardHypothesisForwardScore - old
                              << std::endl;
                }
            }

            for (std::vector<const Bliss::Lemma*>::const_iterator it = lmSuffix.begin(); it != lmSuffix.end(); ++it)
                Lm::addLemmaScore(mc_->languageModel(), *it, mc_->languageModel()->scale(), backwardHypothesisForwardHistory, backwardHypothesisForwardScore);

            backwardHypothesisForwardScore += mc_->languageModel()->sentenceEndScore(backwardHypothesisForwardHistory);

            // Step 4: Compute the backward score for the backward hypothesis

            Lm::History backwordHypothesisBackwardHistory = basicBackwardHistory;

            Score backwardHypothesisCentralLMScore = 0;
            for (std::deque<ForwardBackwardAlignment::Word>::const_reverse_iterator it = alignment.backWords.rbegin(); it != alignment.backWords.rend(); ++it) {
                Score old = backwardHypothesisCentralLMScore;
                Lm::addLemmaScore(backwardRecognizer_->mc_->languageModel(),
                                  it->pron->lemma(),
                                  backwardRecognizer_->mc_->languageModel()->scale(),
                                  backwordHypothesisBackwardHistory,
                                  backwardHypothesisCentralLMScore);

                if (backwardHypothesisCentralLMScore - old >= 0 && std::abs((backwardHypothesisCentralLMScore - old) - it->lmScore) > NUMERICAL_TOLERANE) {
                    if (verboseRefinement_)
                        std::cout << "WARNING: backward word score "
                                  << (backwardHypothesisCentralLMScore - old)
                                  << " differs from score in alignment: "
                                  << it->lmScore
                                  << std::endl;
                }

                if (verboseRefinement_)
                    std::cout << "backward "
                              << " (" << it->start << " -> " << it->end << "): "
                              << it->pron->lemma()->preferredOrthographicForm() << " "
                              << it->pron->id() << " [ am " << it->amScore << ", lm " << it->lmScore << " ]"
                              << " actual " << backwardHypothesisCentralLMScore - old << std::endl;

                verify(it->start < features.size() && it->end < features.size());
            }

            Score backwardHypothesisBackwardScore = basicBackwardScore + backwardHypothesisCentralLMScore;

            for (std::vector<const Bliss::Lemma*>::const_reverse_iterator it = lmPrefix.rbegin(); it != lmPrefix.rend(); ++it)
                Lm::addLemmaScore(backwardRecognizer_->mc_->languageModel(),
                                  *it,
                                  backwardRecognizer_->mc_->languageModel()->scale(),
                                  backwordHypothesisBackwardHistory,
                                  backwardHypothesisBackwardScore);

            backwardHypothesisBackwardScore += backwardRecognizer_->mc_->languageModel()->sentenceEndScore(backwordHypothesisBackwardHistory);

            completeBackwardScore = (backwardHypothesisBackwardScore - backwardHypothesisCentralLMScore) + alignment.backwardScore_;

            if (verboseRefinement_) {
                std::cout << "forward hypothesis: forward LM score: " << forwardHypothesisForwardScore << " backward LM score: " << forwardHypothesisBackwardScore << std::endl;
                std::cout << "forward hypothesis: forward sentence end LM score " << mc_->languageModel()->sentenceEndScore(forwardHypothesisForwardHistory) << std::endl;
                std::cout << "forward hypothesis: backward sentence end LM score " << backwardRecognizer_->mc_->languageModel()->sentenceEndScore(forwardHypothesisBackwardHistory) << std::endl;
                std::cout << "backward hypothesis: forward LM score: " << backwardHypothesisForwardScore << " backward LM score: " << backwardHypothesisBackwardScore << std::endl;
                std::cout << "backward hypothesis: forward sentence end LM score " << mc_->languageModel()->sentenceEndScore(backwardHypothesisForwardHistory) << std::endl;
                std::cout << "backward hypothesis: backward sentence end LM score " << backwardRecognizer_->mc_->languageModel()->sentenceEndScore(backwordHypothesisBackwardHistory) << std::endl;
                std::cout << "complete forward score: " << completeForwardScore << " complete backward score: " << completeBackwardScore << std::endl;
            }

            bool scoreMismatch = false;

            if (std::abs(forwardHypothesisForwardScore - forwardHypothesisBackwardScore) > NUMERICAL_TOLERANE) {
                std::cout << "WARNING: forward and backward score of forward hypothesis differ: "
                          << forwardHypothesisForwardScore << " vs. " << forwardHypothesisBackwardScore
                          << " difference " << std::abs(forwardHypothesisForwardScore - forwardHypothesisBackwardScore)
                          << std::endl;
                scoreMismatch = true;
            }

            if (std::abs(backwardHypothesisForwardScore - backwardHypothesisBackwardScore) > NUMERICAL_TOLERANE) {
                std::cout << "WARNING: forward and backward score of forward hypothesis differ: "
                          << backwardHypothesisForwardScore << " vs. " << backwardHypothesisBackwardScore
                          << " difference " << std::abs(forwardHypothesisForwardScore - forwardHypothesisBackwardScore)
                          << std::endl;
                scoreMismatch = true;
            }

            if (verboseRefinement_ || scoreMismatch) {
                std::cout << "forward hypothesis: forward LM score: " << forwardHypothesisForwardScore << " backward LM score: " << forwardHypothesisBackwardScore << std::endl;
                std::cout << "forward hypothesis: forward sentence end LM score " << mc_->languageModel()->sentenceEndScore(forwardHypothesisForwardHistory) << std::endl;
                std::cout << "forward hypothesis: backward sentence end LM score " << backwardRecognizer_->mc_->languageModel()->sentenceEndScore(forwardHypothesisBackwardHistory) << std::endl;
                std::cout << "backward hypothesis: forward LM score: " << backwardHypothesisForwardScore << " backward LM score: " << backwardHypothesisBackwardScore << std::endl;
                std::cout << "backward hypothesis: forward sentence end LM score " << mc_->languageModel()->sentenceEndScore(backwardHypothesisForwardHistory) << std::endl;
                std::cout << "backward hypothesis: backward sentence end LM score " << backwardRecognizer_->mc_->languageModel()->sentenceEndScore(backwordHypothesisBackwardHistory) << std::endl;
                std::cout << "complete forward score: " << completeForwardScore << " complete backward score: " << completeBackwardScore << std::endl;
            }
        }

        bool scoreThresholdSatisfied = std::abs(completeForwardScore - completeBackwardScore) / (features.size() * 0.01f) <= scoreTolerance_;

        if (verboseRefinement_)
            std::cout << "complete forward score: " << completeForwardScore << " complete backward score: " << completeBackwardScore << std::endl;

        Search::SearchAlgorithm::PruningRef currentPruning         = recognizer_->describePruning();
        Search::SearchAlgorithm::PruningRef currentBackwardPruning = backwardRecognizer_->recognizer_->describePruning();

        bool searchSpaceOK = currentPruning->checkSearchSpace() && currentBackwardPruning->checkSearchSpace();

        if (searchSpaceOK) {
            if (verboseRefinement_)
                std::cout << "search space is OK" << std::endl;
            if (onlyEnforceMinimumSearchSpace_) {
                ret.singleBestLattice = forward;
                return ret;
            }
        }
        else {
            if (verboseRefinement_)
                std::cout << "search space is BAD" << std::endl;
        }

        if (isInitial) {
            if (verboseRefinement_) {
                std::cout << "initial search error rate: " << alignment.errorRate() << std::endl;
                std::cout << "initial update rate: " << alignment.updateRate() << std::endl;
                if (searchSpaceOK)
                    std::cout << "initial search space is OK" << std::endl;
                else
                    std::cout << "initial search space is BAD" << std::endl;
            }
            if (adaptInitialUpdateRate_ && (alignment.forWords.size() + alignment.backWords.size() > 5 || !searchSpaceOK)) {
                if ((alignment.updateRate() > adaptInitialUpdateRate_ && !scoreThresholdSatisfied) || !searchSpaceOK) {
                    if (verboseRefinement_)
                        std::cout << "permanently relaxing pruning for adaptation to match target update rate " << adaptInitialUpdateRate_ << std::endl;
                    recognizer_->relaxPruning(adaptRelaxPruningFactor_, adaptRelaxPruningOffset_);
                    backwardRecognizer_->recognizer_->relaxPruning(adaptRelaxPruningFactor_, adaptRelaxPruningOffset_);
                }
                else if (alignment.updateRate() < adaptInitialUpdateRate_) {
                    if (verboseRefinement_)
                        std::cout << "permanently tightening pruning for adaptation to match target update rate " << adaptInitialUpdateRate_ << std::endl;
                    recognizer_->relaxPruning(1.0 / adaptRelaxPruningFactor_, -adaptRelaxPruningOffset_);
                    backwardRecognizer_->recognizer_->relaxPruning(1.0 / adaptRelaxPruningFactor_, -adaptRelaxPruningOffset_);
                }
                currentPruning         = recognizer_->describePruning();
                currentBackwardPruning = backwardRecognizer_->recognizer_->describePruning();
            }
        }
        else if (isSecond) {
            if (verboseRefinement_)
                std::cout << "second update success: " << ((alignment.errorRate() == 0) ? 1 : 0) << std::endl;

            if (adaptCorrectionRatio_) {
                secondOrderCorrectionHistory_.push_back(alignment.errorRate() == 0);
                if (secondOrderCorrectionHistory_.size() > 10)
                    secondOrderCorrectionHistory_.pop_front();
                u32 good = 0;
                for (std::deque<bool>::const_iterator it = secondOrderCorrectionHistory_.begin(); it != secondOrderCorrectionHistory_.end(); ++it)
                    if (*it)
                        good += 1;
                if (verboseRefinement_)
                    std::cout << "good count: " << good << " out of " << secondOrderCorrectionHistory_.size() << std::endl;

                if (secondOrderCorrectionHistory_.size() == 10) {
                    if (good > adaptCorrectionRatio_ && alignment.errorRate() == 0) {
                        // Tighten more
                        relaxPruningFactor_ = 1.0 + (relaxPruningFactor_ - 1.0) / adaptPruningFactor_;
                        relaxPruningOffset_ /= adaptPruningFactor_;
                        if (verboseRefinement_)
                            std::cout << "Tightened relax-pruning-factor to " << relaxPruningFactor_ << " and relax-pruning-offset to " << relaxPruningOffset_ << std::endl;
                    }
                    else if (good < adaptCorrectionRatio_ && alignment.errorRate() > 0) {
                        // Relax more
                        relaxPruningFactor_ = 1.0 + (relaxPruningFactor_ - 1.0) * adaptPruningFactor_;
                        relaxPruningOffset_ *= adaptPruningFactor_;
                        if (verboseRefinement_)
                            std::cout << "Widened relax-pruning-factor to " << relaxPruningFactor_ << " and relax-pruning-offset to " << relaxPruningOffset_ << std::endl;
                    }
                }
            }
        }

        if (forceFirstWord != 0 && (alignment.forWords.empty() || alignment.forWords[0].pron != forceFirstWord)) {
            if (verboseRefinement_)
                std::cout << "FORWARD PREFIX MISMATCH, needed " << forceFirstWord->lemma()->symbol() << std::endl;
            ret.extension = ExtendLeft;
            return ret;
        }

        if (forceLastWord != 0 && (alignment.forWords.empty() || alignment.forWords.back().pron != forceLastWord)) {
            if (verboseRefinement_)
                std::cout << "FORWARD SUFFIX MISMATCH, needed " << forceLastWord->lemma()->symbol() << std::endl;
            ret.extension = ExtendRight;
            return ret;
        }

        if (forceFirstWord != 0 && (alignment.backWords.empty() || alignment.backWords[0].pron != forceFirstWord)) {
            if (verboseRefinement_)
                std::cout << "BACKWARD PREFIX MISMATCH, needed " << forceFirstWord->lemma()->symbol() << std::endl;
            ret.extension = ExtendLeft;
            return ret;
        }

        if (forceLastWord != 0 && (alignment.backWords.empty() || alignment.backWords.back().pron != forceLastWord)) {
            if (verboseRefinement_)
                std::cout << "BACKWARD SUFFIX MISMATCH, needed " << forceLastWord->lemma()->symbol() << std::endl;
            ret.extension = ExtendRight;
            return ret;
        }

        std::set<ForwardBackwardAlignment::Range> ranges = alignment.select();

        if (ranges.size() && scoreThresholdSatisfied) {
            ranges.clear();
            if (verboseRefinement_) {
                if (completeForwardScore == completeBackwardScore)
                    std::cout << "ACCEPTING diverging result because the forward and backward pass produced the same score" << std::endl;
                else
                    std::cout << "ACCEPTING diverging result because the score difference between forward and backward pass is below the threshold: "
                              << std::abs(completeForwardScore - completeBackwardScore) / (features.size() * 0.01f) << " <= " << scoreTolerance_ << std::endl;
            }
        }

        if (!searchSpaceOK || (correctWholeSegment_ && !ranges.empty())) {
            ranges.clear();
            if (verboseRefinement_)
                std::cout << "repeating complete recognition with relaxed pruning because the search space check failed" << std::endl;
            ForwardBackwardAlignment::Range range;
            range.startTime = 0;
            range.endTime   = alignment.segmentLength_ - 1;
            ranges.insert(range);
            ret.meshEntries.clear();
        }
        else {
            if (ranges.empty()) {
                if (verboseRefinement_)
                    std::cout << "READY" << std::endl;
                ret.singleBestLattice = forward;
                return ret;
            }
        }

        if (verboseRefinement_) {
            std::cout << "total segment length: " << alignment.segmentLength_ << " number of refine-ranges: " << ranges.size() << std::endl;

            for (std::set<ForwardBackwardAlignment::Range>::iterator it = ranges.begin(); it != ranges.end(); ++it)
                std::cout << "refine-range: " << it->startTime << " -> " << it->endTime << std::endl;
        }

        Search::SearchAlgorithm::PruningRef oldForwardPruning  = recognizer_->describePruning();
        Search::SearchAlgorithm::PruningRef oldBackwardPruning = backwardRecognizer_->recognizer_->describePruning();

        if (!recognizer_->relaxPruning(relaxPruningFactor_, relaxPruningOffset_) ||
            !backwardRecognizer_->recognizer_->relaxPruning(relaxPruningFactor_, relaxPruningOffset_)) {
            backwardRecognizer_->recognizer_->resetPruning(oldBackwardPruning);
            recognizer_->resetPruning(oldForwardPruning);

            // Failed relaxing, pruning is already at limit, live with it.
            log() << "failed relaxing pruning";
            if (verboseRefinement_)
                std::cout << "FAILED RELAXING PRUNING" << std::endl;
            ret.singleBestLattice = forward;
            return ret;
        }

        std::map<ForwardBackwardAlignment::Range, ConstLatticeRef> refinements;
        bool                                                       restart = true;
        ret.pruning                                                        = oldForwardPruning->clone();
        while (restart) {
            restart = false;
            for (std::set<ForwardBackwardAlignment::Range>::iterator rangeIt = ranges.begin(); rangeIt != ranges.end(); ++rangeIt) {
                if (verboseRefinement_)
                    std::cout << "updating subrange " << rangeIt->startTime << " -> " << rangeIt->endTime << std::endl;
                verify(rangeIt->startTime >= 0 && rangeIt->endTime < features.size());

                if (refinements.count(*rangeIt)) {
                    if (verboseRefinement_)
                        std::cout << "range was already updated, skipping!" << std::endl;
                    continue;
                }

                if (globalRtf() > maximumRtf_) {
                    if (verboseRefinement_)
                        std::cout << "skipping sub-range update because the RTF limit was already reached: " << globalRtf() << " > " << maximumRtf_ << std::endl;
                    continue;
                }

                Search::SearchAlgorithm::RecognitionContext forwardContext;
                forwardContext.prefix = lmPrefix;
                forwardContext.prefix.insert(forwardContext.prefix.end(), rangeIt->prefix.begin(), rangeIt->prefix.end());
                forwardContext.suffix = rangeIt->suffix;
                forwardContext.suffix.insert(forwardContext.suffix.end(), lmSuffix.begin(), lmSuffix.end());
                forwardContext.prePhon                    = rangeIt->prePhon;
                forwardContext.sufPhon                    = rangeIt->sufPhon;
                forwardContext.coarticulation.first       = rangeIt->coarticulation.final;
                forwardContext.coarticulation.second      = rangeIt->coarticulation.initial;
                forwardContext.finalCoarticulation.first  = rangeIt->finalCoarticulation.final;
                forwardContext.finalCoarticulation.second = rangeIt->finalCoarticulation.initial;
                forwardContext.latticeMode                = Search::SearchAlgorithm::RecognitionContext::No;

                Search::SearchAlgorithm::RecognitionContext backwardContext;
                backwardContext.prefix.assign(forwardContext.suffix.rbegin(), forwardContext.suffix.rend());
                backwardContext.suffix.assign(forwardContext.prefix.rbegin(), forwardContext.prefix.rend());
                backwardContext.coarticulation.first       = rangeIt->backwardCoarticulation.final;
                backwardContext.coarticulation.second      = rangeIt->backwardCoarticulation.initial;
                backwardContext.finalCoarticulation.first  = rangeIt->finalBackwardCoarticulation.final;
                backwardContext.finalCoarticulation.second = rangeIt->finalBackwardCoarticulation.initial;
                backwardContext.latticeMode                = Search::SearchAlgorithm::RecognitionContext::No;
                backwardContext.prePhon                    = rangeIt->sufPhon;
                backwardContext.sufPhon                    = rangeIt->prePhon;

                Search::SearchAlgorithm::RecognitionContext oldForwardContext, oldBackwardContext;
                oldForwardContext  = recognizer_->setContext(forwardContext);
                oldBackwardContext = backwardRecognizer_->recognizer_->setContext(backwardContext);

                RecognizedSequence refined = refine(offset + rangeIt->startTime,
                                                    std::deque<FeatureRef>(features.begin() + rangeIt->startTime, features.begin() + rangeIt->endTime + 1),
                                                    rangeIt->firstForWord != -1 ? alignment.forWords[rangeIt->firstForWord].pron : forceFirstWord,
                                                    rangeIt->lastForWord != -1 ? alignment.forWords[rangeIt->lastForWord].pron : forceLastWord,
                                                    forwardContext.prefix,
                                                    forwardContext.suffix,
                                                    false,
                                                    isInitial);

                for (std::vector<MeshEntry>::iterator entryIt = refined.meshEntries.begin(); entryIt != refined.meshEntries.end(); ++entryIt) {
                    MeshEntry e(*entryIt);
                    e.timeOffset += rangeIt->startTime;
                    ret.meshEntries.push_back(e);
                }

                if (refined.pruning.get())
                    ret.pruning->merge(refined.pruning, features.size(), rangeIt->startTime, rangeIt->endTime);

                recognizer_->setContext(oldForwardContext);
                backwardRecognizer_->recognizer_->setContext(oldBackwardContext);

                if (refined.singleBestLattice.get() == 0) {
                    if (verboseRefinement_)
                        std::cout << "RANGE UPDATE FAILED, NEED TO EXTEND THE RANGE!!" << std::endl;  // Should almost never happen, as forward and backward search agreed on the context words
                    verify(refined.extension != ExtendNone);
                    if (refined.extension & ExtendLeft) {
                        if (rangeIt->firstForWord == -1 || (rangeIt->firstForWord == 0 && forceFirstWord)) {
                            ret.extension = Extension(ret.extension | ExtendLeft);
                        }
                        else {
                            alignment.invalidateForWord(rangeIt->firstForWord);
                        }
                    }
                    if (refined.extension & ExtendRight) {
                        if (rangeIt->lastForWord == -1 || (rangeIt->lastForWord == alignment.forWords.size() - 1 && forceLastWord)) {
                            ret.extension = Extension(ret.extension | ExtendRight);
                        }
                        else {
                            alignment.invalidateForWord(rangeIt->lastForWord);
                        }
                    }
                    if (ret.extension != ExtendNone) {
                        if (verboseRefinement_)
                            std::cout << "NEED TO EXTEND UPWARDS: " << ret.extension << std::endl;
                        recognizer_->resetPruning(oldForwardPruning);
                        backwardRecognizer_->recognizer_->resetPruning(oldBackwardPruning);
                        return ret;
                    }
                    restart = true;
                    alignment.normalize();
                    std::set<ForwardBackwardAlignment::Range> oldRanges = ranges;
                    ranges                                              = alignment.select();
                    for (std::set<ForwardBackwardAlignment::Range>::const_iterator oldRangeIt = oldRanges.begin(); oldRangeIt != oldRanges.end(); ++oldRangeIt)
                        if (!ranges.count(*oldRangeIt))
                            refinements.erase(*oldRangeIt);

                    break;
                }
                refinements.insert(std::make_pair(*rangeIt, refined.singleBestLattice));
            }
        }

        StaticBoundariesRef newBoundaries = StaticBoundariesRef(new StaticBoundaries);
        StaticLatticeRef    newLattice    = StaticLatticeRef(new StaticLattice);
        newLattice->setType(Fsa::TypeAcceptor);
        newLattice->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
        newLattice->setInputAlphabet(lpAlphabet_);
        newLattice->setSemiring(semiring_);
        newLattice->setDescription(Core::form("refine(recog(%s))", segment_->name().c_str()));
        newLattice->setBoundaries(ConstBoundariesRef(newBoundaries));
        newLattice->setInitialStateId(0);

        {
            State*                      currentState = newLattice->newState(0);
            Flf::Lattice::ConstStateRef forwardState = forward->getState(forward->initialStateId());
            newBoundaries->set(currentState->id(), forward->boundary(forwardState->id()));
            currentState->setWeight(forwardState->weight());
            currentState->setTags(forwardState->tags());
            verify(forwardState->nArcs());

            while (forwardState->nArcs()) {
                verify(forwardState->nArcs() == 1);
                std::map<ForwardBackwardAlignment::Range, ConstLatticeRef>::iterator insertIt = refinements.end();
                for (std::map<ForwardBackwardAlignment::Range, ConstLatticeRef>::iterator refineIt = refinements.begin(); refineIt != refinements.end(); ++refineIt) {
                    const ForwardBackwardAlignment::Range& range(refineIt->first);
                    if (range.firstForWord == -1) {
                        if (forwardState->id() == forward->initialStateId()) {
                            insertIt = refineIt;
                            break;
                        }
                    }
                    else if (alignment.forWords[range.firstForWord].originState == forwardState->id()) {
                        insertIt = refineIt;
                        break;
                    }
                }

                if (insertIt != refinements.end()) {
                    // Insert new lattice
                    Speech::TimeframeIndex timeOffset = newBoundaries->get(currentState->id()).time();
                    if (timeOffset != insertIt->first.startTime) {
                        std::cout << "Time-offset mismatch " << timeOffset << " " << insertIt->first.startTime << std::endl;
                    }
                    verify(timeOffset == insertIt->first.startTime);
                    Flf::ConstLatticeRef        insertLattice = insertIt->second;
                    Flf::Lattice::ConstStateRef insertState   = insertLattice->getState(insertLattice->initialStateId());
                    verify(insertState->nArcs() == 1);
                    while (insertState.get()) {
                        verify(insertState->nArcs() == 1);
                        if (lpAlphabet_->lemmaPronunciation(insertState->getArc(0)->input()) == 0) {
                            // Skip final sentence-end arc
                            verify(insertLattice->getState(insertState->getArc(0)->target())->nArcs() == 0);
                            break;
                        }
                        const Arc* insertArc = insertState->getArc(0);
                        State*     newState  = newLattice->newState();
                        currentState->newArc(newState->id(), insertArc->weight(), insertArc->input());
                        currentState      = newState;
                        insertState       = insertLattice->getState(insertArc->target());
                        Boundary boundary = insertLattice->boundary(insertState->id());
                        boundary.setTime(boundary.time() + timeOffset);
                        newBoundaries->set(currentState->id(), boundary);
                        currentState->setWeight(insertState->weight());
                        currentState->setTags(insertState->tags());
                    }
                    while (forwardState->nArcs() && lpAlphabet_->lemmaPronunciation(forwardState->getArc(0)->input()) != 0) {
                        bool matched = insertIt->first.lastForWord != -1 && forwardState->id() == alignment.forWords[insertIt->first.lastForWord].originState;
                        forwardState = forward->getState(forwardState->getArc(0)->target());
                        if (matched)
                            break;
                    }
                    refinements.erase(insertIt);
                }
                else {
                    // Copy one arc from the forward-lattice
                    const Arc* forwardArc = forwardState->getArc(0);

                    State* newState = newLattice->newState();
                    currentState->newArc(newState->id(), forwardArc->weight(), forwardArc->input());
                    currentState = newState;
                    forwardState = forward->getState(forwardArc->target());
                    newBoundaries->set(currentState->id(), forward->boundary(forwardState->id()));
                    currentState->setWeight(forwardState->weight());
                    currentState->setTags(forwardState->tags());
                }
            }
        }
        recognizer_->resetPruning(oldForwardPruning);
        backwardRecognizer_->recognizer_->resetPruning(oldBackwardPruning);
        ret.singleBestLattice = newLattice;
        if (verboseRefinement_) {
            std::cout << "new words for " << features.size() << " features:";
            std::pair<std::deque<ForwardBackwardAlignment::Word>, u32> words = alignment.parseForwardLattice(ret.singleBestLattice);
            for (u32 w = 0; w < words.first.size(); ++w) {
                std::cout << " " << words.first[w].pron->lemma()->preferredOrthographicForm();
            }

            std::cout << std::endl;
        }
        return ret;
    }

    ConstLatticeRef buildLattice(Core::Ref<const Search::LatticeAdaptor> la, bool zeroStartTime) {
        Flf::LatticeHandler* handler = Module::instance().createLatticeHandler(config);
        handler->setLexicon(Lexicon::us());
        if (la->empty())
            return ConstLatticeRef();
        ::Lattice::ConstWordLatticeRef             lattice    = la->wordLattice(handler);
        Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
        Fsa::ConstAutomatonRef                     amFsa      = lattice->part(::Lattice::WordLattice::acousticFsa);
        Fsa::ConstAutomatonRef                     lmFsa      = lattice->part(::Lattice::WordLattice::lmFsa);
        require_(Fsa::isAcyclic(amFsa) && Fsa::isAcyclic(lmFsa));

        StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
        StaticLatticeRef    s = StaticLatticeRef(new StaticLattice);
        s->setType(Fsa::TypeAcceptor);
        s->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
        s->setInputAlphabet(lpAlphabet_);
        s->setSemiring(semiring_);
        s->setDescription(Core::form("recog(%s)", segment_->name().c_str()));
        s->setBoundaries(ConstBoundariesRef(b));
        s->setInitialStateId(0);

        Time timeOffset = zeroStartTime ? (*boundaries)[amFsa->initialStateId()].time() : 0;

        Fsa::Stack<Fsa::StateId>   S;
        Core::Vector<Fsa::StateId> sidMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
        sidMap[amFsa->initialStateId()] = 0;
        S.push_back(amFsa->initialStateId());
        Fsa::StateId nextSid   = 2;
        Time         finalTime = 0;
        while (!S.isEmpty()) {
            Fsa::StateId sid = S.pop();
            verify(sid < sidMap.size());
            const ::Lattice::WordBoundary& boundary((*boundaries)[sid]);
            Fsa::ConstStateRef             amSr = amFsa->getState(sid);
            Fsa::ConstStateRef             lmSr = lmFsa->getState(sid);
            State*                         sp   = new State(sidMap[sid]);
            s->setState(sp);
            b->set(sp->id(), Boundary(boundary.time() - timeOffset,
                                      Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
            if (amSr->isFinal()) {
                sp->newArc(1, buildScore(Fsa::InvalidLabelId, amSr->weight(), lmSr->weight()), sentenceEndLabel_);
                finalTime = std::max(finalTime, boundary.time() - timeOffset);
            }
            for (Fsa::State::const_iterator am_a = amSr->begin(), lm_a = lmSr->begin(); (am_a != amSr->end()) && (lm_a != lmSr->end()); ++am_a, ++lm_a) {
                sidMap.grow(am_a->target(), Fsa::InvalidStateId);
                if (sidMap[am_a->target()] == Fsa::InvalidStateId) {
                    sidMap[am_a->target()] = nextSid++;
                    S.push(am_a->target());
                }
                Fsa::ConstStateRef targetAmSr = amFsa->getState(am_a->target());
                Fsa::ConstStateRef targetLmSr = amFsa->getState(lm_a->target());
                if (targetAmSr->isFinal() && targetLmSr->isFinal()) {
                    if (am_a->input() == Fsa::Epsilon) {
                        ScoresRef scores = buildScore(am_a->input(), am_a->weight(), lm_a->weight());
                        scores->add(amId_, Score(targetAmSr->weight()));
                        scores->add(lmId_, Score(targetLmSr->weight()) / lmScale_);
                        sp->newArc(1, scores, sentenceEndLabel_);
                    }
                    else
                        sp->newArc(sidMap[am_a->target()], buildScore(am_a->input(), am_a->weight(), lm_a->weight()), am_a->input());
                }
                else
                    sp->newArc(sidMap[am_a->target()], buildScore(am_a->input(), am_a->weight(), lm_a->weight()), am_a->input());
            }
        }
        State* sp = new State(1);
        sp->setFinal(semiring_->clone(semiring_->one()));
        s->setState(sp);
        b->set(sp->id(), Boundary(finalTime));

        return ConstLatticeRef(s);
    }

    ConstLatticeRef postProcess(ConstLatticeRef l) {
        if (applyNonWordClosureFilter_) {
            l                                = transducer(l);
            l                                = applyOneToOneLabelMap(l, nonWordToEpsilonMap_);
            StaticLatticeRef filteredLattice = applyEpsClosureWeakDeterminizationFilter(l);
            trimInPlace(filteredLattice);
            l = projectOutput(filteredLattice);
        }
        if (applyUniqueSentenceAlignmentFilter_)
            l = uniqueSentenceAlignmentFilter(l);
        if (addConfidenceScores_ || fwdBwdThreshold_ >= 0 || minArcsPerSecond_ || maxArcsPerSecond_ < Core::Type<f32>::max) {
            std::pair<ConstLatticeRef, ConstFwdBwdRef> latAndFb = FwdBwd::build(l, posteriorSemiring_);
            l                                                   = latAndFb.first;
            ConstFwdBwdRef fb                                   = latAndFb.second;
            if (addConfidenceScores_) {
                ConstPosteriorCnRef cn = buildFramePosteriorCn(l, fb);
                l                      = extendByFCnConfidence(l, cn, confidenceId_, RescoreModeInPlaceCache);
                l                      = persistent(l);
            }
            if (fwdBwdThreshold_ >= 0 || minArcsPerSecond_ || maxArcsPerSecond_ < Core::Type<f32>::max) {
                l = pruneByFwdBwdScores(l,
                                        fb,
                                        fwdBwdThreshold_ < 0 ? (fb->max() - fb->min()) : fwdBwdThreshold_,
                                        minArcsPerSecond_,
                                        maxArcsPerSecond_);

                StaticLatticeRef trimmedLattice = StaticLatticeRef(new StaticLattice);
                copy(l, trimmedLattice.get(), 0);
                trimInPlace(trimmedLattice);
                trimmedLattice->setBoundaries(l->getBoundaries());
                l = normalizeCopy(trimmedLattice);
            }
        }

        return l;
    }

    void logTraceback(const Search::Traceback& traceback) {
        tracebackChannel_ << Core::XmlOpen("traceback") + Core::XmlAttribute("type", "xml");
        u32                 previousIndex = traceback.begin()->time;
        Search::ScoreVector previousScore(0.0, 0.0);
        for (std::vector<Search::TracebackItem>::const_iterator tbi = traceback.begin(); tbi != traceback.end(); ++tbi) {
            if (tbi->pronunciation) {
                tracebackChannel_ << Core::XmlOpen("item") + Core::XmlAttribute("type", "pronunciation")
                                  << Core::XmlFull("orth", tbi->pronunciation->lemma()->preferredOrthographicForm())
                                  << Core::XmlFull("phon", tbi->pronunciation->pronunciation()->format(lexicon_->phonemeInventory()))
                                  << Core::XmlFull("score", f32(tbi->score.acoustic - previousScore.acoustic)) + Core::XmlAttribute("type", "acoustic")
                                  << Core::XmlFull("score", f32(tbi->score.lm - previousScore.lm)) + Core::XmlAttribute("type", "language");
                if (previousIndex < tbi->time)
                    tracebackChannel_ << Core::XmlEmpty("samples") + Core::XmlAttribute("start", f32(featureTimes_[previousIndex].startTime())) +
                                                 Core::XmlAttribute("end", f32(featureTimes_[tbi->time - 1].endTime()))
                                      << Core::XmlEmpty("features") + Core::XmlAttribute("start", previousIndex) +
                                                 Core::XmlAttribute("end", tbi->time - 1);
                tracebackChannel_ << Core::XmlClose("item");
            }
            previousScore = tbi->score;
            previousIndex = tbi->time;
        }
        tracebackChannel_ << Core::XmlClose("traceback");
    }

public:
    IncrementalRecognizer(const Core::Configuration& _config, ModelCombinationRef mc, bool backward = false, std::string forwardLmFile = std::string())
            : Core::Component(_config),
              Speech::Recognizer(_config),
              mc_(mc),
              modelAdaptor_(SegmentwiseModelAdaptorRef(new SegmentwiseModelAdaptor(mc))),
              tracebackChannel_(config, "traceback"),
              segmentFeatureCount_(0),
              lmContextLength_(paramLmContextLength(config)),
              relaxPruningFactor_(paramRelaxPruningFactor(config)),
              relaxPruningOffset_(paramRelaxPruningOffset(config)),
              latticeRelaxPruningFactor_(paramLatticeRelaxPruningFactor(config)),
              latticeRelaxPruningOffset_(paramLatticeRelaxPruningOffset(config)),
              adaptInitialUpdateRate_(paramAdaptInitialUpdateRate(config)),
              adaptRelaxPruningFactor_(paramAdaptRelaxPruningFactor(config)),
              adaptRelaxPruningOffset_(paramAdaptRelaxPruningOffset(config)),
              latticeRelaxPruningInterval_(paramLatticeRelaxPruningInterval(config)),
              adaptCorrectionRatio_(paramAdaptCorrectionRatio(config)),
              scoreTolerance_(paramScoreTolerance(config) * mc->languageModel()->scale()),
              adaptPruningFactor_(paramAdaptPruningFactor(config)),
              minArcsPerSecond_(paramMinArcsPerSecond(config)),
              maxArcsPerSecond_(paramMaxArcsPerSecond(config)),
              maxLatticeRegenerations_(paramMaxLatticeRegenerations(config)),
              onlyEnforceMinimumSearchSpace_(paramOnlyEnforceMinimumSearchSpace(config)),
              correctStrictInitial_(paramCorrectStrictInitial(config)),
              maximumRtf_(paramMaxRtf(config)),
              segment_(0),
              subSegment_(0),
              verboseRefinement_(paramVerboseRefinement(config)),
              considerSentenceBegin_(paramConsiderSentenceBegin(config)),
              preCacheAllFrames_(paramPreCache(config)) {
        if (!backward) {
            select("lm").get("file", forwardLmFile);
            backwardRecognizer_.reset(new IncrementalRecognizer(select("backward"), mc_, true, forwardLmFile));
            contextScorerCache_ = backwardRecognizer_->contextScorerCache_;
        }
        else {
            Core::Configuration lmCfg = select("lm");
            std::string         backwardLmFile;
            lmCfg.get("file", backwardLmFile);
            if (forwardLmFile == backwardLmFile)
                lmCfg.set(lmCfg.getSelection() + ".reverse-lm");
            mc_                 = getModelCombination(config, mc_->acousticModel(), getLm(lmCfg));
            contextScorerCache_ = Core::Ref<Mm::ContextScorerCache>(new Mm::ContextScorerCache(paramCacheFrames(config)));
        }

        Core::Configuration featureExtractionConfig(config, "feature-extraction");
        DataSourceRef       dataSource = DataSourceRef(Speech::Module::instance().createDataSource(featureExtractionConfig));
        featureExtractor_              = SegmentwiseFeatureExtractorRef(new SegmentwiseFeatureExtractor(featureExtractionConfig, dataSource));

        require(mc_);
        pronScale_                          = mc_->pronunciationScale();
        lmScale_                            = mc_->languageModel()->scale();
        meshCombination_                    = paramMeshCombination(config);
        meshRescoring_                      = paramMeshRescoring(config) || meshCombination_;
        expandTransits_                     = paramExpandTransits(config);
        forceForwardBackwardLattices_       = paramForceForwardBackwardLattices(config);
        rescoreWordEndLimit_                = paramRescoreWordEndLimit(config);
        addPronunciationScores_             = paramPronunciationScore(config);
        correctWholeSegment_                = paramCorrectWholeSegment(config);
        correctForceEqualAlignment_         = paramCorrectForceEqualAlignment(config);
        correctForceEqualScore_             = paramCorrectForceEqualScore(config);
        correctIncludeNoise_                = paramCorrectIncludeNoise(config);
        addConfidenceScores_                = paramConfidenceScore(config);
        applyNonWordClosureFilter_          = paramApplyNonWordClosureFilter(config);
        applyUniqueSentenceAlignmentFilter_ = paramApplyUniqueSentenceAlignmentFilter(config);
        fwdBwdThreshold_                    = paramPosteriorPruningThreshold(config);
        {
            Core::Component::Message msg(log());
            lpAlphabet_                 = mc_->lexicon()->lemmaPronunciationAlphabet();
            sentenceEndLabel_           = Fsa::Epsilon;
            const Bliss::Lemma* special = mc_->lexicon()->specialLemma("sentence-end");
            if (special) {
                Bliss::Lemma::LemmaPronunciationRange lpRange = special->pronunciations();
                if (lpRange.first != lpRange.second)
                    sentenceEndLabel_ = lpRange.first->id();
            }
            msg << "Sentence end symbol is \"" << lpAlphabet_->symbol(sentenceEndLabel_) << "\".\n";

            u32 dim = 0;
            amId_   = dim++;
            lmId_   = dim++;
            if (addPronunciationScores_)
                pronunciationId_ = dim++;
            else
                pronunciationId_ = Semiring::InvalidId;
            if (addConfidenceScores_)
                confidenceId_ = dim++;
            else
                confidenceId_ = Semiring::InvalidId;
            semiring_ = Semiring::create(Fsa::SemiringTypeTropical, dim);
            semiring_->setKey(amId_, "am");
            semiring_->setScale(amId_, 1.0);
            semiring_->setKey(lmId_, "lm");
            semiring_->setScale(lmId_, lmScale_);
            if (addPronunciationScores_) {
                semiring_->setKey(pronunciationId_, "pronunciation");
                semiring_->setScale(pronunciationId_, pronScale_);
            }
            if (addConfidenceScores_) {
                semiring_->setKey(confidenceId_, "confidence");
                semiring_->setScale(confidenceId_, 0.0);
            }
            msg << "Semiring is " << semiring_->name() << ".\n";
            if (addConfidenceScores_ || fwdBwdThreshold_ >= 0) {
                posteriorSemiring_ = toLogSemiring(semiring_, paramAlpha(select("fb")));
                msg << "Posterior-semiring is " << posteriorSemiring_->name() << ".\n";
            }
            if (applyNonWordClosureFilter_) {
                nonWordToEpsilonMap_ = LabelMap::createNonWordToEpsilonMap(Lexicon::LemmaPronunciationAlphabetId);
                msg << "Non-word-closure filter is active.\n";
            }
            if (addConfidenceScores_) {
                msg << "Confidence score calculation is active (Attention: Confidence scores are calculated on lemma pronunciations). \n";
            }
            if (fwdBwdThreshold_ >= 0) {
                msg << "Posterior pruning is active (threshold=" << fwdBwdThreshold_ << ").\n";
            }
            if (minArcsPerSecond_) {
                msg << "Min-arcs-per-second: " << minArcsPerSecond_ << "\n";
            }
            if (maxArcsPerSecond_) {
                msg << "Max-arcs-per-second: " << maxArcsPerSecond_ << "\n";
            }
            if (applyUniqueSentenceAlignmentFilter_) {
                msg << "Lattice will be filtered for unique sentence alignments.\n";
            }
        }
        initializeRecognizer(*mc_);
        delayedRecognition_.reset(new Speech::RecognizerDelayHandler(recognizer_, acousticModel_, contextScorerCache_));
        verify(recognizer_);
    }

    bool getData(FeatureRef& feature) {
        if (dataSource_)
            return dataSource_->getData(feature);
        else
            return false;
    }

    void newSegment() {
        if (contextScorerCache_.get())
            contextScorerCache_->clear();
    }

    void startRecognition(const Bliss::SpeechSegment* segment, bool useDataSource = true) {
        if (segment_)
            finishRecognition();

        segment_ = segment;
        if (!segment_->orth().empty()) {
            clog() << Core::XmlOpen("orth") + Core::XmlAttribute("source", "reference")
                   << segment_->orth()
                   << Core::XmlClose("orth");
        }
        recognizer_->resetStatistics();
        recognizer_->setSegment(segment_);
        recognizer_->restart();
        traceback_.clear();

        acousticModel_->setKey(segment_->fullName());

        modelAdaptor_->enterSegment(segment_);
        featureExtractor_->enterSegment(segment_);

        if (!useDataSource) {
            dataSource_.reset();
        }
        else {
            dataSource_ = featureExtractor_->extractor();
            dataSource_->initialize(const_cast<Bliss::SpeechSegment*>(segment_));

            if (backwardRecognizer_) {
                backwardRecognizer_->startRecognition(segment, false);
            }
            else {  /// @todo Also check compatibility when using the backward-recognizer. However we need all features.
                FeatureRef feature;

                if (getData(feature)) {
                    // check the dimension segment
                    AcousticModelRef acousticModel = modelAdaptor_->modelCombination()->acousticModel();
                    if (acousticModel) {
                        Mm::FeatureDescription* description = feature->getDescription(*featureExtractor_);
                        if (!acousticModel->isCompatible(*description))
                            acousticModel->respondToDelayedErrors();
                        delete description;
                    }
                    putFeature(feature);
                }
            }
        }
    }

    void putFeature(FeatureRef feature) {
        featureTimes_.push_back(feature->timestamp());
        delayedRecognition_->add(feature);
    }

    void finalize() {
        if (adaptInitialUpdateRate_)
            log() << "final adapted base pruning: " << recognizer_->describePruning()->format();
        if (adaptCorrectionRatio_)
            log() << "final adapted relax-pruning-factor: " << relaxPruningFactor_ << " , relax-pruning-offset: " << relaxPruningOffset_;
    }

    void reset() {
        if (segment_)
            finishRecognition();
        featureExtractor_->reset();
        if (modelAdaptor_)
            modelAdaptor_->reset();
    }

    std::pair<ConstLatticeRef, ConstSegmentRef> buildLatticeAndSegment(Core::Ref<const Search::LatticeAdaptor> la) {
        verify(segment_);

        Speech::TimeframeIndex startTime;
        {
            Flf::LatticeHandler*                       handler    = Module::instance().createLatticeHandler(config);
            ::Lattice::ConstWordLatticeRef             lattice    = la->wordLattice(handler);
            Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
            Fsa::ConstAutomatonRef                     amFsa      = lattice->part(::Lattice::WordLattice::acousticFsa);
            startTime                                             = (*boundaries)[amFsa->initialStateId()].time();
        }

        ConstLatticeRef partialLattice = buildLattice(la, true);

        Fsa::StateId endState = partialLattice->initialStateId();
        while (partialLattice->getState(endState)->nArcs())
            endState = partialLattice->getState(endState)->getArc(0)->target();
        Speech::TimeframeIndex endTime = partialLattice->boundary(endState).time() + startTime;

        log() << "got partial lattice for interval " << startTime << " -> " << endTime;

        verify(startTime < featureTimes_.size());
        if (endTime >= featureTimes_.size()) {
            log() << "end-time is too high: " << endTime << " max. " << featureTimes_.size() - 1 << ", truncated!";
            endTime = featureTimes_.size() - 1;
        }

        verify(startTime < endTime);
        verify(endTime < featureTimes_.size());

        SegmentRef newSegment(new Flf::Segment(segment_));
        newSegment->setOrthography("");
        newSegment->setStartTime(featureTimes_[startTime].startTime());
        newSegment->setEndTime(featureTimes_[endTime].endTime());
        verify(newSegment->segmentId().size());
        {
            std::string::size_type timeStart = newSegment->segmentId().rfind("_");
            std::string::size_type timeGap   = newSegment->segmentId().rfind("-");
            std::ostringstream     os;
            if (timeStart != std::string::npos && timeGap != std::string::npos && timeGap > timeStart) {
                // Create a new segment name with corrected time information in the identifier
                os << newSegment->segmentId().substr(0, timeStart + 1);
                os << std::setiosflags(std::ios::fixed) << std::setprecision(3) << newSegment->startTime() << "-" << newSegment->endTime();
            }
            else {
                // Create a new segment name by appending "_$subsegment"
                os << newSegment->segmentId() << "_" << subSegment_;
            }
            newSegment->setSegmentId(os.str());
        }
        log() << "created segment " << newSegment->segmentId();
        info(partialLattice, clog());
        subSegment_ += 1;
        return std::make_pair(partialLattice, newSegment);
    }

    bool recognitionPending() const {
        return segment_;
    }

    std::pair<ConstLatticeRef, ConstSegmentRef> recognize() {
        std::pair<ConstLatticeRef, ConstSegmentRef> ret;

        if (!segment_)
            return ret;

        if (backwardRecognizer_.get()) {
            std::deque<FeatureRef> features;
            {
                FeatureRef feature;
                while (getData(feature))
                    features.push_back(feature);
            }

            globalTimer_.start();

            if (preCacheAllFrames_) {
                struct PreCacher : public Search::SearchAlgorithm {
                    PreCacher()
                            : Core::Component(Core::Configuration()),
                              SearchAlgorithm(Core::Configuration()) {}
                    virtual void feed(const Mm::FeatureScorer::Scorer& scorer) {
                        dynamic_cast<const Mm::CachedFeatureScorer::CachedContextScorerOverlay*>(scorer.get())->precache();
                    }
                    virtual void                                    getCurrentBestSentence(Traceback& result) const {}
                    virtual Core::Ref<const Search::LatticeAdaptor> getCurrentWordLattice() const {
                        return {};
                    }
                    virtual void logStatistics() const {}
                    virtual void resetStatistics() {}
                    virtual void restart() {}
                    virtual void setGrammar(Fsa::ConstAutomatonRef) {}
                    virtual bool setModelCombination(const Speech::ModelCombination& modelCombination) {
                        return false;
                    }
                    virtual bool setLanguageModel(Core::Ref<const Lm::ScaledLanguageModel>) {
                        defect();
                    }
                } precacher;
                Core::Timer                    timer;
                Speech::RecognizerDelayHandler handler(&precacher, acousticModel_, contextScorerCache_);
                for (std::deque<FeatureRef>::iterator it = features.begin(); it != features.end(); ++it)
                    handler.add(*it);
                while (handler.flush())
                    ;
            }

            f32 preCachingTime = globalTimer_.user();

            segmentFeatureCount_ = features.size();

            Search::SearchAlgorithm::RecognitionContext refineContext;
            if (meshRescoring_)
                refineContext.latticeMode = Search::SearchAlgorithm::RecognitionContext::Yes;
            else
                refineContext.latticeMode = Search::SearchAlgorithm::RecognitionContext::No;

            Search::SearchAlgorithm::RecognitionContext oldForwardContext, oldBackwardContext;
            oldForwardContext  = recognizer_->setContext(refineContext);
            oldBackwardContext = backwardRecognizer_->recognizer_->setContext(refineContext);

            bool separateLatticeRecognition = latticeRelaxPruningFactor_ > 1.0 || latticeRelaxPruningOffset_ > 0.0;

            if (separateLatticeRecognition) {
                // Due to word boundary crossings, skips can make the forward and backward models unequal,
                // which can make the forward-backward alignment fail forever. When we anyway add an addition
                // recognition pass to generate lattices, then we can completely disable skips during the forward/backward
                // search, as we do here. Otherwise, it is the users option, because the disabling of skips would change the models.
                recognizer_->setAllowHmmSkips(false);
                backwardRecognizer_->recognizer_->setAllowHmmSkips(false);
            }

            RecognizedSequence refined = refine(0, features, 0, 0, std::vector<const Bliss::Lemma*>(), std::vector<const Bliss::Lemma*>(), true);

            if (separateLatticeRecognition) {
                recognizer_->setAllowHmmSkips(true);
                backwardRecognizer_->recognizer_->setAllowHmmSkips(true);
            }

            ret = std::make_pair(refined.singleBestLattice, SegmentRef(new Flf::Segment(segment_)));

            verify(refined.pruning.get());

            std::cout << "used pruning (" << features.size() << " frames): " << refined.pruning->format() << std::endl;
            log() << "used pruning (" << features.size() << " frames): " << refined.pruning->format();

            f32 refinementTime          = globalTimer_.user() - preCachingTime;
            f32 postProcessingStartTime = globalTimer_.user();

            if (separateLatticeRecognition) {
                refined.meshEntries.clear();
                Search::SearchAlgorithm::RecognitionContext latticeContext;
                latticeContext.latticeMode                         = Search::SearchAlgorithm::RecognitionContext::Yes;
                Search::SearchAlgorithm::PruningRef latticePruning = refined.pruning->clone();
                bool                                extended       = latticePruning->extend(latticeRelaxPruningFactor_, latticeRelaxPruningOffset_, latticeRelaxPruningInterval_);
                verify(extended);
                std::cout << "generating lattice for whole segment with pruning: " << latticePruning->format() << std::endl;
                log() << "used extended pruning for lattice-generation (" << features.size() << " timeframes): " << latticePruning->format();
                Search::SearchAlgorithm::PruningRef oldForwardPruning = recognizer_->describePruning();
                recognizer_->resetPruning(latticePruning);
                recognizer_->setContext(latticeContext);

                ret.first = recognizeFeatures(features).first;

                if (meshRescoring_) {
                    MeshEntry e;
                    e.lattice = ret.first;
                    refined.meshEntries.push_back(e);
                }
                else {
                    u32 i = 0;
                    while (true) {
                        ++i;
                        postProcessingStartTime = globalTimer_.user();
                        ret.first               = postProcess(ret.first);
                        if (minArcsPerSecond_) {
                            LatticeCounts counts  = count(ret.first);
                            f32           minArcs = (minArcsPerSecond_ * features.size()) / 100.0;
                            if (counts.nArcs_ < minArcs) {
                                if (i > maxLatticeRegenerations_) {
                                    log() << "not enough arcs: "
                                          << counts.nArcs_
                                          << " need at least "
                                          << minArcs
                                          << ", but NOT regenerating lattice because maximum number of regenerations is already reached (" << i << ")";
                                    break;
                                }
                                else {
                                    log() << "not enough arcs: " << counts.nArcs_ << " need at least " << minArcs;
                                    if (!recognizer_->relaxPruning(relaxPruningFactor_, relaxPruningOffset_)) {
                                        log() << "FAILED relaxing pruning for regeneration";
                                        break;
                                    }
                                    else {
                                        latticePruning = latticePruning->clone();
                                        latticePruning->extend(relaxPruningFactor_, relaxPruningOffset_, 0);
                                        recognizer_->resetPruning(latticePruning);
                                        log() << "regenerating lattice with extended pruning: " << latticePruning->format();
                                        ret.first = recognizeFeatures(features).first;
                                    }
                                }
                            }
                            else {
                                break;
                            }
                        }
                        else {
                            break;
                        }
                    }
                }

                recognizer_->resetPruning(oldForwardPruning);

                if (meshRescoring_ && forceForwardBackwardLattices_) {
                    MeshEntry e;
                    e.lattice       = recognizeFeatures(std::deque<Flf::FeatureRef>(features.rbegin(), features.rend())).first;
                    e.reverseOffset = features.size();
                    refined.meshEntries.push_back(e);
                }

                info(ret.first, clog());
            }

            if (meshRescoring_ && !refined.meshEntries.empty()) {
                log() << "building mesh from " << refined.meshEntries.size() << " individual sub-meshes";
                ret.first = mesh(refined.meshEntries);
                log() << "rescoring with beam " << refined.pruning->maxMasterBeam();
                std::cout << "rescoring with beam " << refined.pruning->maxMasterBeam() << std::endl;
                ret.first = decodeRescoreLm(ret.first, mc_->languageModel(), refined.pruning->maxMasterBeam(), rescoreWordEndLimit_);
                ret.first = postProcess(ret.first);
            }

            recognizer_->setContext(oldForwardContext);
            backwardRecognizer_->recognizer_->setContext(oldBackwardContext);

            globalTimer_.stop();

            if (features.size()) {
                std::cout << "global needed time: " << globalTimer_.user()
                          << " for frame-duration: " << (features.size() / 100.0)
                          << " global RTF: " << globalRtf()
                          << " (postprocessing RTF " << (globalTimer_.user() - postProcessingStartTime) / (features.size() / 100.0)
                          << " forward-backward RTF " << refinementTime / (features.size() / 100.0)
                          << ", precaching RTF " << preCachingTime / (features.size() / 100.0) << ")" << std::endl;
                log() << "global needed time: " << globalTimer_.user()
                      << " for frame-duration: " << (features.size() / 100.0)
                      << " global RTF: " << globalRtf()
                      << " (postprocessing RTF " << (globalTimer_.user() - postProcessingStartTime) / (features.size() / 100.0)
                      << " forward-backward RTF " << refinementTime / (features.size() / 100.0)
                      << ", precaching RTF " << preCachingTime / (features.size() / 100.0) << ")";
            }

            finishRecognition();
            backwardRecognizer_->finishRecognition();
            return ret;
        }

        Core::Timer timer;
        timer.start();

        FeatureRef feature;
        while (getData(feature)) {
            putFeature(feature);
            Core::Ref<const Search::LatticeAdaptor> la = recognizer_->getPartialWordLattice();
            if (la)
                return buildLatticeAndSegment(la);
        }

        while (delayedRecognition_->flush())
            ;

        timer.stop();

        if (featureTimes_.size())
            std::cout << "global needed time: " << timer.user()
                      << " for frame-duration: " << (featureTimes_.size() / 100.0)
                      << " global RTF: " << timer.user() / (featureTimes_.size() / 100.0) << std::endl;

        if (subSegment_) {
            ret = buildLatticeAndSegment(recognizer_->getCurrentWordLattice());
            finishRecognition();
            return ret;
        }
        else {
            Core::Ref<const Search::LatticeAdaptor> recognizerLattice = recognizer_->getCurrentWordLattice();
            ret                                                       = std::make_pair(buildLattice(recognizerLattice, false), SegmentRef(new Flf::Segment(segment_)));

            info(ret.first, clog());
            processResult();
            finishRecognition();
            return ret;
        }
    }

    void finishRecognition() {
        if (dataSource_)
            dataSource_->finalize();
        featureExtractor_->leaveSegment(segment_);
        modelAdaptor_->leaveSegment(segment_);
        recognizer_->logStatistics();
        segment_    = 0;
        subSegment_ = 0;
        dataSource_.reset();
        featureTimes_.clear();
        delayedRecognition_->reset();
    }
};
const Core::ParameterBool IncrementalRecognizer::paramMeshRescoring(
        "mesh-rescoring",
        "",
        false);
const Core::ParameterBool IncrementalRecognizer::paramExpandTransits(
        "expand-transits",
        "",
        true);
const Core::ParameterBool IncrementalRecognizer::paramMeshCombination(
        "mesh-combination",
        "",
        false);
const Core::ParameterBool IncrementalRecognizer::paramForceForwardBackwardLattices(
        "force-forward-backward-lattices",
        "compute new backward lattices even for the extended lattice beam size, and combine them with the main lattices (only with mesh-rescoring)",
        false);
const Core::ParameterInt IncrementalRecognizer::paramRescoreWordEndLimit(
        "rescore-word-end-limit",
        "",
        10000);
const Core::ParameterBool IncrementalRecognizer::paramPronunciationScore(
        "add-pronunication-score",
        "add an extra dimension containing the pronunciation score",
        false);
const Core::ParameterBool IncrementalRecognizer::paramVerboseRefinement(
        "verbose-refinement",
        "print lots of output to the standard-output",
        false);
const Core::ParameterBool IncrementalRecognizer::paramConsiderSentenceBegin(
        "consider-sentence-begin",
        "consider sentence begin token regarding LM score. \
         This must be used when the LM was reversed _correctly_ without a hack that omits the sentence-begin",
        true);
const Core::ParameterBool IncrementalRecognizer::paramCorrectWholeSegment(
        "correct-whole-segment",
        "whether the whole segment should be re-recognized when an error was found",
        false);
const Core::ParameterBool IncrementalRecognizer::paramCorrectForceEqualAlignment(
        "correct-force-equal-alignment",
        "",
        false);
const Core::ParameterBool IncrementalRecognizer::paramCorrectForceEqualScore(
        "correct-force-equal-score",
        "",
        false);
const Core::ParameterBool IncrementalRecognizer::paramCorrectStrictInitial(
        "correct-strict-initial",
        "",
        true);
const Core::ParameterBool IncrementalRecognizer::paramCorrectIncludeNoise(
        "correct-include-noise",
        "",
        false);
const Core::ParameterBool IncrementalRecognizer::paramApplyNonWordClosureFilter(
        "apply-non-word-closure-filter",
        "apply the non word closure filter",
        false);
const Core::ParameterBool IncrementalRecognizer::paramConfidenceScore(
        "add-confidence-score",
        "add an extra dimension containing the confidence score",
        false);
const Core::ParameterFloat IncrementalRecognizer::paramAlpha(
        "alpha",
        "scale dimensions for posterior calculation",
        0.0);
const Core::ParameterFloat IncrementalRecognizer::paramPosteriorPruningThreshold(
        "posterior-pruning-threshold",
        "Prune lattice by posterior (eg. forward-backward-pruning). Values below zero indicate no pruning. Applied after redundancy-removal or nonword-filter.",
        -1);
const Core::ParameterBool IncrementalRecognizer::paramApplyUniqueSentenceAlignmentFilter(
        "apply-redundancy-removal",
        "remove redundancy from lattice",
        false);
const Core::ParameterFloat IncrementalRecognizer::paramScoreTolerance(
        "correct-score-tolerance",
        "consider forward- and backward pass to match exactly when the score difference is less or equal to this value per second (relative to LM scale)",
        0.001);
const Core::ParameterBool IncrementalRecognizer::paramOnlyEnforceMinimumSearchSpace(
        "only-enforce-minimum-search-space",
        "when correct-errors is true, don't really correct errors, but abort as soon as the search space constraints are satisfied",
        false);
const Core::ParameterFloat IncrementalRecognizer::paramMaxRtf(
        "maximum-rtf",
        "maximum rtf which may be accumulated during refinement",
        Core::Type<Score>::max);
const Core::ParameterInt IncrementalRecognizer::paramCacheFrames(
        "cache-frames",
        "for how many frames the emission scorers should be cached (the memory-usage is 4 times the number of emission models per frame)",
        10000);
const Core::ParameterBool IncrementalRecognizer::paramPreCache(
        "precache-all-frames",
        "",
        false);
const Core::ParameterInt IncrementalRecognizer::paramLmContextLength(
        "lm-context-length",
        "length of LM context considered when refining search",
        0);
const Core::ParameterFloat IncrementalRecognizer::paramRelaxPruningFactor(
        "relax-pruning-factor",
        "",
        1.1);
const Core::ParameterFloat IncrementalRecognizer::paramRelaxPruningOffset(
        "relax-pruning-offset",
        "",
        0.5);
const Core::ParameterFloat IncrementalRecognizer::paramLatticeRelaxPruningFactor(
        "lattice-relax-pruning-factor",
        "",
        1.0, 1.0);
const Core::ParameterFloat IncrementalRecognizer::paramLatticeRelaxPruningOffset(
        "lattice-relax-pruning-offset",
        "",
        0, 0);
const Core::ParameterInt IncrementalRecognizer::paramLatticeRelaxPruningInterval(
        "lattice-relax-pruning-interval",
        "number of timeframes over which higher pruning thresholds overlap the context",
        5, 0);
const Core::ParameterFloat IncrementalRecognizer::paramAdaptInitialUpdateRate(
        "decoder-initial-update-rate",
        "If this is 0.0, no adaptation is done (a good target for the initial update rate is for example 0.3).",
        0.3);
const Core::ParameterFloat IncrementalRecognizer::paramAdaptRelaxPruningFactor(
        "decoder-relax-pruning-factor",
        "",
        1.03);
const Core::ParameterFloat IncrementalRecognizer::paramAdaptRelaxPruningOffset(
        "decoder-relax-pruning-offset",
        "relative to lm-scale",
        0.2);
const Core::ParameterInt IncrementalRecognizer::paramAdaptCorrectionRatio(
        "decoder-second-correction-ratio",
        "If this is nonzero, the system will try to adapt relax-pruning-factor so that in X out of 10 cases the second pass resolves all errors",
        0, 0, 10);
const Core::ParameterFloat IncrementalRecognizer::paramAdaptPruningFactor(
        "decoder-adapt-pruning-factor",
        "",
        1.1, 1.005);
const Core::ParameterFloat IncrementalRecognizer::paramMinArcsPerSecond(
        "min-arcs-per-second",
        "minimum number of arcs per second after pruning",
        0, 0);
const Core::ParameterFloat IncrementalRecognizer::paramMaxArcsPerSecond(
        "max-arcs-per-second",
        "maximum number of arcs per second after pruning",
        Core::Type<f32>::max, 1);
const Core::ParameterInt IncrementalRecognizer::paramMaxLatticeRegenerations(
        "max-lattice-regenerations",
        "maximum number of regenerations of the lattice due to min-arcs-per-second",
        20, 1);
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------

class IncrementalRecognizerNode : public Node {
private:
    ModelCombinationRef    mc_;
    IncrementalRecognizer* recognizer_;

public:
    IncrementalRecognizerNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config),
              mc_(),
              recognizer_(0) {}
    virtual ~IncrementalRecognizerNode() {
        delete recognizer_;
    }

    virtual void init(const std::vector<std::string>& arguments) {
        if (connected(0))
            criticalError("Something is connected to port 0. Rescoring is not supported by the refining recognizer.");
        if (!connected(1))
            criticalError("Speech segment at port 1 required");
        if (!Lexicon::us()->isReadOnly())
            warning("Lexicon is not read-only, "
                    "dynamically added/modified lemmas are not considered by the recognizer.");
        AcousticModelRef am = getAm(select("acoustic-model"));
        mc_                 = getModelCombination(config, am, getLm(select("lm")));
        recognizer_         = new IncrementalRecognizer(config, mc_);
    }

    virtual void finalize() {
        recognizer_->reset();
        recognizer_->finalize();
    }

    std::pair<ConstLatticeRef, ConstSegmentRef> buffered_;

    void work() {
        clog() << Core::XmlOpen("layer") + Core::XmlAttribute("name", name);
        buffered_ = recognizer_->recognize();

        if (!buffered_.first) {
            const Bliss::SpeechSegment* segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            recognizer_->newSegment();
            recognizer_->startRecognition(segment);
            buffered_ = recognizer_->recognize();
        }
        clog() << Core::XmlClose("layer");
    }

    virtual ConstSegmentRef sendSegment(Port to) {
        if (!buffered_.second)
            work();
        return buffered_.second;
    }

    virtual ConstLatticeRef sendLattice(Port to) {
        if (!buffered_.first)
            work();
        return buffered_.first;
    }

    virtual void sync() {
        buffered_.first.reset();
        buffered_.second.reset();
    }

    virtual bool blockSync() const {
        return recognizer_->recognitionPending();
    }
};

NodeRef createIncrementalRecognizerNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new IncrementalRecognizerNode(name, config));
}
}  // namespace Flf
