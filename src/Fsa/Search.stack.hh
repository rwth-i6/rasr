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
#ifndef _FSA_SEARCH_HH
#define _FSA_SEARCH_HH

#include <Core/Application.hh>
#include <Core/Statistics.hh>
#include <Core/Vector.hh>
#include "Automaton.hh"
#include "Basic.hh"
#include "Output.hh"
#include "Rational.hh"
#include "Stack.hh"
#include "Static.hh"
#include "Types.hh"

namespace Fsa {

/*
 * TODO:
 * - add time information to lattices
 * - collect garbage:
 *   - purge trace automaton
 * - histogram pruning?
 */
class Search {
private:
    typedef f32     Score;
    typedef u32     Time;
    typedef StateId Trace;

    class Slice {
        friend class Search;

    public:
        static const StateTag StateTagTraceMerge = StateTagUser;

    private:
        const StaticAutomaton* fsa_;
        LabelId                inputEpsilon_;
        LabelId                outputEpsilon_;
        ConstSemiringRef       semiring_;

        struct Token {
            StateId state_;
            Score   score_;
            Trace   trace_;
            Token() {}
            Token(StateId state, Score score, Trace trace)
                    : state_(state), score_(score), trace_(trace) {}
        };
        typedef Stack<Token> TokenSet;

        struct ExpandedToken {
            Score score_;
            Trace trace_;
            ExpandedToken()
                    : score_(Core::Type<Score>::max) {}
            ExpandedToken(Score score, Trace trace)
                    : score_(score), trace_(trace) {}
        };
        typedef Core::Vector<ExpandedToken> ExpandedTokens;

        TokenSet              tokens_;
        Trace                 bestFinalTrace_;
        StaticAutomaton*      trace_;
        ExpandedTokens&       expandedTokens_;
        Score                 threshold_, finalThreshold_, minFinalScore_;
        Core::Statistics<u32> statesBeforePruning_, statesAfterPruning_;

    public:
        Slice(const StaticAutomaton* fsa, ExpandedTokens& expandedTokens, StaticAutomaton* trace,
              Score threshold, Score finalThreshold)
                : fsa_(fsa), inputEpsilon_(fsa->getInputAlphabet()->epsilon()), outputEpsilon_(fsa->getOutputAlphabet()->epsilon()), semiring_(fsa->semiring()), bestFinalTrace_(InvalidStateId), trace_(trace), expandedTokens_(expandedTokens), threshold_(threshold), finalThreshold_(finalThreshold), statesBeforePruning_("states before pruning"), statesAfterPruning_("states after pruning") {}

        void start() {
            tokens_.clear();
            ConstStateRef initial = fsa_->getState(fsa_->initialStateId());
            if (initial) {
                bestFinalTrace_ = InvalidStateId;
                if (trace_)
                    bestFinalTrace_ = trace_->newState(StateTagFinal)->id();
                tokens_.push(Token(initial->id(), 0.0, bestFinalTrace_));
            }
            statesBeforePruning_.clear();
            statesAfterPruning_.clear();
        }

        template<class Scorer>
        Score expand(Slice& next, const Scorer& scorer, Time time, Core::Vector<StateId>& expandedTokens) {
            Score minScore = Core::Type<Score>::max, minThreshold = Core::Type<Score>::max;
            next.bestFinalTrace_ = InvalidStateId;
            Score minFinalScore  = Core::Type<Score>::max;
            while (!tokens_.isEmpty()) {
                Token        t  = tokens_.pop();
                const State* sp = fsa_->fastState(t.state_);
                for (State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
                    Score score = t.score_ + Score(a->weight());
                    if (a->input() != inputEpsilon_)
                        score += scorer->score(a->input());
                    Trace trace = t.trace_;
                    if (a->output() != outputEpsilon_) {
                        Score  diffScore    = score - Score(trace_->fastState(trace)->weight_);
                        State* traceState   = trace_->newState();
                        traceState->weight_ = Weight(score);  // temporary potential
                        traceState->newArc(trace, Weight(diffScore), a->output());
                        trace = traceState->id();
                    }
                    if (fsa_->fastState(a->target())->isFinal()) {
                        Score finalScore = score + Score(fsa_->fastState(a->target())->weight_);
                        if (finalScore < minFinalScore) {
                            if (finalScore < minFinalScore) {
                                minFinalScore        = finalScore;
                                next.bestFinalTrace_ = trace;
                            }
                            State* traceState   = trace_->fastState(trace);
                            traceState->weight_ = Weight(score);
                            Score diffScore     = score - Score(traceState->weight_);
                            for (State::iterator a = traceState->begin(); a != traceState->end(); ++a)
                                a->weight_ = Weight(Score(a->weight_) + diffScore);
                        }
                    }
                    if (a->input_ != inputEpsilon_) {
                        if (score < minThreshold) {
                            if (score < minScore) {
                                minScore     = score;
                                minThreshold = minScore + threshold_;
                            }
                            if (score < expandedTokens_[a->target()].score_) {
                                if (expandedTokens_[a->target()].score_ >= Core::Type<Score>::max)
                                    expandedTokens.push_back(a->target());
                                expandedTokens_[a->target()].score_ = score;
                                expandedTokens_[a->target()].trace_ = trace;
                            }
                        }
                    }
                    else
                        tokens_.push(Token(a->target(), score, trace));
                }
            }
            return minThreshold;
        }

        void prune(Slice& next, const Core::Vector<StateId>& expandedTokens, Score minThreshold) {
            next.statesBeforePruning_ = statesBeforePruning_;
            next.statesBeforePruning_ += expandedTokens.size();
            next.tokens_.clear();
            for (Vector<StateId>::const_iterator i = expandedTokens.begin(); i != expandedTokens.end(); ++i) {
                if (expandedTokens_[*i].score_ < minThreshold)
                    next.tokens_.push(Token(*i, expandedTokens_[*i].score_, expandedTokens_[*i].trace_));
                expandedTokens_[*i].score_ = Core::Type<Score>::max;
            }
            next.statesAfterPruning_ = statesAfterPruning_;
            next.statesAfterPruning_ += next.tokens_.size();
        }

        template<class Scorer>
        void pass(Slice& next, const Scorer& scorer, Time time) {
            Core::Vector<StateId> expandedTokens;
            Score                 minThreshold = expand(next, scorer, time, expandedTokens);
            prune(next, expandedTokens, minThreshold);
        }

        Trace bestFinalTrace() const {
            return bestFinalTrace_;
        }
        const Core::Statistics<u32>& statesBeforePruning() const {
            return statesBeforePruning_;
        }
        const Core::Statistics<u32>& statesAfterPruning() const {
            return statesAfterPruning_;
        }
    };

    const StaticAutomaton* fsa_;
    Time                   time_;
    StaticAutomaton*       trace_;
    ConstAutomatonRef      traceRef_;
    Slice::ExpandedTokens  expandedTokens_;
    Slice                  slice0_, slice1_;
    Slice *                old_, *new_;

public:
    Search(const StaticAutomaton* fsa, Score threshold)
            : fsa_(fsa), trace_(new StaticAutomaton()), traceRef_(trace_), expandedTokens_(fsa->size()), slice0_(fsa_, expandedTokens_, trace_, threshold, 190), slice1_(fsa_, expandedTokens_, trace_, threshold, 190) {
        restart();
    };
    void restart() {
        time_ = 0;
        old_  = &slice0_;
        new_  = &slice1_;
        trace_->clear();
        trace_->setType(TypeAcceptor);
        trace_->setSemiring(TropicalSemiring);
        trace_->setInputAlphabet(fsa_->getOutputAlphabet());
        old_->start();
    }
    template<class Scorer>
    void feed(const Scorer& scorer) {
        old_->pass(*new_, scorer, time_++);
        std::swap(old_, new_);
    }
    ConstAutomatonRef getPartialTraceback() {
        return ConstAutomatonRef();
    }
    ConstAutomatonRef getCurrentWordLattice() {
        Trace tracebackStateId = old_->bestFinalTrace();
        if (tracebackStateId == InvalidStateId) {
            std::cerr << "cannot create word lattice: no active final trace." << std::endl;
            return ConstAutomatonRef();
        }
        return transpose(partial(traceRef_, tracebackStateId));
    }
    const Core::Statistics<u32>& statesBeforePruning() const {
        return old_->statesBeforePruning();
    }
    const Core::Statistics<u32>& statesAfterPruning() const {
        return old_->statesAfterPruning();
    }
};

}  // namespace Fsa

#endif  // _FSA_SEARCH_HH
