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
#include "Sort.hh"
#include "Static.hh"
#include "Types.hh"

namespace Fsa {

class Search {
private:
    typedef f32           Score;
    typedef u32           Time;
    typedef StateId       Trace;
    static const StateTag StateTagHasOutput = StateTagUser;

    class FirstBestTracer : public StaticAutomaton {
    public:
        Trace init(const StaticAutomaton* fsa) {
            clear();
            setType(TypeAcceptor);
            setSemiring(TropicalSemiring);
            setInputAlphabet(fsa->getOutputAlphabet());
            State* sp   = newState(StateTagFinal);
            sp->weight_ = fsa->semiring()->one();
            return sp->id();
        }

        void setTime(Time time) {
        }

        /**
         * add:
         * called upon passing a token along an arc with a new output label.
         **/
        Trace add(Trace trace, LabelId output, Score score) {
            Score diffScore = score - Score(fastState(trace)->weight_);
            // mark states that already have an arc with output label
            State* sp = newState(StateTagHasOutput);
            //std::cout << "add " << trace << " (" << sp->id() << ")" << std::endl;
            sp->weight_ = Weight(score);  // temporary potential
            sp->newArc(trace, Weight(diffScore), output);
            return sp->id();
        }

        /**
         * update:
         * Called upon passing a token to a final state. This is usually the
         * point where the tracer should finalize that trace. It's either
         * recombined or a new one created.
         **/
        Trace update(Trace trace, Trace recombineTrace, Score score) {
            State* sp        = fastState(trace);
            Score  diffScore = score - Score(sp->weight_);
            if ((sp->hasTags(StateTagHasOutput)) || ((sp->nArcs() == 1) && (sp->begin()->input() == Epsilon))) {
                sp->setTags(sp->tags() & ~StateTagHasOutput);
                sp->weight_ = Weight(score);
                for (State::iterator a = sp->begin(); a != sp->end(); ++a)
                    a->weight_ = Weight(Score(a->weight()) + diffScore);
            }
            else {
                sp          = newState();
                sp->weight_ = Weight(score);
                sp->newArc(trace, Weight(diffScore), Epsilon);
                trace = sp->id();
            }
            return trace;
        }
    };

    /**
     * update:
     * - optimize silences
     **/
    class LatticeTracer : public FirstBestTracer {  // assumes non-minimized search network
    private:
        Time time_;

    public:
        void setTime(Time time) {
            time_ = time;
        }
        Trace update(Trace trace, Trace recombineTrace, Score score) {
            require(time_ < Core::Type<Time>::max);
            if (recombineTrace == InvalidStateId)
                return FirstBestTracer::update(trace, recombineTrace, score);
            Score        diffScore = score - Score(fastState(trace)->weight_);
            State*       sp        = fastState(recombineTrace);
            const State* _sp       = fastState(trace);
            if ((_sp->hasTags(StateTagHasOutput)) || ((sp->nArcs() == 1) && (sp->begin()->input() == Epsilon))) {
                for (State::const_iterator a = _sp->begin(); a != _sp->end(); ++a)
                    sp->newArc(a->target(), Weight(f32(Score(a->weight()) + diffScore)), a->input());
            }
            else
                sp->newArc(trace, Weight(f32(diffScore)), Epsilon);
            if (Score(sp->weight_) < score)
                sp->weight_ = Weight(score);
            return recombineTrace;
        }
    };

    typedef FirstBestTracer Tracer;

    struct Token {
        StateId state_;
        Score   score_;
        Trace   trace_;
        Token() {}
        Token(StateId state, Score score, Trace trace)
                : state_(state), score_(score), trace_(trace) {}
    };
    typedef Core::Vector<Token> ActiveTokens;

    struct ExpandedToken {
        Score score_;
        Trace trace_;
        ExpandedToken()
                : score_(Core::Type<Score>::max), trace_(InvalidStateId) {}
        ExpandedToken(Score score, Trace trace)
                : score_(score), trace_(trace) {}
    };
    typedef Core::Vector<ExpandedToken> ExpandedTokens;

    const StaticAutomaton* fsa_;
    ConstSemiringRef       semiring_;
    Time                   time_;
    ActiveTokens           tokens_;
    Trace                  bestFinalTrace_;
    ExpandedTokens         expandedTokens_;
    Core::Ref<Tracer>      tracer_;
    Core::Statistics<u32>  statesBeforePruning_, statesAfterPruning_;
    Score                  threshold_, finalThreshold_, minFinalScore_, minScore_, minThreshold_;

private:
    enum ExpandType { ExpandTypeNone    = 0,
                      ExpandTypeUpdated = 1,
                      ExpandTypeNew     = 2 };
    ExpandType expandToken(Score score, Trace trace, StateId target, LabelId output) {
        if (output != Epsilon)
            trace = tracer_->add(trace, output, score);
        if (fsa_->fastState(target)->isFinal()) {
            Score finalScore = score + Score(fsa_->fastState(target)->weight_);
            if (finalScore < minFinalScore_) {
                minFinalScore_  = finalScore;
                bestFinalTrace_ = trace;
            }
            if (finalScore < minFinalScore_ + 130)
                trace = tracer_->update(trace, expandedTokens_[target].trace_, score);
        }
        if (score < minThreshold_) {
            if (score < minScore_) {
                minScore_     = score;
                minThreshold_ = minScore_ + threshold_;
            }
            if (score < expandedTokens_[target].score_) {
                ExpandType type =
                        (expandedTokens_[target].score_ >= Core::Type<Score>::max ? ExpandTypeNew : ExpandTypeUpdated);
                expandedTokens_[target].score_ = score;
                expandedTokens_[target].trace_ = trace;
                return type;
            }
        }
        return ExpandTypeNone;
    }

    // doesn't work for negative weight epsilon-loops
    void expandEpsilonArcs(Core::Vector<StateId>& expandedStateIds) {
        while (!tokens_.empty()) {
            Core::Vector<StateId> newExpandedStateIds;
            for (ActiveTokens::const_iterator t = tokens_.begin(); t != tokens_.end(); ++t) {
                const State* sp = fsa_->fastState(t->state_);
                if (t->score_ < expandedTokens_[t->state_].score_) {
                    if (expandedTokens_[t->state_].score_ >= Core::Type<Score>::max)
                        expandedStateIds.push_back(t->state_);
                    expandedTokens_[t->state_].score_ = t->score_;
                    expandedTokens_[t->state_].trace_ = t->trace_;
                }
                for (State::const_iterator a = sp->begin(); (a != sp->end()) && (a->input() == Epsilon); ++a) {
                    Score      score = t->score_ + Score(a->weight());
                    ExpandType type  = expandToken(score, t->trace_, a->target(), a->output());
                    if (type >= ExpandTypeUpdated) {
                        newExpandedStateIds.push_back(a->target());
                        if (type == ExpandTypeNew)
                            expandedStateIds.push_back(a->target());
                    }
                }
            }
            ActiveTokens reevaluateTokens;
            for (Core::Vector<StateId>::const_iterator s = newExpandedStateIds.begin(); s != newExpandedStateIds.end(); ++s)
                reevaluateTokens.push_back(Token(*s, expandedTokens_[*s].score_, expandedTokens_[*s].trace_));
            tokens_.swap(reevaluateTokens);
        }
    }

    template<class Scorer>
    void expand(const Scorer& scorer, Core::Vector<StateId>& expandedStateIds) {
        minFinalScore_ = minScore_ = minThreshold_ = Core::Type<Score>::max;
        for (ActiveTokens::const_iterator t = tokens_.begin(); t != tokens_.end(); ++t) {
            const State*          sp = fsa_->fastState(t->state_);
            State::const_iterator a  = sp->begin();
            for (; (a != sp->end()) && (a->input() == Epsilon); ++a)
                ;
            for (; a != sp->end(); ++a) {
                Score score = t->score_ + Score(a->weight()) + scorer->score(a->input());
                if (expandToken(score, t->trace_, a->target(), a->output()) == ExpandTypeNew)
                    expandedStateIds.push_back(a->target());
            }
        }
    }

    void prune(const Core::Vector<StateId>& expandedStateIds) {
        statesBeforePruning_ += expandedStateIds.size();
        tokens_.clear();
        for (Core::Vector<StateId>::const_iterator s = expandedStateIds.begin(); s != expandedStateIds.end(); ++s) {
            ExpandedToken& token = expandedTokens_[*s];
            if (token.score_ < minThreshold_)
                tokens_.push_back(Token(*s, token.score_, token.trace_));
            token.score_ = Core::Type<Score>::max;
            token.trace_ = InvalidStateId;
        }
        statesAfterPruning_ += tokens_.size();
    }

    void addInitialStateToTracer() {
        State* sp = tracer_->newState();
        tracer_->setInitialStateId(sp->id());
        for (ActiveTokens::const_iterator t = tokens_.begin(); t != tokens_.end(); ++t)
            sp->newArc(t->trace_, tracer_->semiring()->one(), Epsilon);
    }

    void purge() {
        addInitialStateToTracer();
        Fsa::removeNonAccessibleStates(tracer_);
        StateMap mapping;
        tracer_->compact(mapping);
        for (ActiveTokens::iterator t = tokens_.begin(); t != tokens_.end(); ++t)
            t->trace_ = mapping[t->trace_];
        if (bestFinalTrace_ != InvalidStateId)
            bestFinalTrace_ = mapping[bestFinalTrace_];
    }

    Trace getBestFinalTrace() const {
        return bestFinalTrace_;
    }

    Core::Vector<ConstAutomatonRef> getFinalAutomata() const {
        Core::Vector<ConstAutomatonRef> finalAutomata;
        for (ActiveTokens::const_iterator t = tokens_.begin(); t != tokens_.end(); ++t)
            if (fsa_->fastState(t->state_)->isFinal())
                finalAutomata.push_back(partial(ConstAutomatonRef(tracer_.get()), t->trace_));
        return finalAutomata;
    }

public:
    Search(const StaticAutomaton* fsa, Score threshold)
            : fsa_(fsa), statesBeforePruning_("states before pruning"), statesAfterPruning_("states after pruning") {
        // TODO: test for epsilon cycles?

        // require(fsa->hasProperty(PropertySortedByInput));

        semiring_  = fsa->semiring();
        threshold_ = threshold;
        tracer_    = Core::Ref<Tracer>(new Tracer());
        tokens_.resize(fsa_->size());
        expandedTokens_.resize(fsa_->size());
        restart();
    };

    void restart() {
        time_ = 0;
        tokens_.clear();
        ConstStateRef initial = fsa_->getState(fsa_->initialStateId());
        if (initial) {
            bestFinalTrace_ = InvalidStateId;
            if (tracer_)
                bestFinalTrace_ = tracer_->init(fsa_);
            tokens_.push_back(Token(initial->id(), 0.0, bestFinalTrace_));
        }
        statesBeforePruning_.clear();
        statesAfterPruning_.clear();
        bestFinalTrace_ = InvalidStateId;
        minFinalScore_ = minScore_ = minThreshold_ = Core::Type<Score>::max;
        Core::Vector<StateId> expandedStateIds;
        expandEpsilonArcs(expandedStateIds);
        prune(expandedStateIds);
    }

    template<class Scorer>
    void feed(const Scorer& scorer) {
        tracer_->setTime(time_);
        bestFinalTrace_ = InvalidStateId;
        Core::Vector<StateId> expandedStateIds;
        expand(scorer, expandedStateIds);
        prune(expandedStateIds);
        expandedStateIds.clear();
        expandEpsilonArcs(expandedStateIds);
        prune(expandedStateIds);
        if (++time_ % 10 == 0)
            purge();
    }

    ConstAutomatonRef getPartialTraceback() {
        Trace tracebackStateId = getBestFinalTrace();
        if (tracebackStateId == InvalidStateId) {
            std::cerr << "cannot create partial traceback: no active final trace." << std::endl;
            return ConstAutomatonRef();
        }
        return transpose(partial(tracer_, tracebackStateId));
    }

    ConstAutomatonRef getCurrentWordLattice() {
        Core::Vector<ConstAutomatonRef> finalAutomata = getFinalAutomata();
        if (finalAutomata.size() == 0) {
            std::cerr << "cannot create word lattice: no active final trace." << std::endl;
            return ConstAutomatonRef();
        }
        return transpose(unite(finalAutomata));
    }

    u32 nActiveTokens() const {
        return tokens_.size();
    }
    const Core::Statistics<u32>& statesBeforePruning() const {
        return statesBeforePruning_;
    }
    const Core::Statistics<u32>& statesAfterPruning() const {
        return statesAfterPruning_;
    }
};

}  // namespace Fsa

#endif  // _FSA_SEARCH_HH
