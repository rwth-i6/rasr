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
#include <unordered_map>

#include <Core/Vector.hh>
#include "Arithmetic.hh"
#include "Basic.hh"
#include "Best.hh"
#include "Cache.hh"
#include "Compose.hh"
#include "Dfs.hh"
#include "Levenshtein.hh"
#include "Minimize.hh"
#include "Project.hh"
#include "Sort.hh"
#include "RemoveEpsilons.hh"

namespace Fsa {

#if 1
    class LevenshteinAutomaton : public Automaton {
    private:
        ConstStateRef state_;
        ConstAlphabetRef inputAlphabet_, outputAlphabet_;
    public:
        LevenshteinAutomaton(ConstAlphabetRef ref, ConstAlphabetRef test, f32 delCost, f32 insCost, f32 subCost, f32 corCost) :
            inputAlphabet_(ref), outputAlphabet_(test) {
            setProperties(PropertyStorage, PropertyStorage);

            AlphabetMapping mapping;
            mapAlphabet(inputAlphabet_, outputAlphabet_, mapping, false);

            State *sp = new State(0, StateTagFinal, Weight(0.0));
            for (Alphabet::const_iterator i = inputAlphabet_->begin(); i != inputAlphabet_->end(); ++i) {
                if (mapping[LabelId(i)] != InvalidLabelId)
                    sp->newArc(0, Weight(corCost), LabelId(i), mapping[LabelId(i)]); // match
                sp->newArc(0, Weight(delCost), LabelId(i), Epsilon); // deletion
                for (Alphabet::const_iterator j = outputAlphabet_->begin(); j != outputAlphabet_->end(); ++j)
                    if (mapping[LabelId(i)] != j) sp->newArc(0, Weight(subCost), LabelId(i), LabelId(j)); // substitution
            }
            for (Alphabet::const_iterator i = outputAlphabet_->begin(); i != outputAlphabet_->end(); ++i)
                sp->newArc(0, Weight(insCost), Epsilon, LabelId(i)); // insertion
            state_ = ConstStateRef(sp);
        }
        virtual ~LevenshteinAutomaton() {}

        virtual Type type() const { return TypeTransducer; }
        virtual ConstSemiringRef semiring() const { return TropicalSemiring; }
        virtual StateId initialStateId() const { return state_->id(); }
        virtual ConstAlphabetRef getInputAlphabet() const { return inputAlphabet_; }
        virtual ConstAlphabetRef getOutputAlphabet() const { return outputAlphabet_; }
        virtual ConstStateRef getState(StateId s) const { return state_; }
        virtual void releaseState(StateId s) const {}
        virtual std::string describe() const { return "LevenshteinAutomaton"; }
    };

    ConstAutomatonRef levenshtein(ConstAutomatonRef ref, ConstAutomatonRef test, f32 delCost, f32 insCost, f32 subCost, f32 corCost) {
        LevenshteinAutomaton *tmp = new LevenshteinAutomaton(ref->getOutputAlphabet(), test->getInputAlphabet(), delCost, insCost, subCost, corCost);
        //return ConstAutomatonRef(tmp);
        return composeMatching(composeMatching(ref, ConstAutomatonRef(tmp)), test);
    }
#else
    class IntersectionAutomaton : public Automaton {
    private:
        ConstAutomatonRef fl_;
        ConstAutomatonRef fr_;

        typedef std::pair<StateId, StateId> Context;
        struct ContextHash
        {
            size_t operator() (const Context &c) const {
                return (2239 * c.first + c.second);
            }
        };

        struct ContextEqual
        {
            bool operator() (const Context &c1, const Context &c2) const {
                return ((c1.first == c2.first) && (c1.second == c2.second));
            }
        };

        mutable Core::Vector<Context> contexts_;
        mutable std::unordered_map<Context, StateId, ContextHash, ContextEqual> states_;
    private:
        LabelId unseen() const { return LastLabelId; }
        StateId stateId(const Context &c) const {
            if (states_.find(c) == states_.end()) {
                states_[c] = states_.size() - 1;
                contexts_.grow(states_[c], Context(InvalidStateId, InvalidStateId));
                contexts_[states_[c]] = c;
            }
            return states_[c];
        }
        const Context& context(StateId s) const {
            verify(s < contexts_.size());
            return contexts_[s];
        }
        State::const_reverse_iterator unseenIterator(ConstStateRef sp) const {
            for (State::const_reverse_iterator it = sp->rbegin(); it != sp->rend(); ++it)
                if (it->input() == unseen())
                    return it;
            return sp->rend();
        }
    public:
        IntersectionAutomaton(ConstAutomatonRef fl, ConstAutomatonRef fr);
        virtual ~IntersectionAutomaton() {}

        virtual Type type() const { return fr_->type(); }
        virtual ConstSemiringRef semiring() const { return fr_->semiring(); }
        virtual ConstAlphabetRef getInputAlphabet() const {
            return fl_->getInputAlphabet();
        }
        virtual ConstAlphabetRef getOutputAlphabet() const {
            return fr_->getInputAlphabet();
        }
        virtual StateId initialStateId() const {
            if (fl_->initialStateId() != InvalidStateId &&
                fr_->initialStateId() != InvalidStateId)
                return stateId(Context(fl_->initialStateId(),
                                       fr_->initialStateId()));
            return InvalidStateId;
        }
        virtual ConstStateRef getState(StateId s) const;
        virtual std::string describe() const { return "IntersectionAutomaton"; }
    };

    IntersectionAutomaton::IntersectionAutomaton(ConstAutomatonRef fl, ConstAutomatonRef fr) :
        fl_(sort(cache(minimize(determinize(mapInput(fl, fr->getInputAlphabet()))))), SortTypeByInput),
        fr_(sort(cache(removeEpsilons(fr)), SortTypeByInput))
    {
        require(fl_->type() == TypeAcceptor && fr_->type() == TypeAcceptor);
    }

    ConstStateRef IntersectionAutomaton::getState(StateId s) const
    {
        if (s != InvalidStateId) {
            const Context &c = context(s);
            verify(c.first != InvalidStateId && c.second != InvalidStateId);
            ConstStateRef sl = fl_->getState(c.first);
            ConstStateRef sr = fr_->getState(c.second);
            State *sp = new State(s, sl->tags() & sr->tags(), sl->weight_);
            State::const_reverse_iterator unseenIt = unseenIterator(sl);
            for (State::const_iterator al = sl->begin(), ar = sr->begin(); ar != sr->end(); ++ar) {
                while (al != sl->end() && al->input() < ar->input()) ++al;
                if (al != sl->end() && al->input() == ar->input())
                    sp->newArc(stateId(Context(al->target(), ar->target())),
                               al->weight(), ar->input());
                if (unseenIt != sl->rend())
                    sp->newArc(stateId(Context(unseenIt->target(), ar->target())),
                               unseenIt->weight(), ar->input());
            }
            return ConstStateRef(sp);
        }
        return ConstStateRef();
    }

    ConstAutomatonRef intersect(ConstAutomatonRef fl, ConstAutomatonRef fr) {
        return ConstAutomatonRef(new IntersectionAutomaton(fl, fr));
    }

    class LevenshteinAutomaton : public SlaveAutomaton
    {
    private:
        f32 delCost_;
        f32 insCost_;
        f32 subCost_;
        f32 corCost_;
    private:
        LabelId unseen() const { return LastLabelId; }
    public:
        LevenshteinAutomaton(ConstAutomatonRef f, f32 delCost, f32 insCost, f32 subCost, f32 corCost) :
            SlaveAutomaton(f), delCost_(delCost), insCost_(insCost), subCost_(subCost), corCost_(corCost) {}
        virtual ~LevenshteinAutomaton() {}

        ConstStateRef getState(StateId s) const {
            ConstStateRef _sp = fsa_->getState(s);
            State *sp = new State(s, _sp->tags(), _sp->weight_);
            for (State::const_iterator a = _sp->begin(); a != _sp->end(); ++ a) {
                require(a->input() != Epsilon);
                //deletion
                sp->newArc(a->target(), Weight(delCost_), Epsilon);
                //substitution
                sp->newArc(a->target(), Weight(subCost_), unseen());
                //correct
                sp->newArc(a->target(), Weight(corCost_), a->input());
            }
            //insertion
            sp->newArc(s, Weight(insCost_), unseen());
            return ConstStateRef(sp);
        }
        virtual std::string describe() const { return "LevenshteinAutomaton"; }
    };

    ConstAutomatonRef levenshtein(ConstAutomatonRef ref, ConstAutomatonRef test, f32 delCost, f32 insCost, f32 subCost, f32 corCost) {
        // pre-generate fullLevenshtein?
        ConstAutomatonRef levenshtein =
            cache(
                minimize(
                    determinize(
                        removeEpsilons(
                            ConstAutomatonRef(
                                new LevenshteinAutomaton(
                                    ref, delCost, insCost, subCost, corCost))))));
        ConstAutomatonRef intersected =
            minimize(
                determinize(
                    cache(
                        intersect(
                            levenshtein,
                            multiply(
                                test,
                                Weight(0.0))))));
        return intersected;
    }
#endif

    class LevenshteinDfsState : public DfsState {
    private:
        LevenshteinInfo info_;
    public:
        LevenshteinDfsState(ConstAutomatonRef f) : DfsState(f) {
            info_.del_ = info_.ins_ = info_.sub_ = info_.total_ = 0;
        }
        virtual void discoverState(ConstStateRef sp) {
            for (State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
                if (a->input() == Epsilon) info_.del_++;
                else {
                    if (a->output() == Epsilon) info_.ins_++;
                    else if (a->input() != a->output()) info_.sub_++;
                    info_.total_++;
                }
            }
        }
        LevenshteinInfo info() const { return info_; }
    };

    LevenshteinInfo levenshteinInfo(ConstAutomatonRef levensh) {
        LevenshteinDfsState s(best(levensh));
        s.dfs();
        return s.info();
    }

} // namespace Fsa
