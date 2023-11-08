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
#include "TransitionModel.hh"
#include <Core/ProgressIndicator.hh>
#include <Fsa/Dfs.hh>
#include <Fsa/Static.hh>
#include <Math/Utilities.hh>
#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#include <stack>

// implementation details for the TransitionModel::apply function
namespace {

/**
 * Apply TransitionModel to a flat automaton.
 *
 * What is does: The purpose of this algorithm is to add loop and skip
 * transitions to a "flat" automaton.  (Flat meaning that it does not
 * contain loops and skips.)  The emission labels, i.e. the labels
 * that are repeated or skipped are on the input side of the
 * automaton, while the output labels will be unchanged.
 *
 * This can be viewed as a specialized compose algorithm for the
 * time-distortion transducer (left) and a given automaton (right).
 * If you read on, you will discover that considerable care must be
 * taken to creating compact results.
 *
 * How it works: The state space is expanded so that we remember the
 * most recent emission label, this is called "left state" in the
 * following.  "Right state" refers to the corresponding state in the
 * original automaton.  This expansion is necessary to provide the
 * loop transitions.  The representation of the left state is rather
 * verbose.  It consists of a mask stating which kinds of transition
 * are possible, a reference to the state's transition model, and of
 * course the most recent emission label.  In fact only a small number
 * of combinations of the possible values are actually used.  (One
 * could slim down the data structure to represent only the valid
 * combinations.  However priority was given to clarity and
 * maintainability of the code, over the small increase in efficiency.)
 * The function isStateLegitimate() specifies which potential states
 * can be used.  It is good to make these constraints as tight as
 * possible in order to ensure the result automaton does not contain
 * unnecessary states.
 *
 * The most recent emission label may be empty (Fsa::Epsilon).  We
 * call this a "discharged" state.  This happens for three reasons: 1)
 * At the word start no emission label has been consumed.  2) After
 * processing an input epsilon arc.  3) In some situations we
 * deliberately forget the emission label (see below).
 *
 * In the expanded state space, loop transitions are simple to
 * implement.  (In "discharged" states they are not allowed.)
 * Concerning the other (forward, exit and skip) transitions, there is
 * a little twist: When a right state has multiple incoming and
 * outgoing arcs, we choose to first discharge the recent-most
 * emission label by going to an appropriate left state via an epsilon
 * transition.  The alternative would be to avoid the epsilon
 * transition and directly connect to all successor states.  However,
 * in practice this would dramatically increase the total number of
 * arcs needed.  So discharging is the preferable alternative.  The
 * discharged state can be thought of as the state when we have
 * decided to leave the current state, but not yet chosen where to go
 * to.  As mentioned before, the appropriate set of transition weights
 * is recorded, so that we know what to do when we forward or skip
 * from the discharged state.
 *
 * Concerning skips: In general, a skip consists of two transitions:
 * First an epsilon transition goes to an intermediate state that
 * allows a forward only, and then another transition leads to the
 * target state.  In "favorable" situations this is optimized into a
 * single transitions (skip optimization).  If you have read so far,
 * you are certainly able to figure out what these favorable
 * conditions are.
 *
 * As you noticed, there is some freedom in designing the discharge
 * transitions.  It turns out that compact results can be obtained by
 * combining the forward discharge with the intermediate skip states,
 * and to combine exit and loop discharge states.
 *
 * Any disambiguator label is interpreted as a word boundary and is
 * given the following special treatment: No loop transition, since
 * the word boundary cannot be repeated.  No skip transitions: The word
 * boundary cannot be skipped and the final state before the boundary
 * cannot be skipped.  The latter is done for consistency with
 * WordConditionedTreeSearch.
 *
 * Once the state space is constructed as describe above, it is
 * relatively straight forward to figure out, which transition weight
 * (aka time distortion penalty or TDP) must be applied to which arc.
 * Unfortunately the current scheme is not able to distinguish phone-1
 * from phone-2 states.  This will require additional state space
 * expansion by counting repetitive emission labels.  Alternatively,
 * and probably simpler, we change the labels to allow the distinction
 * between phone-1 and phone-2.
 *
 * Todo:
 * - An even better condition for discharging would be:
 *   (number of DIFFERENT incoming arc labels > 1) && (number of outgoing arcs > 1)
 *   Presently it is
 *   (number of incoming arcs > 1) && (number of outgoing arcs > 1).
 * - Implement proper distinction between between phone-1 and phone-2 states.
 * - Reconsider the transition model.  (see discussion in Wiki)
 * - It might be nice to convert this to lazy evaluation (should
 *   be straight forward).
 **/

struct ApplicatorState {
    using Mask = u8;
    using StateType = Am::TransitionModel::StateType;

    Mask              mask;     /**< bitmask of allowed transitions (1 << TransitionType) */
    Fsa::LabelId      emission; /**< recent most emission */
    StateType         weights;  /**< transition model to apply */

    static const Mask allowLoop    = (1 << Am::StateTransitionModel::loop);
    static const Mask allowForward = (1 << Am::StateTransitionModel::forward);
    static const Mask allowSkip    = (1 << Am::StateTransitionModel::skip);
    static const Mask allowExit    = (1 << Am::StateTransitionModel::exit);
    static const Mask isFinal      = (1 << Am::StateTransitionModel::nTransitionTypes);

    static const StateType noWeights = (StateType)-1;

    Fsa::StateId right;

    ApplicatorState() : mask(), emission(Fsa::Epsilon), weights(noWeights), right() {
    }

    ApplicatorState(Mask m, Fsa::LabelId e, StateType t, Fsa::StateId r)
            : mask(m), emission(e), weights(t), right(r) {}

    struct Equality {
        bool operator()(ApplicatorState const& ll, ApplicatorState const& rr) const {
            return (ll.emission == rr.emission) && (ll.mask == rr.mask) && (ll.weights == rr.weights) && (ll.right == rr.right);
        }
    };
    struct Hash {
        size_t operator()(ApplicatorState const& s) const {
            return (((((size_t(s.right) << 12) ^ size_t(s.emission)) << 2) ^ size_t(s.weights)) << 2) ^ size_t(0x0f ^ s.mask);
        }
    };
};

struct ApplicatorStateWithContext : public ApplicatorState {
    Fsa::LabelId context;

    ApplicatorStateWithContext() : ApplicatorState(), context(Fsa::Epsilon) {
    }

    ApplicatorStateWithContext(Mask m, Fsa::LabelId e,  Fsa::LabelId c, StateType t, Fsa::StateId r)
            : ApplicatorState(m, e, t, r), context(c) {}

};

class Applicator {
public:
    Applicator() : 
              alphabet_(),
              silenceLabel_(Fsa::InvalidLabelId),
              applyExitTransitionToFinalStates_(false) {}

    virtual Fsa::ConstAutomatonRef apply(Fsa::ConstAutomatonRef in) = 0;

    Fsa::ConstAlphabetRef alphabet_;
    Fsa::LabelId          silenceLabel_;
    bool                  applyExitTransitionToFinalStates_;

    virtual ~Applicator() = default;
};

template<class AppState>
class AbstractApplicator : public Applicator {
protected:
    friend class TransitionModel;
    using StateType = Am::TransitionModel::StateType;

    // -----------------------------------------------------------------------
    class StateDegrees : public Fsa::DfsState {
    public:
        enum Direction {
            incoming = 0,
            outgoing = 2
        };
        enum Type {
            emitting       = 0,
            epsilon        = 4,
            disambiguating = 8
        };
        enum {
            none = 0x00,
            one  = 0x01,
            many = 0x03
        };

        class Degree {
            u16 flags_;

        public:
            Degree()
                    : flags_(0) {}

            void add(Direction direction, Type type) {
                u32 shift = direction + type;
                if (flags_ & (one << shift))
                    flags_ |= (many << shift);
                else
                    flags_ |= (one << shift);
            };

            u8 operator()(Direction direction, Type type) const {
                u32 shift = direction + type;
                return (flags_ >> shift) & 0x03;
            };
        };

    private:
        Fsa::ConstAlphabetRef alphabet_;
        Core::Vector<Degree>  degrees_;

    public:
        void exploreArc(Fsa::ConstStateRef from, const Fsa::Arc& arc) {
            degrees_.grow(from->id(), Degree());
            degrees_.grow(arc.target(), Degree());
            Type type = (arc.input() == Fsa::Epsilon) ? epsilon
                                                      : (alphabet_->isDisambiguator(arc.input())) ? disambiguating
                                                                                                  : emitting;
            degrees_[from->id()].add(outgoing, type);
            degrees_[arc.target()].add(incoming, type);
        }

        virtual void exploreTreeArc(Fsa::ConstStateRef from, const Fsa::Arc& arc) {
            exploreArc(from, arc);
        }
        virtual void exploreNonTreeArc(Fsa::ConstStateRef from, const Fsa::Arc& arc) {
            exploreArc(from, arc);
        }

        StateDegrees(Fsa::ConstAutomatonRef ff, Fsa::ConstAlphabetRef aa)
                : Fsa::DfsState(ff), alphabet_(aa) {}

        const Degree& operator[](Fsa::StateId ii) const {
            return degrees_[ii];
        }
    };

    struct StackItem : AppState {
        Fsa::StateRef result;
        StackItem(AppState const& state, Fsa::StateRef _result)
                : AppState(state), result(_result) {}
    };

    typedef std::stack<StackItem>                                                                            StateStack;
    typedef std::unordered_map<AppState, Fsa::StateId, typename AppState::Hash, typename AppState::Equality> StateMap;

    // -----------------------------------------------------------------------

    Am::TransitionModel const&      transitionModel_;
    Fsa::ConstAutomatonRef          in_;
    Core::Ref<Fsa::StaticAutomaton> t_;

    StateDegrees* rightStateDegrees_;
    StateStack    todo_;
    StateMap      states_;

    // -----------------------------------------------------------------------

    bool isStateLegitimate(AppState const& s) const;

    Fsa::StateId getStateId(AppState const& s);

    Core::Ref<Fsa::State> createState(AppState const& s);

    Fsa::Weight weight(StateType st, Am::StateTransitionModel::TransitionType type) const;

    virtual AppState createStateFromCurrentState(AppState const& current, typename AppState::Mask m, Fsa::LabelId e, StateType t, Fsa::StateId r) = 0;

    virtual Fsa::Weight getWeightForForward(AppState const& current) = 0;

    virtual Fsa::Weight getWeightForSkip(AppState const& current) = 0;

    /**
     * \todo proper distinction between phone-1 and phone-2 states
     */
    StateType stateType(Fsa::LabelId emission) const;

    virtual void doEpsilon(StackItem const& current, Fsa::Arc const* ra);
    virtual void doForward(StackItem const& current, Fsa::Arc const* ra);
    virtual void doLoop(StackItem const& current);
    virtual void doSkip(StackItem const& current, Fsa::Arc const* ra);
    virtual void doExit(StackItem const& current, Fsa::Arc const* ra);
    virtual void doDischarge(StackItem const& current);

public:
    AbstractApplicator(const Am::TransitionModel& tm)
            : Applicator(),
              transitionModel_(tm) {
              }

    virtual Fsa::ConstAutomatonRef apply(Fsa::ConstAutomatonRef in);
};

template<class AppState>
bool AbstractApplicator<AppState>::isStateLegitimate(AppState const& s) const {
    if (alphabet_->isDisambiguator(s.emission))
        return false;
    // word start state
    if ((s.mask == (AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal)) &&
        (s.weights == Am::TransitionModel::entryM1) &&
        (s.emission == Fsa::Epsilon))
        return true;
    // normal emitting state
    if ((s.mask == (AppState::allowForward | AppState::allowLoop | AppState::allowSkip | AppState::allowExit | AppState::isFinal)) &&
        (s.weights >= Am::TransitionModel::silence) &&
        (s.emission != Fsa::Epsilon))
        return true;
    // discharged forward and intermediate skip state
    if ((s.mask == (AppState::allowForward)) &&
        (s.weights == AppState::noWeights) &&
        (s.emission == Fsa::Epsilon))
        return true;
    // discharged skip and exit state
    if ((s.mask == (AppState::allowSkip | AppState::allowExit)) &&
        (s.weights != AppState::noWeights) &&
        (s.emission == Fsa::Epsilon))
        return true;
    // post-epsilon state
    if ((s.mask == (AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal)) &&
        (s.weights >= Am::TransitionModel::silence) &&
        (s.emission == Fsa::Epsilon))
        return true;
    return false;
}

template <class AppState>
Fsa::StateId AbstractApplicator<AppState>::getStateId(AppState const& s) {
    typename StateMap::iterator i = states_.find(s);
    if (i == states_.end()) {
        verify(isStateLegitimate(s));
        StackItem si(s, createState(s));
        i = states_.insert(std::make_pair(s, si.result->id())).first;
        todo_.push(si);
    }
    return i->second;
}

template <class AppState>
Core::Ref<Fsa::State> AbstractApplicator<AppState>::createState(AppState const& s) {
    Core::Ref<Fsa::State> result = Core::ref(t_->newState());
    Fsa::ConstStateRef    sr;
    bool                  isFinal = (s.mask & AppState::isFinal) && ((sr = in_->getState(s.right))->isFinal());
    if (isFinal) {
        Fsa::Weight w(sr->weight_);
        if (applyExitTransitionToFinalStates_)
            w = t_->semiring()->extend(w, weight(s.weights, Am::StateTransitionModel::exit));
        result->setFinal(w);
    }
    return result;
}

template <class AppState>
Fsa::Weight AbstractApplicator<AppState>::weight(const StateType st, Am::StateTransitionModel::TransitionType type) const {
    if (st == AppState::noWeights) {
        return Fsa::Weight(0.0);
    }
    else {
        return Fsa::Weight((*transitionModel_[st])[type]);
    }
}

/**
 * \todo proper distinction between phone-1 and phone-2 states
 */
template <class AppState>
typename AbstractApplicator<AppState>::StateType AbstractApplicator<AppState>::stateType(Fsa::LabelId emission) const {
    if (emission == silenceLabel_) {
        return Am::TransitionModel::silence;
    }
    else {
        if (dynamic_cast<const Am::CartTransitionModel*>(&transitionModel_)) {
            return dynamic_cast<const Am::EmissionAlphabet*>(alphabet_.get()) ? (StateType)emission : transitionModel_.classifyIndex(emission);
        }
        else {
            return Am::TransitionModel::phone0;
        }
    }
}

template <class AppState>
void AbstractApplicator<AppState>::doEpsilon(const StackItem& current, const Fsa::Arc* ra) {
    require(ra->input() == Fsa::Epsilon);
    AppState newState = createStateFromCurrentState(current,
                                      current.mask & ~(AppState::allowLoop),
                                      Fsa::Epsilon,
                                      current.weights,
                                      ra->target());
    current.result->newArc(getStateId(newState),
                           ra->weight(),
                           Fsa::Epsilon, ra->output());
}

template <class AppState>
void AbstractApplicator<AppState>::doForward(const StackItem& current, const Fsa::Arc* ra) {
    require(!alphabet_->isDisambiguator(ra->input()));
    require(ra->input() != Fsa::Epsilon);

    Fsa::Weight w = getWeightForForward(current);
    AppState newState = createStateFromCurrentState(current,
                                                    AppState::allowLoop | AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal,
                                                    ra->input(),
                                                    stateType(ra->input()),
                                                    ra->target());
    current.result->newArc(getStateId(newState),
                           t_->semiring()->extend(ra->weight(), w),
                           ra->input(), ra->output());

}

template <class AppState>
void AbstractApplicator<AppState>::doLoop(const StackItem& current) {
    require(current.emission != Fsa::Epsilon);
    current.result->newArc(current.result->id(),
                           weight(current.weights, Am::StateTransitionModel::loop),
                           current.emission, Fsa::Epsilon);
}

template <class AppState>
void AbstractApplicator<AppState>::doSkip(const StackItem& current, const Fsa::Arc* ra) {
    require(!alphabet_->isDisambiguator(ra->input()));
    require(ra->input() != Fsa::Epsilon);

    typename AbstractApplicator<AppState>::StateDegrees::Degree targetDegree       = (*rightStateDegrees_)[ra->target()];
    bool                 wouldSkipToDeadEnd = ((targetDegree(StateDegrees::outgoing, StateDegrees::emitting) +
                                                targetDegree(StateDegrees::outgoing, StateDegrees::epsilon)) == 0);
    if (wouldSkipToDeadEnd)
        return;

    Fsa::Weight w = getWeightForSkip(current);

    if (t_->semiring()->compare(w, t_->semiring()->max()) >= 0) {
        return;
    }

    bool isEligbleForSkipOptimization = ((targetDegree(StateDegrees::outgoing, StateDegrees::disambiguating) == 0) &&
                                         (targetDegree(StateDegrees::outgoing, StateDegrees::epsilon) == 0) &&
                                         (targetDegree(StateDegrees::outgoing, StateDegrees::emitting) == 1));

    Fsa::ConstStateRef rat;
    const Fsa::Arc*    ras = 0;
    if (isEligbleForSkipOptimization) {
        rat = in_->getState(ra->target());
        verify(rat->nArcs() == 1);
        ras = &*rat->begin();
        if (ras->output() != Fsa::Epsilon)
            isEligbleForSkipOptimization = false;
    }

    Fsa::Arc* ca = current.result->newArc();
    ca->weight_  = t_->semiring()->extend(ra->weight(), w);
    if (isEligbleForSkipOptimization) {
        verify(ras->input() != Fsa::Epsilon);
        verify(!alphabet_->isDisambiguator(ras->input()));
        AppState newState = createStateFromCurrentState(current,
                                                        AppState::allowLoop | AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal,
                                                        ras->input(),
                                                        stateType(ras->input()),
                                                        ras->target());
        ca->target_ = getStateId(newState);
        ca->weight_ = t_->semiring()->extend(ca->weight_, ras->weight());
        ca->input_  = ras->input();
    }
    else {
        AppState newState = createStateFromCurrentState(current,
                                                        AppState::allowForward,
                                                        Fsa::Epsilon,
                                                        AppState::noWeights,
                                                        ra->target());
        ca->target_ = getStateId(newState);
        ca->input_  = Fsa::Epsilon;
    }
    ca->output_ = ra->output();
}

template <class AppState>
void AbstractApplicator<AppState>::doExit(const StackItem& current, const Fsa::Arc* ra) {
    require(alphabet_->isDisambiguator(ra->input()));
    verify(!applyExitTransitionToFinalStates_);
    AppState newState = createStateFromCurrentState(current,
                                      AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal,
                                      Fsa::Epsilon,
                                      Am::TransitionModel::entryM1,
                                      ra->target());
    current.result->newArc(getStateId(newState),
                           t_->semiring()->extend(ra->weight(), weight(current.weights, Am::StateTransitionModel::exit)),
                           ra->input(), ra->output());

}

template <class AppState>
void AbstractApplicator<AppState>::doDischarge(const StackItem& current) {
    AppState newState = createStateFromCurrentState(current,
                                      AppState::allowForward,
                                      Fsa::Epsilon,
                                      AppState::noWeights,
                                      current.right);
    current.result->newArc(getStateId(newState),
                           weight(current.weights, Am::StateTransitionModel::forward),
                           Fsa::Epsilon, Fsa::Epsilon);

    newState = createStateFromCurrentState(current,
                                      AppState::allowSkip | AppState::allowExit,
                                      Fsa::Epsilon,
                                      current.weights,
                                      current.right);
    current.result->newArc(getStateId(newState),
                           t_->semiring()->one(),
                           Fsa::Epsilon, Fsa::Epsilon);
}


template <class AppState>
Fsa::ConstAutomatonRef AbstractApplicator<AppState>::apply(Fsa::ConstAutomatonRef input) {
    require(alphabet_);
    in_ = input;

    Core::ProgressIndicator pi("applying transition model", "states");
    rightStateDegrees_ = new StateDegrees(in_, alphabet_);
    rightStateDegrees_->dfs(&pi);

    t_ = Core::ref(new Fsa::StaticAutomaton);
    t_->setType(input->type());
    t_->setSemiring(input->semiring());
    t_->setInputAlphabet(input->getInputAlphabet());
    t_->setOutputAlphabet(input->getOutputAlphabet());

    AppState initialState = createStateFromCurrentState(AppState(),
                                      AppState::allowForward | AppState::allowSkip | AppState::allowExit | AppState::isFinal,
                                      Fsa::Epsilon,
                                      Am::TransitionModel::entryM1,
                                      in_->initialStateId());
    Fsa::StateId initial = getStateId(initialState);
    t_->setInitialStateId(initial);
    pi.start();
    while (!todo_.empty()) {
        const StackItem current(todo_.top());
        todo_.pop();
        Fsa::ConstStateRef currentRight = in_->getState(current.right);

        typename StateDegrees::Degree degree = (*rightStateDegrees_)[current.right];

        bool shouldDischarge = (degree(StateDegrees::incoming, StateDegrees::emitting) == StateDegrees::many) &&
                               ((degree(StateDegrees::outgoing, StateDegrees::emitting) == StateDegrees::many) ||
                                (degree(StateDegrees::outgoing, StateDegrees::disambiguating) == StateDegrees::many));

        //	std::cerr << Core::form("expand: mask=%x\temission=%d\tweights=%d\tright=%zd\n", current.mask, current.emission, current.weights, current.right); // DEBUG

        if (current.mask & AppState::allowLoop) {
            doLoop(current);
        }
        if (current.emission != Fsa::Epsilon && shouldDischarge) {
            doDischarge(current);
        }
        else {
            for (Fsa::State::const_iterator ra = currentRight->begin(); ra != currentRight->end(); ++ra) {
                if (ra->input() == Fsa::Epsilon) {
                    doEpsilon(current, &*ra);
                }
                else if (alphabet_->isDisambiguator(ra->input())) {
                    if (current.mask & AppState::allowExit)
                        doExit(current, &*ra);
                }
                else {
                    if (current.mask & AppState::allowForward)
                        doForward(current, &*ra);
                    if (current.mask & AppState::allowSkip)
                        doSkip(current, &*ra);
                }
            }
        }
        pi.notify(t_->size());
    }
    pi.finish();
    delete rightStateDegrees_;
    in_.reset();
    Fsa::removeInvalidArcsInPlace(t_);
    Fsa::trimInPlace(t_);
    Fsa::ConstAutomatonRef result = t_;
    t_.reset();
    return result;
}

class LegacyApplicator : public AbstractApplicator<ApplicatorState> {
public:
    LegacyApplicator(const Am::TransitionModel& tm)
            : AbstractApplicator<ApplicatorState>(tm) {}

protected:
    virtual ApplicatorState createStateFromCurrentState(ApplicatorState const& current, ApplicatorState::Mask m, Fsa::LabelId e, StateType t, Fsa::StateId r) {
        return ApplicatorState(m, e, t, r);
    }

    virtual Fsa::Weight getWeightForForward(ApplicatorState const& current) {
        Fsa::Weight w = weight(current.weights, Am::StateTransitionModel::forward);
        return w;
    }

    virtual Fsa::Weight getWeightForSkip(ApplicatorState const& current) {
        Fsa::Weight w = weight(current.weights, Am::StateTransitionModel::skip);
        return w;
    }

};

class CorrectedApplicator : public AbstractApplicator<ApplicatorStateWithContext> {
protected:
    virtual ApplicatorStateWithContext createStateFromCurrentState(ApplicatorStateWithContext const& current, ApplicatorStateWithContext::Mask m, Fsa::LabelId e, StateType t, Fsa::StateId r) {
        return ApplicatorStateWithContext(m, e, current.emission, t, r);
    }

    virtual Fsa::Weight getWeightForForward(ApplicatorStateWithContext const& current) {
        Fsa::Weight w;
        if (current.emission == Fsa::Epsilon && current.context == Fsa::Epsilon) {
            // handle how you leave the virtual start state
            w = t_->semiring()->one();
        }
        else {
            if (current.emission == Fsa::Epsilon ) {
                w =  weight(stateType(current.context), Am::StateTransitionModel::forward);
            }
            else {
                w =  weight(current.weights, Am::StateTransitionModel::forward);
            }
        }
        return w;
    }

    virtual Fsa::Weight getWeightForSkip(ApplicatorStateWithContext const& current) {
        Fsa::Weight w;
        if (current.emission == Fsa::Epsilon) {
            w =  weight(stateType(current.context), Am::StateTransitionModel::skip);
        }
        else {
            w =  weight(current.weights, Am::StateTransitionModel::skip);
        }
        return w;
    }

public:
    CorrectedApplicator(Am::TransitionModel const& tm)
            : AbstractApplicator(tm) {}
};

}  // namespace


using namespace Am;

// ===========================================================================
const Core::ParameterFloat StateTransitionModel::paramScores[nTransitionTypes] = {
        Core::ParameterFloat("loop", "negative logarithm of probability for loop transition", 3.0),
        Core::ParameterFloat("forward", "negative logarithm of probability for forward transition", 0.0),
        Core::ParameterFloat("skip", "negative logarithm of probability for skip transition", 3.0),
        Core::ParameterFloat("exit", "negative logarithm of probability for word end transition", 0.0),
};

StateTransitionModel::StateTransitionModel(const Core::Configuration& c)
        : Core::Configurable(c) {
    clear();
}

void StateTransitionModel::load(Score scale) {
    for (int i = 0; i < nTransitionTypes; ++i) {
        tdps_[i] = scale * paramScores[i](config);
        tdps_[i] = Core::clip(tdps_[i]);
        ensure(!Math::isnan(tdps_[i]));
    }
}

void StateTransitionModel::load(Score scale, const Mm::Scales& scores) {
    for (int i = 0; i < nTransitionTypes; ++i) {
        tdps_[i] = scale * scores[i];
        tdps_[i] = Core::clip(tdps_[i]);
        ensure(!Math::isnan(tdps_[i]));
    }
}

void StateTransitionModel::clear() {
    for (int i = 0; i < nTransitionTypes; ++i) {
        tdps_[i] = 0;
    }
}

StateTransitionModel& StateTransitionModel::operator+=(const StateTransitionModel& m) {
    for (int i = 0; i < nTransitionTypes; ++i) {
        tdps_[i] = Core::clip(tdps_[i] + m.tdps_[i]);
        ensure(!Math::isnan(tdps_[i]));
    }
    return *this;
}

void StateTransitionModel::getDependencies(Core::DependencySet& dependencies) const {
    std::string value;
    for (int i = 0; i < nTransitionTypes; ++i) {
        value += Core::form("%s=%f", paramScores[i].name().c_str(), tdps_[i]);
        if (i + 1 < nTransitionTypes)
            value += "; ";
    }
    dependencies.add(name(), value);
}

void StateTransitionModel::dump(Core::XmlWriter& o) const {
    o << Core::XmlOpen(name()) + Core::XmlAttribute(paramScores[0].name(), tdps_[0]) + Core::XmlAttribute(paramScores[1].name(), tdps_[1]) + Core::XmlAttribute(paramScores[2].name(), tdps_[2]) + Core::XmlAttribute(paramScores[3].name(), tdps_[3]);
    o << Core::XmlClose(name());
}

// ===========================================================================
Am::TransitionModel::TransitionModel(const Core::Configuration& c)
        : Core::Component(c) {}

Am::TransitionModel::~TransitionModel() {
    for (u32 i = 0; i < transitionModels_.size(); ++i) {
        delete transitionModels_[i];
    }
}

void Am::TransitionModel::dump(Core::XmlWriter& o) const {
    for (u32 t = 0; t < transitionModels_.size(); ++t) {
        if (transitionModels_[t]) {
            transitionModels_[t]->dump(o);
        }
    }
}

bool Am::TransitionModel::load(Mc::Scale scale) {
    const std::string tdpValuesFile = paramTdpValuesFile(config);
    if (tdpValuesFile.empty()) {
        for (u32 i = 0; i < transitionModels_.size(); ++i) {
            if (transitionModels_[i]) {
                transitionModels_[i]->load(scale);
            }
        }
    }
    else {
        criticalError("cannot load tdp values from file. Module MM_ADVANCED is not available");
    }
    correct();
    return true;
}

void Am::TransitionModel::clear() {
    for (u32 i = 0; i < transitionModels_.size(); ++i) {
        if (transitionModels_[i]) {
            transitionModels_[i]->clear();
        }
    }
}

Am::TransitionModel& Am::TransitionModel::operator+=(const Am::TransitionModel& m) {
    for (u32 i = 0; i < transitionModels_.size(); ++i)
        (*transitionModels_[i]) += (*m.transitionModels_[i]);
    return *this;
}

bool Am::TransitionModel::correct() {
    bool result = true;
    for (int t = entryM1; t <= entryM2; ++t) {
        if (Core::isSignificantlyLess((*transitionModels_[t])[StateTransitionModel::loop],
                                      Core::Type<StateTransitionModel::Score>::max)) {
            result = false;
            warning("Changing loop probability for entry state to zero, was: %f",
                    exp(-(*transitionModels_[t])[StateTransitionModel::loop]));
            transitionModels_[t]->set(StateTransitionModel::loop, Core::Type<StateTransitionModel::Score>::max);
        }
    }
    return result;
}

void Am::TransitionModel::getDependencies(Core::DependencySet& dependencies) const {
    Core::DependencySet d;
    for (u32 i = 0; i < transitionModels_.size(); ++i) {
        if (transitionModels_[i]) {
            transitionModels_[i]->getDependencies(d);
        }
    }
    dependencies.add(name(), d);
}

Fsa::ConstAutomatonRef Am::TransitionModel::apply(Fsa::ConstAutomatonRef in,
                                                  Fsa::LabelId           silenceLabel,
                                                  bool                   applyExitTransitionToFinalStates) const {
    std::unique_ptr<Applicator> ap;

    Core::Choice::Value appChoice = Am::TransitionModel::paramApplicatorType(config);
    if (appChoice == Core::Choice::IllegalValue)
        criticalError("unknwon transition applicator type.");

    switch(Am::TransitionModel::ApplicatorType(appChoice)) {
        case Am::TransitionModel::ApplicatorType::LegacyApplicator:
            ap.reset(new LegacyApplicator(*this));
            break;
        case Am::TransitionModel::ApplicatorType::CorrectedApplicator:
            ap.reset(new CorrectedApplicator(*this));
            break;
        default:
            defect();
    }

    ap->alphabet_                         = in->getInputAlphabet();
    ap->silenceLabel_                     = silenceLabel;
    ap->applyExitTransitionToFinalStates_ = applyExitTransitionToFinalStates;

    return ap->apply(in);
}

// ===========================================================================
GlobalTransitionModel::GlobalTransitionModel(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {
    transitionModels_.resize(nStateTypes);
    transitionModels_[entryM1] = new StateTransitionModel(select("entry-m1"));
    transitionModels_[entryM2] = new StateTransitionModel(select("entry-m2"));
    transitionModels_[silence] = new StateTransitionModel(select("silence"));
    transitionModels_[phone0]  = new StateTransitionModel(select("state-0"));
    transitionModels_[phone1]  = new StateTransitionModel(select("state-1"));
}

// ===========================================================================
const Core::ParameterStringVector NonWordAwareTransitionModel::paramNonWordPhones(
        "nonword-phones",
        "Non-word (noise) phone symbols with separate tdps. Wildcards can be used at boundaries to select multiple phonemes.",
        ",");

NonWordAwareTransitionModel::NonWordAwareTransitionModel(const Core::Configuration& c,
                                                         ClassicStateModelRef       stateModel)
        : Core::Component(c),
          Precursor(c),
          stateModel_(stateModel) {
    std::vector<std::string>   nonWordPhones = paramNonWordPhones(config);
    Bliss::PhonemeInventoryRef pi            = stateModel_->phonology().getPhonemeInventory();
    const AllophoneAlphabet&   allophones    = stateModel_->allophoneAlphabet();
    for (std::vector<std::string>::const_iterator s = nonWordPhones.begin(); s != nonWordPhones.end(); ++s) {
        std::vector<std::string>     selection;
        std::set<Bliss::Phoneme::Id> selectedPhones = pi->parseSelection(*s);
        for (std::set<Bliss::Phoneme::Id>::iterator phoneIt = selectedPhones.begin(); phoneIt != selectedPhones.end(); ++phoneIt) {
            Allophone allo(*phoneIt, Allophone::isInitialPhone | Allophone::isFinalPhone);
            if (allophones.isSilence(allo))
                continue;
            const Allophone* allophone = allophones.allophone(allophones.index(allo));
            log("using nonword tdps for allophone %s", allophones.toString(*allophone).c_str());
            const ClassicHmmTopology* hmmTopology = stateModel_->hmmTopology(allophone);
            const u32                 nStates     = hmmTopology->nPhoneStates();
            for (u32 state = 0; state < nStates; ++state) {
                const AllophoneStateIndex si = stateModel_->allophoneStateAlphabet().index(allophone, state);
                nonWordStates_.insert(si);
            }
        }
    }
    if (nonWordStates_.empty()) {
        warning("no non-word phone defined");
    }
    transitionModels_.push_back(new StateTransitionModel(select("nonword-0")));
    transitionModels_.push_back(new StateTransitionModel(select("nonword-1")));
}

// ===========================================================================
CartTransitionModel::CartTransitionModel(const Core::Configuration& c,
                                         ClassicStateModelRef       stateModel)
        : Core::Component(c),
          Precursor(c),
          nSubStates_(1) {
    stateTying_ = ClassicStateTying::createClassicStateTyingRef(select("state-tying"),
                                                                stateModel);
    if (stateTying_->hasFatalErrors()) {
        criticalError("failed to initialize state tying.");
        stateTying_.reset();
    }

    verify(nSubStates_ == 1);

    transitionModels_.resize(silence + (stateTying_->nClasses() * nSubStates_), 0);
    transitionModels_[entryM1] = new StateTransitionModel(select("entry-m1"));
    transitionModels_[entryM2] = new StateTransitionModel(select("entry-m2"));
    transitionModels_[silence] = new StateTransitionModel(select("silence"));
    // silence is assumed to have index=0 but is ignored
    for (u32 s = 1; s < stateTying_->nClasses(); ++s) {
        verify(!transitionModels_[silence + s]);
        const std::string selection    = Core::form("state-%d-0", s);
        transitionModels_[silence + s] = new StateTransitionModel(select(selection));
    }
}

// ===========================================================================
Core::Choice Am::TransitionModel::choiceTyingType(
        "global",             static_cast<int>(Am::TransitionModel::TyingType::GlobalTransitionModel),
        "global-and-nonword", static_cast<int>(Am::TransitionModel::TyingType::NonWordAwareTransitionModel),
        "cart",               static_cast<int>(Am::TransitionModel::TyingType::CartTransitionModel),

        Core::Choice::endMark());

Core::ParameterChoice Am::TransitionModel::paramTyingType(
        "tying-type",
        &choiceTyingType,
        "type of tying scheme",
        static_cast<int>(Am::TransitionModel::TyingType::GlobalTransitionModel));

Core::Choice Am::TransitionModel::choiceApplicatorType(
        "legacy",    static_cast<int>(Am::TransitionModel::ApplicatorType::LegacyApplicator),
        "corrected", static_cast<int>(Am::TransitionModel::ApplicatorType::CorrectedApplicator),
         Core::Choice::endMark());

Core::ParameterChoice Am::TransitionModel::paramApplicatorType(
        "applicator-type",
        &choiceApplicatorType,
        "The applicator used for adding weights on the FSA."
        "The LegacyType applicator has a buggy behavior, namely silence.exit = silence.forward - phone?.forward due to epsilon arcs",
        static_cast<int>(Am::TransitionModel::ApplicatorType::LegacyApplicator));

/**
 * This solution is supported because the parameter
 * mechanism cannot efficiently handle a large number
 * of transition models, e.g. one for each CART label!
 */
Core::ParameterString Am::TransitionModel::paramTdpValuesFile(
        "file",
        "file with tdp values, overwrites paramScores()",
        "");

Am::TransitionModel* Am::TransitionModel::createTransitionModel(const Core::Configuration& configuration,
                                                                ClassicStateModelRef       stateModel) {

    Core::Choice::Value transChoice = TransitionModel::paramTyingType(configuration);

    switch (Am::TransitionModel::TyingType(transChoice)) {
        case Am::TransitionModel::TyingType::GlobalTransitionModel:
            return new GlobalTransitionModel(configuration);
            break;
        case Am::TransitionModel::TyingType::NonWordAwareTransitionModel:
            return new NonWordAwareTransitionModel(configuration, stateModel);
            break;
        case Am::TransitionModel::TyingType::CartTransitionModel:
            CartTransitionModel* result = new CartTransitionModel(configuration, stateModel);
            return result;
            break;
    }

    return 0;
}

// ===========================================================================
ScaledTransitionModel::ScaledTransitionModel(const Core::Configuration& c,
                                             ClassicStateModelRef       stateModel)
        : Core::Component(c), Mc::Component(c), transitionModel_(0) {
    transitionModel_ = TransitionModel::createTransitionModel(c, stateModel);
}

ScaledTransitionModel::~ScaledTransitionModel() {
    delete transitionModel_;
}

// ===========================================================================
CombinedTransitionModel::CombinedTransitionModel(const Core::Configuration&                           c,
                                                 const std::vector<Core::Ref<ScaledTransitionModel>>& transitionModels,
                                                 ClassicStateModelRef                                 stateModel)
        : Core::Component(c),
          Precursor(c, stateModel),
          transitionModels_(transitionModels) {
    for (u32 i = 0; i < transitionModels_.size(); ++i)
        transitionModels_[i]->setParentScale(scale());
}

CombinedTransitionModel::~CombinedTransitionModel() {}

bool CombinedTransitionModel::load() {
    transitionModel_->clear();
    bool result = true;
    for (u32 i = 0; i < transitionModels_.size(); ++i) {
        if (transitionModels_[i]->load())
            (*this) += (*transitionModels_[i]);
        else
            result = false;
    }
    transitionModel_->correct();
    return result;
}

void CombinedTransitionModel::distributeScaleUpdate(const Mc::ScaleUpdate& scaleUpdate) {
    transitionModel_->clear();
    for (u32 i = 0; i < transitionModels_.size(); ++i) {
        transitionModels_[i]->updateScales(scaleUpdate);
        (*this) += (*transitionModels_[i]);
    }
    transitionModel_->correct();
}

