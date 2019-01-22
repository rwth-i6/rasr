#include "PushForwardRescoring.hh"

#include <numeric>
#include <queue>
#include <vector>

#include <Lm/Module.hh>
#include <Math/Utilities.hh>

#include "FlfCore/Basic.hh"
#include "FlfCore/LatticeInternal.hh"
#include "Lexicon.hh"

namespace {
    struct Hypothesis {
        Lm::History  history;
        Flf::Score   seq_score;
        Flf::Score   seq_prospect_score;
        Flf::Score   score;
        unsigned     prev_hyp;
        Fsa::StateId start_state;
        unsigned     arc;
    };

    struct CompareSeqScore {
        bool operator()(Hypothesis const& a, Hypothesis const& b) {
            return a.seq_score < b.seq_score;
        }
    };

    struct CompareSeqProspectScore {
        bool operator()(Hypothesis const& a, Hypothesis const& b) {
            return a.seq_prospect_score < b.seq_prospect_score;
        }
    };

    using SeqScorePriorityQueue      = std::priority_queue<Hypothesis, std::vector<Hypothesis>, CompareSeqScore>;
    using ProspectScorePriorityQueue = std::priority_queue<Hypothesis, std::vector<Hypothesis>, CompareSeqProspectScore>;

    std::vector<Flf::Score> calculate_lookahead(Flf::ConstLatticeRef l, Flf::ConstStateMapRef toposort) {
        std::vector<Flf::Score> lookahead(toposort->maxSid + 1ul, std::numeric_limits<Flf::Score>::infinity());
        lookahead[toposort->back()] = 0.f;
        for (size_t topo_idx = toposort->size() - 1ul; topo_idx > 0ul;) {
            --topo_idx;
            Fsa::StateId current_state = (*toposort)[topo_idx];
            Flf::ConstStateRef s = l->getState(current_state);
            for (Flf::State::const_iterator a = s->begin(); a != s->end(); ++a) {
                Fsa::StateId to        = a->target();
                Flf::Score   arc_score = l->semiring()->project(a->weight());
                lookahead[current_state] = std::min(lookahead[current_state], lookahead[to] + arc_score);
            }
        }
        return lookahead;
    }

    SeqScorePriorityQueue recombine(Core::Ref<Lm::LanguageModel> lm, ProspectScorePriorityQueue& hs, unsigned history_limit) {
        SeqScorePriorityQueue result;
        std::unordered_map<Lm::History, Hypothesis, Lm::History::Hash> recombination;
        while (not hs.empty()) {
            Lm::History recomb_history = hs.top().history;
            if (history_limit > 0) {  // perform pruning based on reduced history
                recomb_history = lm->reducedHistory(recomb_history, history_limit);
            }
            auto iter = recombination.find(recomb_history);
            if (iter != recombination.end() and iter->second.seq_prospect_score > hs.top().seq_prospect_score) {
                iter->second = hs.top();
            }
            else {
                recombination.insert(std::make_pair(recomb_history, hs.top()));
            }
            hs.pop();
        }
        for (auto& kv : recombination) {
            result.push(kv.second);
        }
        return result;
    }
}

namespace Flf {

class ReplaceSingleDimensionLattice : public SlaveLattice {
public:
    typedef SlaveLattice Precursor;

    ReplaceSingleDimensionLattice(ConstLatticeRef l, std::vector<size_t> const& state_offsets, std::vector<Score> const& scores, ScoreId id);
    ~ReplaceSingleDimensionLattice() = default;

    virtual ConstStateRef getState(Fsa::StateId sid) const;
    virtual std::string   describe() const;
private:
    std::vector<size_t> state_offsets_;
    std::vector<Score>  scores_;
    ScoreId             id_;
};

ReplaceSingleDimensionLattice::ReplaceSingleDimensionLattice(ConstLatticeRef l, std::vector<size_t> const& state_offsets, std::vector<Score> const& scores, ScoreId id)
                             : Precursor(l), state_offsets_(state_offsets), scores_(scores), id_(id) {
}

ConstStateRef ReplaceSingleDimensionLattice::getState(Fsa::StateId sid) const {
    require_lt(sid, state_offsets_.size());
    ConstStateRef sr = fsa_->getState(sid);
    State* sp = clone(*fsa_->semiring(), sr);
    sp->setId(sid);
    size_t arc_counter = 0ul;
    for (State::iterator a = sp->begin(); a != sp->end(); ++a) {
        size_t arc_offset = state_offsets_[sid] + arc_counter;
        require_lt(arc_offset, scores_.size());
        a->weight_->set(id_, scores_[arc_offset]);
        arc_counter++;
    }
    return ConstStateRef(sp);
}

std::string ReplaceSingleDimensionLattice::describe() const {
    return Core::form("replaceSingleDimension(%s;dim=%zu)",
                      fsa_->describe().c_str(),
                      id_);
}

// ----------------------------------------------------------------------

const Core::Choice          PushForwardRescorer::rescorerTypeChoice   ("single-best",               singleBestRescoringType,
                                                                       "replacement-approximation", replacementApproximationRescoringType,
                                                                       "traceback-approximation",   tracebackApproximationRescoringType,
                                                                       Core::Choice::endMark());
const Core::ParameterChoice PushForwardRescorer::paramRescorerType    ("rescorer-type", &PushForwardRescorer::rescorerTypeChoice, "what sort of rescoring should be performed", singleBestRescoringType);
const Core::ParameterInt    PushForwardRescorer::paramMaxHypothesis   ("max-hypotheses",    "maximum number of hypotheses per node", 5);
const Core::ParameterFloat  PushForwardRescorer::paramPruningThreshold("pruning-threshold", "pruning threshold for rescoring (relative to lm-scale)", 14.0);
const Core::ParameterInt    PushForwardRescorer::paramHistoryLimit    ("history-limit",     "reduce history to at most this many tokens (0 = no limit)", 0, 0);
const Core::ParameterFloat  PushForwardRescorer::paramLookaheadScale  ("lookahead-scale",   "scale lookahead with this factor", 1.0);


PushForwardRescorer::PushForwardRescorer(Core::Configuration const& config, Core::Ref<Lm::LanguageModel> lm)
                                        : Precursor(config), lm_(lm), rescoring_type_(static_cast<RescorerType>(paramRescorerType(config))), max_hyps_(paramMaxHypothesis(config)),
                                          pruning_threshold_(paramPruningThreshold(config)), history_limit_(paramHistoryLimit(config)), lookahead_scale_(paramLookaheadScale(config)) {
}

ConstLatticeRef PushForwardRescorer::rescore(ConstLatticeRef l, ScoreId id) {
    if (l->initialStateId() == Fsa::InvalidStateId) {
        return l;  // empty lattice
    }

    Core::Ref<const Bliss::LemmaAlphabet>              l_alphabet;
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lp_alphabet;
    switch (Lexicon::us()->alphabetId(l->getInputAlphabet())) {
    case Lexicon::LemmaAlphabetId:
        l_alphabet = Lexicon::us()->lemmaAlphabet();
        break;
    case Lexicon::LemmaPronunciationAlphabetId:
        lp_alphabet = Lexicon::us()->lemmaPronunciationAlphabet();
        break;
    default:
        defect();
    }

    ConstSemiringRef original_semiring = l->semiring();
    Score            original_scale    = original_semiring->scale(id);
    ConstSemiringRef rescaled_semiring = rescaleSemiring(original_semiring, id, 0.0);

    // we need to traverse the automaton in chronological order (preserves topological order)
    ConstStateMapRef toposort = sortChronologically(l);
    require_ne(toposort->maxSid, Fsa::InvalidStateId);
    ConstBoundariesRef boundaries = l->getBoundaries();

    // some statistics
    size_t num_expansions = 0ul;
    size_t total_num_arcs = 0ul;

    // these are our main datastructures
    std::vector<ProspectScorePriorityQueue> all_hyps(toposort->maxSid + 1ul); // list of unexpanded hypotheses for each state
    std::vector<Hypothesis>                 traceback;                        // list of expanded hypotheses

    std::vector<Flf::Score> best_score_per_time(boundaries->time(toposort->back()) + 1, std::numeric_limits<Flf::Score>::infinity());
    std::vector<Flf::Score> lookahead = calculate_lookahead(l, toposort);
    std::transform(lookahead.begin(), lookahead.end(), lookahead.begin(), std::bind(std::multiplies<Flf::Score>(), lookahead_scale_, std::placeholders::_1));

    // insert inital hypothesis
    all_hyps[toposort->front()].push(Hypothesis{ lm_->startHistory(), 0.0, lookahead[toposort->front()], 0.0, 0ul, toposort->front(), 0ul });

    // now we go through all states and expand their hypotheses
    for (size_t topo_idx = 0ul; topo_idx < toposort->size(); topo_idx++) {
        Fsa::StateId           current_state = (*toposort)[topo_idx];
        Speech::TimeframeIndex current_time  = boundaries->time(current_state);
        SeqScorePriorityQueue  hyps          = recombine(lm_, all_hyps[current_state], history_limit_);
        Flf::Score             pruning_limit = best_score_per_time[current_time] + original_scale * pruning_threshold_;

        // expand
        while (not hyps.empty()) {
            // always generate a traceback
            Hypothesis const& hyp = hyps.top();
            unsigned predecessor = traceback.size();
            traceback.push_back(hyp);

            // prune by not expanding
            if (not hyps.size() <= 1 and (hyps.size() > max_hyps_ or hyps.top().seq_prospect_score > pruning_limit)) {
                hyps.pop();
                continue;
            }

            unsigned arc_counter = 0ul;
            ConstStateRef s = l->getState(current_state);
            for (State::const_iterator a = s->begin(); a != s->end(); ++a) {
                Fsa::StateId to       = a->target();
                Fsa::LabelId label_id = a->input();

                Hypothesis new_hyp{ hyp.history, hyp.seq_score, 0.0, 0.0, predecessor, current_state, arc_counter };

                if (label_id != Fsa::Epsilon) {
                    Bliss::Lemma const* lemma = l_alphabet ? l_alphabet->lemma(label_id) : lp_alphabet->lemmaPronunciation(label_id)->lemma();
                    Lm::addLemmaScore(lm_, 1.0, lemma, 1.0, new_hyp.history, new_hyp.score);
                }
                else if (to == toposort->back()) { // word end symbol
                    new_hyp.score = lm_->sentenceEndScore(new_hyp.history);
                }
                else {
                    new_hyp.score = a->weight()->get(id);
                }
                new_hyp.seq_score += original_scale * new_hyp.score + rescaled_semiring->project(a->weight());
                new_hyp.seq_prospect_score =  new_hyp.seq_score + lookahead[to];

                best_score_per_time[boundaries->time(to)] = std::min(best_score_per_time[boundaries->time(to)], new_hyp.seq_prospect_score);
                all_hyps[to].push(new_hyp);
                num_expansions += 1ul;
                arc_counter += 1ul;
            }

            hyps.pop();
        }
    }

    log("num expansions: ") << static_cast<double>(num_expansions) / static_cast<double>(total_num_arcs);

    ConstLatticeRef result;

    // do traceback
    switch (rescoring_type_) {
        case singleBestRescoringType: {
            Flf::StaticBoundariesRef new_boundaries(new Flf::StaticBoundaries());
            Flf::StaticLatticeRef    output_lattice(new Flf::StaticLattice(l->type()));
            output_lattice->setSemiring(original_semiring);
            output_lattice->setBoundaries(new_boundaries);
            output_lattice->setInputAlphabet(l->getInputAlphabet());
            if (l->type() == Fsa::TypeTransducer) {
                output_lattice->setOutputAlphabet(l->getOutputAlphabet());
            }
            output_lattice->addProperties(Fsa::PropertyLinear);
            output_lattice->addProperties(Fsa::PropertyAcyclic);
            output_lattice->setDescription(Core::form("singleBestLatticeRescoring(%s;dim=%zu)", l->describe().c_str(), id));

            ConstStateRef original_final_state = l->getState(toposort->back());
            State* state = output_lattice->newState(original_final_state->tags(), original_final_state->weight());
            new_boundaries->set(state->id(), boundaries->get(original_final_state->id()));

            require_ge(traceback.size(), 1ul);
            size_t hyp_idx = traceback.size() - 1ul;
            while (true) {
                auto& hyp = traceback[hyp_idx];
                ConstStateRef  original_state = l->getState(hyp.start_state);
                Arc const*     original_arc   = original_state->getArc(hyp.arc);
                Flf::ScoresRef new_weight     = original_semiring->clone(original_arc->weight());
                new_weight->set(id, hyp.score);

                State* prev_state = output_lattice->newState(original_state->tags(), original_state->weight());
                prev_state->newArc(state->id(), new_weight, original_arc->input(), original_arc->output());
                new_boundaries->set(prev_state->id(), boundaries->get(original_state->id()));

                state = prev_state;
                if (hyp_idx == hyp.prev_hyp) { // check if we arrived at the first hypothesis
                    break;
                }
                hyp_idx = hyp.prev_hyp;
            }
            output_lattice->setInitialStateId(state->id());

            result = output_lattice;
        } break;
        case replacementApproximationRescoringType: {
            // calculate offsets for each state
            std::vector<size_t> num_arcs(toposort->maxSid + 1ul); // number of arcs for each state
            std::vector<size_t> state_offsets(num_arcs.size(), 0ul);

            for (auto it = toposort->begin(); it != toposort->end(); ++it) {
                num_arcs[*it] = l->getState(*it)->nArcs();
                total_num_arcs += num_arcs[*it];
            }
            std::partial_sum(num_arcs.begin(), num_arcs.end() - 1, state_offsets.begin() + 1); // TODO: replace with inplace std::exclusive_scan once we switch to C++17
            std::vector<Flf::Score> scores(state_offsets.back() + num_arcs.back(), std::numeric_limits<Flf::Score>::infinity());

            std::vector<bool> visited(traceback.size(), false);
            visited.front() = true;
            for (size_t i = traceback.size(); i > 0ul;) {
                --i;
                size_t hyp_index = i;
                auto& hyp = traceback[hyp_index];
                while (not visited[hyp_index]) {
                    visited[hyp_index] = true;
                    size_t offset = state_offsets[hyp.start_state] + hyp.arc;
                    if (Math::isinf(scores[offset])) {
                        scores.at(offset) = hyp.score;
                    }

                    hyp_index = hyp.prev_hyp;
                    hyp = traceback[hyp_index];
                }
            }
            result = ConstLatticeRef(new ReplaceSingleDimensionLattice(l, state_offsets, scores, id));
        } break;
        case tracebackApproximationRescoringType: {
            Flf::StaticBoundariesRef new_boundaries(new Flf::StaticBoundaries());
            Flf::StaticLatticeRef    output_lattice(new Flf::StaticLattice(l->type()));
            output_lattice->setSemiring(original_semiring);
            output_lattice->setBoundaries(new_boundaries);
            output_lattice->setInputAlphabet(l->getInputAlphabet());
            if (l->type() == Fsa::TypeTransducer) {
                output_lattice->setOutputAlphabet(l->getOutputAlphabet());
            }
            output_lattice->addProperties(Fsa::PropertyLinear);
            output_lattice->addProperties(Fsa::PropertyAcyclic);
            output_lattice->setDescription(Core::form("tracebackApproximationLatticeRescoring(%s;dim=%zu)", l->describe().c_str(), id));

            std::vector<Fsa::StateId> end_state_ids(traceback.size(), 0u); // cache end states for easier access later
            for (size_t t = 1ul; t < end_state_ids.size(); t++) {
                end_state_ids[t] = l->getState(traceback[t].start_state)->getArc(traceback[t].arc)->target();
            }

            std::vector<State*> new_end_states(traceback.size(), nullptr);
            std::vector<bool>   visited(traceback.size(), false);

            // create start state
            new_end_states[0ul] = output_lattice->newState();
            new_boundaries->set(new_end_states[0ul]->id(), boundaries->get(l->getState(toposort->front())->id()));
            output_lattice->setInitialStateId(new_end_states[0ul]->id());
            visited.front() = true;

            // create final state
            ConstStateRef original_final_state = l->getState(toposort->back());
            State* final_state = output_lattice->newState(original_final_state->tags(), original_final_state->weight());
            new_boundaries->set(final_state->id(), boundaries->get(original_final_state->id()));

            for (size_t i = traceback.size(); i > 0ul;) {
                --i;
                if (visited[i]) {
                    continue;
                }
                bool pruned_path = end_state_ids[i] != toposort->back();
                if (not pruned_path) {
                    new_end_states[i] = final_state;
                }
                else if (end_state_ids[i] == end_state_ids[i+1] and new_end_states[i+1] != nullptr) {
                    new_end_states[i] = new_end_states[i+1];
                }
                else {
                    hope(false); // should not happen
                    continue;
                }
                for (size_t hyp_index = i; not visited[hyp_index]; hyp_index = traceback[hyp_index].prev_hyp) {
                    auto& hyp = traceback[hyp_index]; 
                    visited[hyp_index] = true;

                    if (new_end_states[hyp.prev_hyp] == nullptr) {
                        new_end_states[hyp.prev_hyp] = output_lattice->newState();
                        new_boundaries->set(new_end_states[hyp.prev_hyp]->id(), boundaries->get(hyp.start_state));
                    }

                    ConstStateRef  original_state = l->getState(hyp.start_state);
                    Arc const*     original_arc   = original_state->getArc(hyp.arc);
                    Flf::ScoresRef new_weight     = original_semiring->clone(original_arc->weight());
                    new_weight->set(id, hyp.score);
                    new_end_states[hyp.prev_hyp]->newArc(new_end_states[hyp_index]->id(), new_weight,
                                                         original_arc->input(), original_arc->output());
                }
            }

            result = output_lattice;
        } break;
        default: {
            defect();
        }
    }

    return result;
}

// ----------------------------------------------------------------------

PushForwardRescoringNode::PushForwardRescoringNode(std::string const& name, Core::Configuration const& config) : Precursor(name, config), rescorer_(nullptr) {
}

void PushForwardRescoringNode::init(const std::vector<std::string> &arguments) {
    auto lm = Lm::Module::instance().createLanguageModel(select("lm"), Lexicon::us());
    if (!lm) {
        criticalError("PushForwardRescoringNode: failed to load language model");
    }
    rescorer_.reset(new PushForwardRescorer(config, lm));
}

ConstLatticeRef PushForwardRescoringNode::rescore(ConstLatticeRef l, ScoreId id) {
    if (!l) {
        return l;
    }
    if (!rescored_lattice_) {
        rescored_lattice_ = rescorer_->rescore(l, id);
    }

    return rescored_lattice_;
}

// ----------------------------------------------------------------------

NodeRef createPushForwardRescoringNode(const std::string &name, const Core::Configuration &config) {
    return NodeRef(new PushForwardRescoringNode(name, config));
}

} // namespace Flf
