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
#include "AllophoneStateFsaExporter.hh"

#include <limits>

#include <Fsa/Basic.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Project.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Math/Utilities.hh>
#include <Speech/Types.hh>

namespace {

static void apply_state_map(std::vector<Fsa::StateId> const&             state_map,
                            std::vector<Fsa::StateId>&                   states,
                            std::vector<std::pair<Fsa::StateId, float>>& final_states,
                            std::vector<Edge>&                           edges) {
    for (size_t s = 0ul; s < states.size(); s++) {
        states[s] = state_map[states[s]];
    }
    for (size_t f = 0ul; f < final_states.size(); f++) {
        final_states[f].first = state_map[final_states[f].first];
    }
    for (Edge& e : edges) {
        e.from = state_map[e.from];
        e.to   = state_map[e.to];
    }
}

static bool cmp_edges(Edge const& a, Edge const& b) {
    Fsa::StateId adiff = (a.to - a.from);
    Fsa::StateId bdiff = (b.to - b.from);
    if (adiff == bdiff) {
        if (a.to == b.to) {
            return a.from < b.from;
        }
        return a.to < b.to;
    }
    else {
        return adiff < bdiff;
    }
}

static void toposort(std::vector<Fsa::StateId>&                   states,
                     std::vector<std::pair<Fsa::StateId, float>>& final_states,
                     std::vector<Edge>&                           edges) {
    std::vector<Fsa::StateId> state_map(states.size());

    std::vector<size_t> in_count(states.size(), 0ul);
    for (Edge const& e : edges) {
        if (e.to != e.from) {
            in_count[e.to] += 1ul;
        }
    }

    Fsa::StateId                        next_id = 0u;
    std::vector<size_t>::const_iterator it;
    while ((it = std::find(in_count.begin(), in_count.end(), 0ul)) != in_count.end()) {
        Fsa::StateId state = static_cast<Fsa::StateId>(it - in_count.begin());
        state_map[state]   = next_id++;
        // this could be more efficient, but I am too lazy now
        for (Edge const& e : edges) {
            if (e.from == state and e.to != e.from) {
                in_count[e.to]--;
            }
        }
        in_count[state] = std::numeric_limits<size_t>::max();
    }

    apply_state_map(state_map, states, final_states, edges);

    std::sort(edges.begin(), edges.end(), &cmp_edges);
}

static void filter_edges(std::vector<Edge>& edges) {
    size_t cur = 0ul;
    for (size_t i = 1ul; i < edges.size(); i++) {
        if (edges[cur].from == edges[i].from && edges[cur].to == edges[i].to && edges[cur].emission_idx == edges[i].emission_idx) {
            edges[cur].weight = Math::scoreSum(edges[cur].weight, edges[i].weight);
        }
        else {
            edges[++cur] = edges[i];
        }
    }
    edges.resize(cur + 1);
}

static void make_single_final_state(std::vector<Fsa::StateId>& states, std::vector<std::pair<Fsa::StateId, float>>& final_states, std::vector<Edge>& edges) {
    if (final_states.size() == 1 && final_states.front().first == states.back() && final_states.front().second == 0) {
        return;  // nothing to do
    }

    Fsa::StateId new_final = static_cast<Fsa::StateId>(states.size());
    states.push_back(new_final);
    size_t old_edges_size = edges.size();
    for (size_t e = 0ul; e < old_edges_size; e++) {
        for (std::pair<Fsa::StateId, float>& final_state : final_states) {
            if (final_state.first != edges[e].to)
                continue;
            Edge new_edge = edges[e];
            new_edge.to   = new_final;
            new_edge.weight += final_state.second;
            edges.push_back(new_edge);
            break;
        }
    }

    final_states.resize(1ul);
    final_states.front() = std::make_pair(new_final, 0.0f);
}

u32 getStateDepth(Fsa::StateId sId, std::vector<u32>& stateDepth, Core::Ref<const Fsa::StaticAutomaton> fsa) 
{
    verify( sId < stateDepth.size() );
    if ( stateDepth[sId] == Core::Type<u32>::max ) 
    {
        const Fsa::State* state = fsa->fastState(sId);
        for (Fsa::State::const_iterator arc=state->begin(); arc!=state->end(); ++arc)
        {
            Fsa::StateId target = arc->target_;
            if ( target == sId )
                continue;
            u32 depth = getStateDepth(target, stateDepth, fsa) + 1;
            if ( depth < stateDepth[sId] )
                stateDepth[sId] = depth;
        }
        if ( state->isFinal() )
            stateDepth[sId] = 0;
    }
    return stateDepth[sId];
}

}  // namespace

namespace Nn {

// blank-based topology: CTC by default or RNA if no loop
Core::ParameterBool paramAddBlankTransition(
    "add-blank-transition",
    "insert optional blank arcs between states of automaton",
    false);
Core::ParameterInt paramBlankIndex("blank-label-index", "class id of blank label", -1);
Core::ParameterBool paramAllowLabelLoop(
    "allow-label-loop",
    "allow label loop in addition to blank transition",
    true);

// HMM topology only
Core::ParameterInt paramMinOccur("label-min-occurance", "speech only", 1, 1);
Core::ParameterFloat paramFrameShift("feature-frame-shift", "in seconds", 0.01, 0.0);
Core::ParameterInt paramReduceFrameFactor("reduce-frame-factor", "subsampling", 1, 1);
// overwrite transition weights of the automaton for more flexibility (-log_prob)
// Note: sent-begin ratio is directly applied as transition for the initial arcs
//       cross-word ratio is further applied to the forward transition
Core::ParameterFloatVector paramTransitionWeights(
    "transition-weights",
    "speech forward|loop, sil forward|loop, optional ratio sent-begin speech|sil, cross-word speech|sil",
    ",", 0.0);

AllophoneStateFsaExporter::AllophoneStateFsaExporter(Core::Configuration const& config) :
  Precursor(config),
  mc_(select("model-combination"), Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel, Am::AcousticModel::noEmissions) 
{
    mc_.load();
    Am::AllophoneStateIndex idx = mc_.acousticModel()->silenceAllophoneStateIndex();
    silenceIndex_ = mc_.acousticModel()->emissionIndex(idx);

    if ( paramAddBlankTransition(config) )
    {   // blank-based transducer topology
        int paramBlank = paramBlankIndex(config);
        if ( paramBlank == -1 )
        {   // blank replace silence
            blankIndex_ = silenceIndex_;
            silenceIndex_ = Core::Type<u32>::max;
        } else {
            blankIndex_ = paramBlank;
            verify( blankIndex_ != silenceIndex_ );
        }
        log() << "Add blank transitions to automaton (blank labelId: " << blankIndex_ << ")";
        // Note: set TDP accordingly to control label loop (infinity to disallow)
        // here is more for automaton modification logic
        labelLoop_ = paramAllowLabelLoop(config);
        if ( !labelLoop_ )
            log() << "disallow label loop";
        minOccur_ = 1;
    } else { // HMM topology
        blankIndex_ = Core::Type<u32>::max;
        labelLoop_ = true; // no effect: determined in allophone_state_graph_builder_
        minOccur_ = paramMinOccur(config);
        frameShift_ = paramFrameShift(config);
        reduceFrameFactor_ = paramReduceFrameFactor(config);
        transitionWeights_ = paramTransitionWeights(config);

        log() << "HMM topology based automaton";
        if ( minOccur_ > 1 )
            log() << "each speech label has to occur for at least " << minOccur_ << " frames"
                  << " (" << frameShift_ << " seconds shift in audio and reduced by factor " << reduceFrameFactor_ << ")";

        if ( !transitionWeights_.empty() ) 
        {
            verify( transitionWeights_.size() >= 4 );
            transitionWeights_.resize(8, 0.0);
            log() << "apply transition weight: speech-forward=" << transitionWeights_[0]
                  << " speech-loop=" << transitionWeights_[1]
                  << " silence-forward=" << transitionWeights_[2]
                  << " silence-loop=" << transitionWeights_[3]
                  << "  sent-begin ratio speech=" << transitionWeights_[4]
                  << " silence=" << transitionWeights_[5]
                  << "  cross-word ratio speech=" << transitionWeights_[6]
                  << " silence=" << transitionWeights_[7];
        }
    }

    allophone_state_graph_builder_ = Core::Ref<Speech::AllophoneStateGraphBuilder>(
                                       new Speech::AllophoneStateGraphBuilder(select("allophone-state-graph-builder"),
                                       mc_.lexicon(), mc_.acousticModel()));
}


AllophoneStateFsaExporter::ExportedAutomaton AllophoneStateFsaExporter::exportFsaForOrthography(std::string const& orthography, f64 time) const {
    Core::Ref<Am::AcousticModel>   am    = mc_.acousticModel();
    Speech::AllophoneStateGraphRef graph = allophone_state_graph_builder_->build(orthography);
    graph                                = Fsa::projectInput(graph);
    graph                                = Fsa::removeDisambiguationSymbols(graph);
    graph                                = Fsa::removeEpsilons(graph);
    graph                                = Fsa::normalize(graph);
    // TODO: use Fsa::topologicalSort here, remove toposort below, cleanup
    Core::Ref<Fsa::StaticAutomaton> automaton = Fsa::staticCopy(graph);
    require_eq(automaton->initialStateId(), 0);

    std::vector<Fsa::StateId>                   states;
    std::vector<std::pair<Fsa::StateId, float>> final_states;
    std::vector<Edge>                           edges;

    std::unordered_set<Fsa::StateId> silLoopStates;

    for (Fsa::StateId s = 0ul; s <= automaton->maxStateId(); s++) {
        if (automaton->hasState(s)) {
            states.push_back(s);
            Fsa::State const* state = automaton->fastState(s);
            for (Fsa::State::const_iterator a = state->begin(); a != state->end(); ++a) {
                verify(automaton->hasState(a->target_));
                // Note: set TDPs to dis-allow certain transitions (filtered here)
                if (Speech::Score(a->weight_) >= Core::Type<Speech::Score>::max)
                    continue;
                if ( !labelLoop_ )
                    verify( s != a->target_ );
                else if ( !transitionWeights_.empty() ) {
                    if ( s == a->target_ && am->emissionIndex(a->input_) == silenceIndex_ )
                        silLoopStates.insert(s);
                }
                edges.push_back(Edge(s, a->target_, am->emissionIndex(a->input_), Speech::Score(a->weight_)));
            }
            if (state->isFinal()) {
                final_states.push_back(std::make_pair(s, Speech::Score(state->weight())));
            }
        }
    }

    if ( blankIndex_ != Core::Type<u32>::max )
    {   // blank-based topology: add blank transitions to the automaton
        // no label loop, simply add blank loop arcs on each state including initial and final
        if ( !labelLoop_ )
            for (std::vector<Fsa::StateId>::const_iterator sIt=states.begin(); sIt!=states.end(); ++sIt)
                edges.emplace_back(*sIt, *sIt, blankIndex_, 0);
        else { // label loop preserved: add additional path with blank arcs
            Fsa::StateId maxStateId = automaton->maxStateId();
            u32 nEdges = edges.size();
            for (u32 idx = 0; idx < nEdges; ++idx)
            {   // skip loop and blank arcs
                Edge e = edges[idx]; // copy on purpose as reference may be invalidated
                if ( e.from == e.to || e.emission_idx == blankIndex_ )
                    continue;
                states.push_back(++maxStateId);
                edges.emplace_back(e.from, maxStateId, blankIndex_, 0);
                edges.emplace_back(maxStateId, maxStateId, blankIndex_, 0);
                edges.emplace_back(maxStateId, e.to, e.emission_idx, e.weight);
            }
            // transitionWeights_ also applicable ?
        }
    } else { // HMM topology
        if ( !transitionWeights_.empty() )
            modifyTransitionWeights(edges, silLoopStates);
        if ( minOccur_ > 1 )
            modifyMinDuration(edges, states, automaton, time);
    }

    if ( blankIndex_ != Core::Type<u32>::max && labelLoop_ )
    {   // tailing blanks: loop on the single final state
        verify( final_states.size() == 1 );
        edges.emplace_back(final_states.front().first, final_states.front().first, blankIndex_, 0);
    }

    toposort(states, final_states, edges);
    filter_edges(edges);
    make_single_final_state(states, final_states, edges);

    ExportedAutomaton result;
    result.num_states = states.size();
    result.num_edges  = edges.size();
    result.edges      = std::vector<u32>(edges.size() * 3ul);
    result.weights    = std::vector<f32>(edges.size());

    for (size_t e = 0ul; e < edges.size(); e++) {
        result.edges[e]                    = edges[e].from;
        result.edges[e + edges.size()]     = edges[e].to;
        result.edges[e + 2 * edges.size()] = edges[e].emission_idx;
        result.weights[e]                  = edges[e].weight;
    }

    return result;
}

// Note: pronunciation variants ignored here, i.e. no normalization
void AllophoneStateFsaExporter::modifyTransitionWeights(std::vector<Edge>& edges, const std::unordered_set<Fsa::StateId>& silLoopStates) const
{   // HMM topology (label loop and no blank): overwrite transition weights
    verify( labelLoop_ && silenceIndex_ != Core::Type<u32>::max );
    u32 nEdges = edges.size();
    for (u32 idx = 0; idx < nEdges; ++idx) 
    {
        Edge& e = edges[idx]; // no size change: always valid
        if ( e.from == 0 )
            e.weight = e.emission_idx == silenceIndex_ ? transitionWeights_[5] : transitionWeights_[4];
        else if ( e.from == e.to )
            e.weight = e.emission_idx == silenceIndex_ ? transitionWeights_[3] : transitionWeights_[1];
        else if ( silLoopStates.count(e.from) > 0 )
            e.weight = transitionWeights_[2]; // silence forward
        else { // speech forward
            float factor = e.emission_idx == silenceIndex_ ? transitionWeights_[7] : transitionWeights_[6];
            e.weight = transitionWeights_[0] + factor;
        }
    }
}

void AllophoneStateFsaExporter::modifyMinDuration(std::vector<Edge>& edges, std::vector<Fsa::StateId>& states, Core::Ref<const Fsa::StaticAutomaton> automaton, f64 time) const
{   // expanded sequence should not exceed number of frames
    u32 expand = minOccur_;
    if ( time >= 0 ) 
    {   // if negative time: unknown length, assume always expandable
        u32 maxLength = time / frameShift_ / reduceFrameFactor_;
        std::vector<u32> stateDepth(automaton->size(), Core::Type<u32>::max);
        u32 seqLength = getStateDepth(0, stateDepth, automaton); // shortest sequence length
        while ( seqLength * expand > maxLength && expand > 1 )
            --expand; // reduce expansion
    }
    if ( expand > 1 ) 
    {
        verify( silenceIndex_ != Core::Type<u32>::max );
        Fsa::StateId maxStateId = automaton->maxStateId();
        u32 nEdges = edges.size();
        for (u32 idx = 0; idx < nEdges; ++idx)
        {   // expand each speech forward transition to minOccur_
            Edge e = edges[idx];
            verify( e.emission_idx != blankIndex_ );
            if ( e.from == e.to || e.emission_idx == silenceIndex_ )
                continue;
            Fsa::StateId target = e.to;
            for (u32 rp = 1; rp < expand; ++rp) 
            {   // Note: no weight for forced loop
                states.push_back(++maxStateId);
                edges.emplace_back(maxStateId, target, e.emission_idx, 0);
                target = maxStateId;
            }
            edges[idx].to = maxStateId;
        }
    } else
        warning() << "can't expand segment for label-min-occurance " << minOccur_ << " (exceeding number of frames)";
}

}  // namespace Nn
