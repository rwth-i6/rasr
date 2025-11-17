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

struct Edge {
    Edge()
            : from(0u),
              to(0u),
              emission_idx(0u),
              weight(0.0f) {}
    Edge(Fsa::StateId from, Fsa::StateId to, Am::AcousticModel::EmissionIndex emission_idx, float cost)
            : from(from),
              to(to),
              emission_idx(emission_idx),
              weight(cost) {}

    Fsa::StateId                     from;
    Fsa::StateId                     to;
    Am::AcousticModel::EmissionIndex emission_idx;
    float                            weight;
};

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

}  // namespace

namespace Nn {

AllophoneStateFsaExporter::ExportedAutomaton AllophoneStateFsaExporter::exportFsaForOrthography(std::string const& orthography) const {
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

    for (Fsa::StateId s = 0ul; s <= automaton->maxStateId(); s++) {
        if (automaton->hasState(s)) {
            states.push_back(s);
            Fsa::State const* state = automaton->fastState(s);
            for (Fsa::State::const_iterator a = state->begin(); a != state->end(); ++a) {
                verify(automaton->hasState(a->target_));
                if (Speech::Score(a->weight_) >= Core::Type<Speech::Score>::max)
                    continue;
                edges.push_back(Edge(s, a->target_, am->emissionIndex(a->input_), Speech::Score(a->weight_)));
            }
            if (state->isFinal()) {
                final_states.push_back(std::make_pair(s, Speech::Score(state->weight())));
            }
        }
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

}  // namespace Nn
