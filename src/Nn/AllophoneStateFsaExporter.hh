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
#ifndef NN_ALIGNMENTFSAEXPORTER_HH
#define NN_ALIGNMENTFSAEXPORTER_HH

#include <Core/Component.hh>
#include <Speech/AllophoneStateGraphBuilder.hh>
#include <Speech/ModelCombination.hh>

namespace {

struct Edge {
    Edge()  : from(0u), to(0u), emission_idx(0u), weight(0.0f) {}
    Edge(Fsa::StateId from, Fsa::StateId to, Am::AcousticModel::EmissionIndex emission_idx, float cost)
            : from(from), to(to), emission_idx(emission_idx), weight(cost) {}

    Fsa::StateId                     from;
    Fsa::StateId                     to;
    Am::AcousticModel::EmissionIndex emission_idx;
    float                            weight;
};

}

namespace Nn {

class AllophoneStateFsaExporter : Core::Component {
public:
    typedef Core::Component Precursor;

    struct ExportedAutomaton {
        size_t           num_states;
        size_t           num_edges;
        // separate from,to,emissionIdx: num_edges == edges.size() / 3 == weights.size()
        std::vector<u32> edges;
        std::vector<f32> weights;
    };

    AllophoneStateFsaExporter(Core::Configuration const& config);
    ~AllophoneStateFsaExporter() {}

    ExportedAutomaton exportFsaForOrthography(std::string const& orthography, f64 time=-1.0) const;

protected:
    void modifyTransitionWeights(std::vector<Edge>&, const std::unordered_set<Fsa::StateId>&) const;
    void modifyMinDuration(std::vector<Edge>&, std::vector<Fsa::StateId>&, Core::Ref<const Fsa::StaticAutomaton>, f64) const;

private:
    Speech::ModelCombination                      mc_;
    Core::Ref<Speech::AllophoneStateGraphBuilder> allophone_state_graph_builder_;

    u32 silenceIndex_;
    u32 blankIndex_;
    bool labelLoop_;

    // HMM topology only (no blank)
    u32 minOccur_; // minimum occurance of each speech label (force loop)
    f64 frameShift_; // frame shift in seconds: to compute audio length T for each segment
    u32 reduceFrameFactor_; // subsampling
    std::vector<f64> transitionWeights_; // transition scores
};

}  // namespace Nn

#endif  // NN_ALIGNMENTFSAEXPORTER_HH
