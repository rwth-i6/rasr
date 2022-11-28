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
#ifndef NN_ALLOPHONE_STATE_FSA_HH
#define NN_ALLOPHONE_STATE_FSA_HH

#include <Core/Component.hh>
#include <Speech/AllophoneStateGraphBuilder.hh>
#include <Speech/ModelCombination.hh>

namespace Nn {

struct Edge {
    Edge() : from(0u), to(0u), emission_idx(0u), weight(0.0f) {}
    Edge(Fsa::StateId from, Fsa::StateId to, Am::AcousticModel::EmissionIndex eIdx, float cost)
        : from(from), to(to), emission_idx(eIdx), weight(cost) {}

    Fsa::StateId                     from;
    Fsa::StateId                     to;
    Am::AcousticModel::EmissionIndex emission_idx;
    float                            weight;
};


// Fsa graph modifier: HMM/CTC/RNA topology 
// - filter arcs with inf weight
// - minimum duration
// - customize transition weights
// - label-dependent loop
class FsaGraphModifier : Core::Component {
    typedef Core::Component Precursor;

public:
    FsaGraphModifier(const Core::Configuration& config, Core::Ref<Am::AcousticModel> am);
    ~FsaGraphModifier() {}

    void modify(Speech::AllophoneStateGraphRef, std::vector<Fsa::StateId>&, std::vector<Edge>&) const;

protected:
    void modifyTransitionWeights(std::vector<Edge>&, const std::unordered_set<Fsa::StateId>&) const;
    void modifyMinDuration(std::vector<Edge>&, std::vector<Fsa::StateId>&, Core::Ref<const Fsa::StaticAutomaton>) const;

private:
    Core::Ref<Am::AcousticModel> am_;

    u32  silenceIndex_;
    u32  blankIndex_;
    bool labelLoop_;
    u32  minOccur_; // minimum occurance of speech label (force loop)
    u32  loopShift_;

    std::vector<f64> transitionWeights_; // transition scores
};


class AllophoneStateFsaExporter : Core::Component {
public:
    typedef Core::Component Precursor;

    struct ExportedAutomaton {
        size_t           num_states;
        size_t           num_edges;
        // separate each Edge's from,to,emissionIdx into one container with stepsize 3 
        std::vector<u32> edges;   
        std::vector<f32> weights; // edges.size() / 3 == weights.size()
    };

    AllophoneStateFsaExporter(Core::Configuration const& config)
            : Precursor(config),
              mc_(select("model-combination"),
                  Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel,
                  Am::AcousticModel::noEmissions),
              allophone_state_graph_builder_(),
              graphModifier_(config, mc_.acousticModel()) {
        mc_.load();
        allophone_state_graph_builder_ = Core::Ref<Speech::AllophoneStateGraphBuilder>(
                new Speech::AllophoneStateGraphBuilder(select("allophone-state-graph-builder"),
                                                       mc_.lexicon(),
                                                       mc_.acousticModel(),
                                                       false));
    };
    ~AllophoneStateFsaExporter() {}

    ExportedAutomaton exportFsaForOrthography(std::string const& orthography) const;

private:
    Speech::ModelCombination                      mc_;
    Core::Ref<Speech::AllophoneStateGraphBuilder> allophone_state_graph_builder_;
    FsaGraphModifier                              graphModifier_;
};

}  // namespace Nn

#endif  // NN_ALLOPHONE_STATE_FSA_HH
