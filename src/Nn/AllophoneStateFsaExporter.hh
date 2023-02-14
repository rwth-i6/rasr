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
#include <Speech/Module.hh>

namespace Nn {

class AllophoneStateFsaExporter : Core::Component {
public:
    typedef Core::Component Precursor;

    struct ExportedAutomaton {
        size_t           num_states;
        size_t           num_edges;
        std::vector<u32> edges;  // contains from,to,emissionIdx, thus num_edges == edges.size() / 3 == weights.size()
        std::vector<f32> weights;
    };

    AllophoneStateFsaExporter(Core::Configuration const& config)
            : Precursor(config),
              mc_(select("model-combination"),
                  Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel,
                  Am::AcousticModel::noEmissions),
              allophone_state_graph_builder_() {
        mc_.load();
        allophone_state_graph_builder_ = Core::Ref<Speech::AllophoneStateGraphBuilder>(
                Speech::Module::instance().createAllophoneStateGraphBuilder(
                        select("allophone-state-graph-builder"),
                        mc_.lexicon(),
                        mc_.acousticModel()));
    };
    ~AllophoneStateFsaExporter() {}

    ExportedAutomaton exportFsaForOrthography(std::string const& orthography) const;

private:
    Speech::ModelCombination                      mc_;
    Core::Ref<Speech::AllophoneStateGraphBuilder> allophone_state_graph_builder_;
};

}  // namespace Nn

#endif  // NN_ALIGNMENTFSAEXPORTER_HH
