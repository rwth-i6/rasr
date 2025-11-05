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
#ifndef _TENSORFLOW_GRAPH_LOADER_HH
#define _TENSORFLOW_GRAPH_LOADER_HH

#include <Core/Component.hh>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/util/port.h>
#include "Graph.hh"

namespace Tensorflow {

class GraphLoader : public virtual Core::Component {
public:
    typedef Core::Component Precursor;

    static Core::ParameterStringVector paramRequiredLibraries;

    GraphLoader(Core::Configuration const& config);
    virtual ~GraphLoader() = default;

    virtual std::unique_ptr<Graph> load_graph() = 0;
    virtual void                   initialize(Session& session);

protected:
    std::vector<std::string> required_libraries_;

    static void setGraphDef(Graph& graph, tf::GraphDef const& graph_def);
};

inline GraphLoader::GraphLoader(Core::Configuration const& config)
        : Precursor(config),
          required_libraries_(paramRequiredLibraries(config)) {
    log("TF version: %s\nTF_GRAPH_DEF_VERSION: %d\nCUDA/MKL: %d/%d",
        TF_VERSION_STRING, TF_GRAPH_DEF_VERSION,
        tensorflow::IsGoogleCudaEnabled(), tensorflow::IsMklEnabled());
}

inline void GraphLoader::setGraphDef(Graph& graph, tf::GraphDef const& graph_def) {
    graph.graph_def_ = graph_def;
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_GRAPH_LOADER_HH
