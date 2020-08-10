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
#include "VanillaGraphLoader.hh"

#include <tensorflow/core/platform/env.h>

namespace Tensorflow {

Core::ParameterString VanillaGraphLoader::paramFile(
        "file",
        "path of the GraphDef protobuffer to load",
        "");

std::unique_ptr<Graph> VanillaGraphLoader::load_graph() {
    if (file_.empty()) {
        criticalError("no graph-def-path set");
    }

    tf::Env*     env = tf::Env::Default();
    tf::GraphDef graph_def;
    tf::Status   status = ReadBinaryProto(env, file_, &graph_def);
    if (not status.ok()) {
        criticalError("error reading graph def %s", status.ToString().c_str());
    }

    std::unique_ptr<Graph> result(new Graph());
    setGraphDef(*result, graph_def);

    for (int i = 0; i < graph_def.node_size(); i++) {
        auto& node = graph_def.node(i);
        if (node.op() == "Placeholder" or node.op() == "PlaceholderV2") {
            result->addInput(node.name());
        }
        else if (node.op() == "Variable" or node.op() == "VariableV2") {
            DataType         dt       = node.attr().find("dtype")->second.type();
            std::string      var_name = node.name() + ":0";  // to use the same name as is used inside collections we append ":0" here
            auto const&      shape    = node.attr().find("_output_shapes")->second.list().shape(0);
            std::vector<s64> dims(static_cast<size_t>(shape.dim_size()));
            for (int i = 0; i < shape.dim_size(); i++) {
                dims[i] = shape.dim(i).size();
            }
            result->addVariable({var_name, "", "", "", dt, dims});
        }
    }

    return result;
}

}  // namespace Tensorflow
