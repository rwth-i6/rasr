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
#include "MetaGraphLoader.hh"

#include <tensorflow/core/framework/variable.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include "Session.hh"

namespace Tensorflow {

Core::ParameterString MetaGraphLoader::paramMetaGraphFile(
        "meta-graph-file",
        "path of the MetaGraphDef protobuffer to load",
        "");
Core::ParameterString MetaGraphLoader::paramSavedModelFile(
        "saved-model-file",
        "path to the stored model variables",
        "");

std::unique_ptr<Graph> MetaGraphLoader::load_graph() {
    auto timer_start = std::chrono::steady_clock::now();
    if (meta_graph_file_.empty()) {
        criticalError("no graph-def-path set");
    }

    tf::Env*         env = tf::Env::Default();
    tf::MetaGraphDef meta_graph_def;
    tf::Status       status = ReadBinaryProto(env, meta_graph_file_, &meta_graph_def);
    if (not status.ok()) {
        criticalError("error reading graph def %s", status.ToString().c_str());
    }

    if (not meta_graph_def.has_graph_def()) {
        criticalError("meta-graph has not graph def");
    }
    if (not meta_graph_def.has_saver_def()) {
        criticalError("meta-graph has not saver def");
    }

    restore_op_name_              = meta_graph_def.saver_def().restore_op_name();
    restore_filename_tensor_name_ = meta_graph_def.saver_def().filename_tensor_name();

    tf::GraphDef const&    graph_def = meta_graph_def.graph_def();
    std::unique_ptr<Graph> result(new Graph());
    setGraphDef(*result, graph_def);

    for (std::string const& lib : required_libraries_) {
        result->addLibrary(lib);
    }

    std::unordered_map<std::string, Variable> vars;
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
            vars[var_name] = {var_name, "", "", "", dt, dims};
        }
    }

    variable_initializers_.clear();
    auto const& collections = meta_graph_def.collection_def();
    for (auto const& keyval : collections) {
        if (keyval.first == "variables") {
            if (not keyval.second.has_bytes_list()) {
                error("variables collection is not a byte-list");
                break;
            }
            auto const& list = keyval.second.bytes_list();
            for (int i = 0; i < list.value_size(); i++) {
                std::string const& var_def_string = list.value(i);
                tf::VariableDef    var_def;
                var_def.ParseFromString(var_def_string);
                auto iter = vars.find(var_def.variable_name());
                if (iter != vars.end()) {
                    iter->second.initial_value_name = var_def.initial_value_name();
                    iter->second.initializer_name   = var_def.initializer_name();
                    iter->second.snapshot_name      = var_def.snapshot_name();
                    result->addVariable(iter->second);
                    variable_initializers_.push_back(var_def.initializer_name());
                }
            }
        }
        else if (keyval.first == "update_ops") {
            if (not keyval.second.has_node_list()) {
                error("update_ops collection is not a node-list");
                break;
            }
            auto const& list = keyval.second.node_list();
            for (int i = 0; i < list.value_size(); i++) {
                result->addUpdateOp(list.value(i));
            }
        }
        else if (keyval.first == "_RETURNN_state_vars") {
            if (not keyval.second.has_node_list()) {
                error("_RETURNN_state_vars collection is not a node-list");
                break;
            }
            auto const& list = keyval.second.node_list();
            for (int i = 0; i < list.value_size(); i++) {
                result->addStateVar(list.value(i));
            }
        }
    }

    auto timer_end = std::chrono::steady_clock::now();
    log("Session::loadGraph: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count() << "ms";

    return result;
}

void MetaGraphLoader::initialize(Session& session) {
    auto timer_start = std::chrono::steady_clock::now();

    if (saved_model_file_.empty()) {
        criticalError("no saved-model-file set");
    }

    std::vector<Tensor> outputs;
    Tensor              filename_tensor;
    filename_tensor.set<Tensorflow::tstring>(saved_model_file_);
    session.run({std::make_pair<>(restore_filename_tensor_name_, filename_tensor)}, {}, {restore_op_name_}, outputs);

    auto timer_end = std::chrono::steady_clock::now();
    log("Session::initialize: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count() << "ms " << saved_model_file_;
}

}  // namespace Tensorflow
