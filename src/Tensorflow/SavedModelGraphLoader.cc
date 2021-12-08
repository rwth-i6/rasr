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
#include "Session.hh"
#include "VanillaGraphLoader.hh"


namespace Tensorflow {

Core::ParameterString VanillaGraphLoader::paramSavedModelDir(
        "saved-model-dir",
        "path of the SavedModel dir to load",
        "");

std::unique_ptr<Graph> VanillaGraphLoader::load_graph() {
    auto timer_start = std::chrono::steady_clock::now();
    if (saved_model_dir_.empty()) {
        criticalError("no graph-def-path set");
    }

    tf::SessionOptions session_options;
    tf::RunOptions run_options;
    tf::Status status = tf::LoadSavedModel(session_options, run_options, saved_model_dir_, {"serve"},
		                   &bundle);
    if (not status.ok()) {
        criticalError("error reading graph def %s", status.ToString().c_str());
    }

    if (not bundle.meta_graph_def.has_graph_def()) {
	criticalError("meta-graph has not graph def");
    }
    if (not bundle.meta_graph_def.has_saver_def()) {
	criticalError("meta-graph has not saver def");
    }


    tf::GraphDef const&    graph_def = bundle.meta_graph_def.graph_def();
    std::unique_ptr<Graph> result(new Graph());
    setGraphDef(*result, graph_def);
    auto timer_end = std::chrono::steady_clock::now();
    log("Session::loadGraph: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count() << "ms";

    return result;
}

void VanillaGraphLoader::initialize(Session& session) {
    auto timer_start = std::chrono::steady_clock::now();

    session.setSession(bundle.GetSession());
    auto timer_end = std::chrono::steady_clock::now();
    log("Session::initialize: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count() << "ms " << saved_model_dir_;
}

}  // namespace Tensorflow
