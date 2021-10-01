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

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/version.h>

namespace Tensorflow {

Core::ParameterBool Session::paramProfileRun("profile-run",
                                             "store runtime profiles",
                                             false);

Core::ParameterString Session::paramProfilePrefix("profile-prefix",
                                                  "filename prefix for stored profiles",
                                                  "profile");

Core::ParameterBool Session::paramLogDevicePlacement("log-device-placement",
                                                     "print placement of tensorflow ops",
                                                     false);

/* For detailed description of options see:
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
 */
Core::ParameterInt   Session::paramIntraOpParallelismThreads("intra-op-parallelism-threads",
                                                           "Number of threads of execution of parallelizable ops, 0 = system picks appropriate number",
                                                           1, 0);
Core::ParameterInt   Session::paramInterOpParallelismThreads("inter-op-parallelism-threads",
                                                           "Execute parallel nodes with this many threads",
                                                           1, 0);
Core::ParameterFloat Session::paramPerProcessGpuMemoryFraction("per-process-gpu-memory-fraction",
                                                               "Fraction of GPU memory to allocate on session creation",
                                                               0.95, 0.0, 1.0);
Core::ParameterBool  Session::paramAllowGpuMemoryGrowth("allow-gpu-memory-growth",
                                                       "Allow GPU memory allocations after session creation",
                                                       true);

void Session::addGraph(Graph const& graph) {
    auto     timer_start = std::chrono::steady_clock::now();
    tf::Env* env         = tf::Env::Default();
    for (std::string const& lib : graph.libraries()) {
        void*      handle = nullptr;
#if TF_MAJOR_VERSION < 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 4)
        tf::Status status = env->LoadLibrary(lib.c_str(), &handle);
#else
        tf::Status status = env->LoadDynamicLibrary(lib.c_str(), &handle);
#endif
        log("Loading library: ") << lib.c_str();
        if (!status.ok()) {
            criticalError("error loading library: %s", status.ToString().c_str());
        }
    }

    tf::Status status = session_->Create(graph.graph_def_);
    if (not status.ok()) {
        criticalError("error creating session %s", status.ToString().c_str());
    }

    std::vector<std::string> var_init;
    for (auto const& v : graph.variables()) {
        if (not v.second.initializer_name.empty()) {
            var_init.push_back(v.second.initializer_name);
        }
    }
    if (not var_init.empty()) {
        run({}, var_init);
    }
    auto timer_end = std::chrono::steady_clock::now();
    log("Session::addGraph ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count() << "ms";
}

bool Session::run(std::vector<std::pair<std::string, Tensor>> const& inputs,
                  std::vector<std::string> const&                    output_tensor_names,
                  std::vector<std::string> const&                    target_node_names,
                  std::vector<Tensor>&                               outputs) {
    std::vector<std::pair<std::string, tf::Tensor>> tf_inputs;
    tf_inputs.reserve(inputs.size());
    std::vector<tf::Tensor> tf_outputs;

    for (auto& input : inputs) {
        tf_inputs.push_back(std::make_pair(input.first, *input.second.tensor_));
    }

    tf::Status status;

    if (profileRun_) {
        tf::RunOptions options;
        options.set_trace_level(tf::RunOptions::SOFTWARE_TRACE);
        tf::RunMetadata meta_data;
        status = session_->Run(options, tf_inputs, output_tensor_names, target_node_names, &tf_outputs, &meta_data);
        std::ofstream out(profilePrefix_ + std::to_string(profileCounter_++), std::ios::out | std::ios::trunc);
        meta_data.SerializeToOstream(&out);
    }
    else {
        status = session_->Run(tf_inputs, output_tensor_names, target_node_names, &tf_outputs);
    }

    if (!status.ok()) {
        std::string target;
        for (auto const& t : target_node_names) {
            target += t + " ";
        }
        criticalError("error calling Session::Run (target: %s): %s", target.c_str(), status.ToString().c_str());
    }

    outputs.clear();
    for (auto& tensor : tf_outputs) {
        outputs.emplace_back(Tensor(std::move(tensor)));
    }

    return status.ok();
}

}  // namespace Tensorflow
