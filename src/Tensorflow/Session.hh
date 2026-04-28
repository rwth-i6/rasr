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
#ifndef _TENSORFLOW_SESSION_HH
#define _TENSORFLOW_SESSION_HH

#include <memory>

#include <tensorflow/core/public/session.h>

#include <Core/Component.hh>
#include "Graph.hh"
#include "Tensor.hh"

namespace Tensorflow {

namespace tf = tensorflow;

class Session : public Core::Component {
public:
    typedef Core::Component Precursor;

    static Core::ParameterBool   paramProfileRun;
    static Core::ParameterString paramProfilePrefix;

    static Core::ParameterBool  paramLogDevicePlacement;
    static Core::ParameterInt   paramIntraOpParallelismThreads;
    static Core::ParameterInt   paramInterOpParallelismThreads;
    static Core::ParameterFloat paramPerProcessGpuMemoryFraction;
    static Core::ParameterBool  paramAllowGpuMemoryGrowth;

    Session(Core::Configuration const& config);
    virtual ~Session();

    void addGraph(Graph const& graph);

    bool run(std::vector<std::pair<std::string, Tensor>> const& inputs,
             std::vector<std::string> const&                    target_node_names);

    bool run(std::vector<std::pair<std::string, Tensor>> const& inputs,
             std::vector<std::string> const&                    output_tensor_names,
             std::vector<std::string> const&                    target_node_names,
             std::vector<Tensor>&                               outputs);

private:
    const bool        profileRun_;
    const std::string profilePrefix_;
    size_t            profileCounter_;

    std::unique_ptr<tf::Session> session_;
};

// Inline implementations of simple functions

inline Session::Session(Core::Configuration const& config)
        : Precursor(config),
          profileRun_(paramProfileRun(config)),
          profilePrefix_(paramProfilePrefix(config)),
          profileCounter_(0ul) {
    tf::SessionOptions options;
    options.config.set_log_device_placement(paramLogDevicePlacement(config));
    options.config.set_intra_op_parallelism_threads(paramIntraOpParallelismThreads(config));
    options.config.set_inter_op_parallelism_threads(paramInterOpParallelismThreads(config));
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(paramPerProcessGpuMemoryFraction(config));
    options.config.mutable_gpu_options()->set_allow_growth(paramAllowGpuMemoryGrowth(config));
    session_.reset(tf::NewSession(options));
}

inline Session::~Session() {
    if (session_.get() != nullptr) {
        tsl::Status unused = session_->Close();
        (void)unused;
    }
}

inline bool Session::run(std::vector<std::pair<std::string, Tensor>> const& inputs,
                         std::vector<std::string> const&                    target_node_names) {
    std::vector<Tensor> outputs;
    return run(inputs, {}, target_node_names, outputs);
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_SESSION_HH
