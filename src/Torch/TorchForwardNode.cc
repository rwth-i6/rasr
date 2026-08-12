/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "TorchForwardNode.hh"

#include <algorithm>
#include <chrono>

#include <Flow/Vector.hh>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

// -----------------------------------------------------------------------------
//                               TorchForwardNode
// -----------------------------------------------------------------------------

Core::ParameterString TorchForwardNode::paramId(
        "id", "Changing the id resets the caches for the recurrent connections.");

TorchForwardNode::TorchForwardNode(Core::Configuration const& c)
        : Core::Component(c),
          Precursor(c),
          computationDone_(false),
          torchModel_(select("model")),
          currentOutputFrame_(0) {
}

bool TorchForwardNode::setParameter(const std::string& name, const std::string& value) {
    if (paramId.match(name)) {
        // New id means we entered a new segment
        // => Reset node and clear computation cache
        computationDone_ = false;
        timestamps_.clear();
        outputCache_.clear();
        currentOutputFrame_ = 0;
    }
    return true;
}

bool TorchForwardNode::work(Flow::PortId p) {
    // Only one output port
    require_eq(static_cast<size_t>(p), 0);

    // Perform computation if not yet done (only run computation once per segment)
    if (not computationDone_) {
        computationDone_ = true;

        auto timer_start = std::chrono::steady_clock::now();

        // gather timestamped DataPtr's (features) from input port
        std::deque<Flow::DataPtr<Flow::Timestamp>> data;
        bool                                       success = true;
        while (success) {
            Flow::DataPtr<Flow::Timestamp> d;
            success = getData(0, d);
            if (success and Flow::Data::isNotSentinel(&(*d))) {
                data.push_back(d);
            }
            timestamps_.push_back(*d.get());
        }

        // No input features available -> immediate EOS
        if (data.empty()) {
            return putData(p, Flow::Data::eos());
        }

        // Create session inputs
        torch::Tensor features = toTensor(data);
        torch::Tensor lengths  = torch::tensor({static_cast<int64_t>(data.size())}, torch::kInt64);

        std::vector<torch::Tensor> session_outputs;
        std::vector<torch::Tensor> session_inputs = torchModel_.makeInputs(features, lengths);

        // Run session to compute outputs
        auto t_start = std::chrono::steady_clock::now();
        torchModel_.session().run(session_inputs, session_outputs);

        // Print AM timing statistics
        auto t_end     = std::chrono::steady_clock::now();
        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        log("num_frames: %zu elapsed: %f AM_RTF: %f",
            data.size(), t_elapsed, t_elapsed / (static_cast<double>(data.size()) / 100.0));

        // Read model outputs
        torch::Tensor out = torchModel_.outputsFrom(session_outputs).contiguous();

        // Append session outputs to cache
        appendToOutput(out);

        // Print overall timing statistics
        auto timer_end = std::chrono::steady_clock::now();
        log("flow fwd time: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    }

    // If we have not-yet-returned outputs available, send one of them
    if (currentOutputFrame_ < outputCache_.size()) {
        return putData(p, outputCache_[currentOutputFrame_++]);
    }

    // All available outputs have been returned -> reached EOS
    return putData(p, Flow::Data::eos());
}

torch::Tensor TorchForwardNode::toTensor(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const {
    // Expect a non-empty sequence of float feature vectors
    require(!data.empty());
    require_eq(data.front()->datatype(), Flow::Vector<f32>::type());

    auto* first = dynamic_cast<Flow::Vector<f32>*>(data.front().get());
    require(first != nullptr);

    const size_t T = data.size();
    const size_t F = first->size();

    // Create a Torch input tensor with shape [1, T, F]
    torch::Tensor features   = torch::empty({1, static_cast<long>(T), static_cast<long>(F)}, torch::kFloat32);
    auto          featureAcc = features.accessor<float, 3>();

    // Copy each Flow frame into the corresponding time step of the tensor
    for (size_t t = 0; t < T; ++t) {
        auto* vec = dynamic_cast<Flow::Vector<f32>*>(data[t].get());
        require(vec != nullptr);
        require_eq(vec->size(), F);

        for (size_t f = 0; f < F; ++f) {
            featureAcc[0][t][f] = (*vec)[f];
        }
    }

    return features;
}

void TorchForwardNode::appendToOutput(torch::Tensor const& out) {
    // Expect batched model output with shape [1, T, F]
    require_eq(out.dim(), 3);
    require(out.size(2) > 0);

    const size_t outputDim = out.size(2);
    auto         outAcc    = out.accessor<float, 3>();

    // Convert each output frame into a Flow vector and cache it for emission
    for (size_t t = 0; t < static_cast<size_t>(out.size(1)); ++t) {
        auto* outputVec = new Flow::Vector<f32>(outputDim);
        for (size_t i = 0; i < outputDim; ++i) {
            (*outputVec)[i] = outAcc[0][t][i];
        }

        // Reuse the closest available input timestamp for this output frame
        outputVec->setTimestamp(timestamps_[std::min(t, timestamps_.size() - 1ul)]);
        outputCache_.push_back(outputVec);
    }
}

}  // namespace Torch