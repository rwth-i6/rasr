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

#ifndef _TORCH_FORWARD_NODE_HH
#define _TORCH_FORWARD_NODE_HH

#include <deque>

#include <Flow/Attributes.hh>
#include <Flow/Datatype.hh>
#include <Flow/Node.hh>
#include <Flow/Timestamp.hh>

#include "Model.hh"

namespace Torch {

/*
 * Flow node that runs an exported Torch model on a buffered input segment
 */
class TorchForwardNode : public Flow::SleeveNode {
public:
    using Precursor = Flow::SleeveNode;

    static Core::ParameterString paramId;

    static std::string filterName();

    explicit TorchForwardNode(Core::Configuration const& c);
    virtual ~TorchForwardNode() = default;

    // Handles parameter updates and resets internal cached outputs if needed
    bool setParameter(const std::string& name, const std::string& value) override;

    // Pulls input data, runs the model once per segment and emits cached outputs
    bool work(Flow::PortId p) override;

protected:
    bool computationDone_;  // current segment finished yes/no

    // Wrapped Torch model
    Model torchModel_;

    // Cached input timestamps and computed output frames
    std::deque<Flow::Timestamp> timestamps_;
    std::deque<Flow::Data*>     outputCache_;
    size_t                      currentOutputFrame_;

    // Helper to convert buffered flow input data to a Torch tensor
    torch::Tensor toTensor(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const;

    // Helper to convert a Torch tensor back to Flow data and add them to the output cache
    void appendToOutput(torch::Tensor const& out);
};

// inline implementations

inline std::string TorchForwardNode::filterName() {
    return {"torch-forward"};
}

}  // namespace Torch

#endif  // _TORCH_FORWARD_NODE_HH