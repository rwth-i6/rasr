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
#include "TensorflowForwardNode.hh"

#include <Flow/Vector.hh>

#include "Module.hh"

namespace {

}  // namespace

namespace Tensorflow {

// -----------------------------------------------------------------------------
//                           TensorflowForwardNode
// -----------------------------------------------------------------------------

Core::ParameterString TensorflowForwardNode::paramId(
        "id", "Changing the id resets the caches for the recurrent connections.");

TensorflowForwardNode::TensorflowForwardNode(Core::Configuration const& c)
        : Core::Component(c),
          Precursor(c),
          eos_(false),
          input_port_names_(),
          output_port_names_(),
          input_port_map_(),
          output_port_map_(),
          output_tensor_names_(),
          session_(select("session")),
          loader_(Module::instance().createGraphLoader(select("loader"))),
          graph_(loader_->load_graph()),
          tensor_input_map_(select("input-map")),
          tensor_output_map_(select("output-map")) {
    session_.addGraph(*graph_);
    loader_->initialize(session_);
}

Flow::PortId TensorflowForwardNode::getInput(std::string const& name) {
    auto iter = input_port_map_.find(name);
    if (iter == input_port_map_.end()) {
        Flow::PortId port = input_port_names_.size();
        input_port_names_.push_back(name);
        input_port_map_.insert(std::make_pair(name, port));
        require(tensor_input_map_.has_info(name));
        addInput(port);
        return port;
    }
    return iter->second;
}

Flow::PortId TensorflowForwardNode::getOutput(std::string const& name) {
    auto iter = output_port_map_.find(name);
    if (iter == output_port_map_.end()) {
        Flow::PortId port = output_port_names_.size();
        output_port_names_.push_back(name);
        output_port_map_.insert(std::make_pair(name, port));
        current_output_frame_.push_back(0ul);
        outputs_.emplace_back();
        require(tensor_output_map_.has_info(name));
        output_tensor_names_.push_back(tensor_output_map_.get_info(name).tensor_name());
        addOutput(port);
        return port;
    }
    return iter->second;
}

bool TensorflowForwardNode::setParameter(const std::string& name, const std::string& value) {
    if (paramId.match(name)) {
        eos_ = false;
        timestamps_.clear();
        for (auto& out : outputs_) {
            out.clear();
        }
        std::fill(current_output_frame_.begin(), current_output_frame_.end(), 0ul);
    }
    return true;
}

bool TensorflowForwardNode::work(Flow::PortId p) {
    // check if computation is needed
    require_lt(static_cast<size_t>(p), current_output_frame_.size());
    if (current_output_frame_[p] >= outputs_[p].size() and not eos_) {
        auto timer_start = std::chrono::steady_clock::now();
        // gather inputs, for all input ports we gather all data and feed it to the tensorflow runtime in one go
        size_t                                                  start_frame = timestamps_.size();
        std::vector<std::deque<Flow::DataPtr<Flow::Timestamp>>> data(input_port_names_.size());
        for (size_t i = 0ul; i < input_port_names_.size(); i++) {
            bool success = true;
            while (success) {
                Flow::DataPtr<Flow::Timestamp> d;
                success = getData(i, d);
                if (success and Flow::Data::isNotSentinel(&(*d))) {
                    data[i].push_back(d);
                }
                if (i == 0ul) {
                    timestamps_.push_back(*d.get());
                }
            }
        }
        eos_ = true;

        // check if there is a non-empty stream
        bool all_empty = true;
        for (auto const& stream : data) {
            all_empty = all_empty and stream.empty();
        }
        if (all_empty) {
            return putData(p, Flow::Data::eos());
        }

        std::vector<std::pair<std::string, Tensor>> inputs;
        for (size_t i = 0ul; i < input_port_names_.size(); i++) {
            auto const& input_name  = input_port_names_[i];
            auto const& tensor_info = tensor_input_map_.get_info(input_name);
            inputs.push_back(std::make_pair(tensor_info.tensor_name(), toTensor(data[i])));
            if (not tensor_info.seq_length_tensor_name().empty()) {
                inputs.push_back(std::make_pair(tensor_info.seq_length_tensor_name(), Tensor::create(std::vector<s32>{static_cast<s32>(data[i].size())})));
            }
        }

        std::vector<Tensor> tf_output;
        session_.run(inputs, output_tensor_names_, {}, tf_output);

        for (size_t i = 0ul; i < tf_output.size(); i++) {
            appendToOutput(tf_output[i], start_frame, outputs_[i]);
        }
        auto timer_end = std::chrono::steady_clock::now();
        log("flow fwd time: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    }

    // the tensorflow graph is not required to have outputs of the same length, thus we have to check again here
    if (current_output_frame_[p] >= outputs_[p].size()) {
        return putData(p, Flow::Data::eos());
    }
    else {
        return putData(p, outputs_[p][current_output_frame_[p]++]);
    }
}

Tensor TensorflowForwardNode::toTensor(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const {
    if (data.front()->datatype() == Flow::Vector<f32>::type()) {
        return vectorToTensor<f32>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<f64>::type()) {
        return vectorToTensor<f64>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s64>::type()) {
        return vectorToTensor<s64>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u64>::type()) {
        return vectorToTensor<u64>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s32>::type()) {
        return vectorToTensor<s32>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u32>::type()) {
        return vectorToTensor<u32>(data);
    }
    else {
        criticalError("Unsupported input datatype: ") << *data.front()->datatype();
    }

    return Tensor();
}

template<typename T>
Tensor TensorflowForwardNode::vectorToTensor(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const {
    Flow::Vector<T>* first = dynamic_cast<Flow::Vector<T>*>(data.front().get());
    require(first != nullptr);

    std::vector<Math::FastMatrix<T>> batches;
    batches.emplace_back(first->size(), data.size());
    auto& matrix = batches.front();
    for (size_t i = 0ul; i < data.size(); i++) {
        Flow::Vector<T>* vec = dynamic_cast<Flow::Vector<T>*>(data[i].get());
        require(first != nullptr);
        require_eq(vec->size(), matrix.nRows());

        std::copy(vec->begin(), vec->end(), &matrix.at(0, i));
    }

    Tensor tensor;
    tensor.set(batches, true);
    return tensor;
}

void TensorflowForwardNode::appendToOutput(Tensor const& tensor, size_t start_frame, std::deque<Flow::Data*>& data, size_t drop_left, size_t drop_right) const {
    int         num_dims = tensor.numDims();
    std::string dt_name  = tensor.dataTypeName();
    require_eq(num_dims, 3);  // TODO: support for one dimensional outputs
    if (dt_name == "DT_FLOAT") {
        appendVectorsToOutput<f32>(tensor, start_frame, data, drop_left, drop_right);
    }
    else if (dt_name == "DT_DOUBLE") {
        appendVectorsToOutput<f64>(tensor, start_frame, data, drop_left, drop_right);
    }
    else if (dt_name == "DT_INT64") {
        appendVectorsToOutput<s64>(tensor, start_frame, data, drop_left, drop_right);
    }
    else if (dt_name == "DT_UINT64") {
        appendVectorsToOutput<u64>(tensor, start_frame, data, drop_left, drop_right);
    }
    else if (dt_name == "DT_INT32") {
        appendVectorsToOutput<s32>(tensor, start_frame, data, drop_left, drop_right);
    }
    else if (dt_name == "DT_UINT32") {
        appendVectorsToOutput<u32>(tensor, start_frame, data, drop_left, drop_right);
    }
    else {
        criticalError("Unsupported output datatype: ") << dt_name;
    }
}

template<typename T>
void TensorflowForwardNode::appendVectorsToOutput(Tensor const& tensor, size_t start_frame, std::deque<Flow::Data*>& data, size_t drop_left, size_t drop_right) const {
    require_ge(tensor.dimSize(2), 0);
    for (size_t t = drop_left; t + drop_right < static_cast<size_t>(tensor.dimSize(1)); t++) {
        Flow::Vector<T>* vec = new Flow::Vector<T>(static_cast<size_t>(tensor.dimSize(2)));
        tensor.get<T>(0ul, t, *vec);
        vec->setTimestamp(timestamps_[std::min(start_frame + t - drop_left, timestamps_.size() - 1ul)]);
        data.push_back(vec);
    }
}

// -----------------------------------------------------------------------------
//                     TensorflowOverlappingForwardNode
// -----------------------------------------------------------------------------

Core::ParameterInt TensorflowOverlappingForwardNode::paramContextSize_(
        "context-size", "Number of frames to discard at the left/right.", 0, 0);

Core::ParameterInt TensorflowOverlappingForwardNode::paramMaxBufferSize_(
        "max-buffer-size", "Maximum number of input features to be forwarded in one run.", 1000, 1);

TensorflowOverlappingForwardNode::TensorflowOverlappingForwardNode(Core::Configuration const& c)
        : Core::Component(c), Precursor(c), contextSize_(paramContextSize_(config)), maxBufferSize_(paramMaxBufferSize_(config)) {
    require_gt(maxBufferSize_, 2 * contextSize_);
}

Flow::PortId TensorflowOverlappingForwardNode::getInput(std::string const& name) {
    Flow::PortId port = Precursor::getInput(name);
    while (static_cast<Flow::PortId>(featureBuffer_.size()) <= port) {
        featureBuffer_.emplace_back();
    }
    return port;
}

bool TensorflowOverlappingForwardNode::setParameter(const std::string& name, const std::string& value) {
    if (paramId.match(name)) {
        leftContextSize_ = 0u;
        for (auto& fb : featureBuffer_) {
            fb.clear();
        }
    }
    return Precursor::setParameter(name, value);
}

bool TensorflowOverlappingForwardNode::work(Flow::PortId p) {
    // check if computation is needed
    require_lt(static_cast<size_t>(p), current_output_frame_.size());

    if (current_output_frame_[p] >= outputs_[p].size() and not eos_) {
        auto timer_start = std::chrono::steady_clock::now();
        // gather inputs for all input ports, assume they are of same length
        require_ge(timestamps_.size(), leftContextSize_);
        size_t start_frame = timestamps_.size() - leftContextSize_;
        while (featureBuffer_[0].size() < maxBufferSize_ and not eos_) {
            Flow::Timestamp                             ts;
            std::vector<Flow::DataPtr<Flow::Timestamp>> data;
            for (size_t i = 0ul; i < input_port_names_.size(); i++) {
                Flow::DataPtr<Flow::Timestamp> d;
                bool                           success = getData(i, d);
                if (success and Flow::Data::isNotSentinel(&(*d))) {
                    data.push_back(d);
                }
                if (i == 0ul) {
                    ts = *d.get();
                }
            }
            if (data.size() == input_port_names_.size()) {
                for (size_t i = 0ul; i < featureBuffer_.size(); i++) {
                    featureBuffer_[i].emplace_back(std::move(data[i]));
                }
                timestamps_.push_back(ts);
            }
            else {
                eos_ = true;
            }
        }

        if (eos_) {
            rightContextSize_ = 0u;
        }
        else {
            rightContextSize_ = contextSize_;
        }

        // check if there is data to process
        if (featureBuffer_[0].empty()) {
            return putData(p, Flow::Data::eos());
        }

        std::vector<std::pair<std::string, Tensor>> inputs;
        for (size_t i = 0ul; i < input_port_names_.size(); i++) {
            auto const& input_name  = input_port_names_[i];
            auto const& tensor_info = tensor_input_map_.get_info(input_name);
            inputs.push_back(std::make_pair(tensor_info.tensor_name(), toTensor(featureBuffer_[i])));
            if (not tensor_info.seq_length_tensor_name().empty()) {
                inputs.push_back(std::make_pair(tensor_info.seq_length_tensor_name(), Tensor::create(std::vector<s32>{static_cast<s32>(featureBuffer_[i].size())})));
            }
        }

        std::vector<Tensor> tf_output;
        session_.run(inputs, output_tensor_names_, {}, tf_output);

        for (size_t i = 0ul; i < tf_output.size(); i++) {
            appendToOutput(tf_output[i], start_frame, outputs_[i], leftContextSize_, rightContextSize_);
        }

        leftContextSize_ = rightContextSize_;
        for (size_t i = 0ul; i < featureBuffer_.size(); i++) {
            while (featureBuffer_[i].size() > 2 * leftContextSize_) {
                featureBuffer_[i].pop_front();
            }
        }

        auto timer_end = std::chrono::steady_clock::now();
        log("flow fwd time: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    }

    // the tensorflow graph is not required to have outputs of the same length, thus we have to check again here
    if (current_output_frame_[p] >= outputs_[p].size()) {
        return putData(p, Flow::Data::eos());
    }
    else {
        return putData(p, outputs_[p][current_output_frame_[p]++]);
    }
}

}  // namespace Tensorflow
