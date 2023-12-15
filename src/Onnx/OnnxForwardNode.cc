/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "OnnxForwardNode.hh"
#include <chrono>

#include <Flow/Vector.hh>

#include "IOSpecification.hh"
#include "Util.hh"

namespace {

}  // namespace

namespace Onnx {

// -----------------------------------------------------------------------------
//                               OnnxForwardNode
// -----------------------------------------------------------------------------

Core::ParameterString OnnxForwardNode::paramId(
        "id", "Changing the id resets the caches for the recurrent connections.");

OnnxForwardNode::OnnxForwardNode(Core::Configuration const& c)
        : Core::Component(c),
          Precursor(c),
          computation_done_(false),
          session_(select("session")),
          mapping_(select("io-map"), ioSpec_),
          validator_(select("validator")),
          features_onnx_name_(mapping_.getOnnxName("features")),
          features_size_onnx_name_(mapping_.getOnnxName("features-size")),
          output_onnx_names_({mapping_.getOnnxName("output")}),
          current_output_frame_(0) {
    bool valid = validator_.validate(ioSpec_, mapping_, session_);
    if (not valid) {
        warning("Failed to validate input model.");
    }
}

const std::vector<Onnx::IOSpecification> OnnxForwardNode::ioSpec_ = {
        Onnx::IOSpecification{
                "features",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "features-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "output",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

bool OnnxForwardNode::setParameter(const std::string& name, const std::string& value) {
    if (paramId.match(name)) {
        // New id means we entered a new segment
        // => Reset node and clear computation cache
        computation_done_ = false;
        timestamps_.clear();
        output_cache_.clear();
        current_output_frame_ = 0;
    }
    return true;
}

bool OnnxForwardNode::work(Flow::PortId p) {
    // Only one output port
    require_eq(static_cast<size_t>(p), 0);

    // Perform computation if not yet done (only run computation once per segment)
    if (not computation_done_) {
        computation_done_ = true;

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
        log("Create inputs");
        std::vector<std::pair<std::string, Value>> inputs;
        inputs.emplace_back(features_onnx_name_, toValue(data));
        log("Data (") << features_onnx_name_ << "): ";
        if (mapping_.hasOnnxName("features-size")) {
            log("Size (") << features_size_onnx_name_ << "): ";
            inputs.emplace_back(features_size_onnx_name_,
                                Value::create(std::vector<s32>{static_cast<s32>(data.size())}));
        }

        // Run session to compute outputs
        auto t_start = std::chrono::steady_clock::now();

        std::vector<Value> session_outputs;
        session_.run(std::move(inputs), output_onnx_names_, session_outputs);

        // Print AM timing statistics
        auto t_end     = std::chrono::steady_clock::now();
        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        log("num_frames: %zu elapsed: %f AM_RTF: %f",
               data.size(), t_elapsed, t_elapsed / (static_cast<double>(data.size()) / 100.0));

        // Append session outputs to cache
        appendToOutput(session_outputs[0]);

        // Print overall timing statistics
        auto timer_end = std::chrono::steady_clock::now();
        log("flow fwd time: ") << std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    }

    // If we have not-yet-returned outputs available, send one of them
    if (current_output_frame_ < output_cache_.size()) {
        return putData(p, output_cache_[current_output_frame_++]);
    }

    // All available outputs have been returned -> reached EOS
    return putData(p, Flow::Data::eos());
}

Value OnnxForwardNode::toValue(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const {
    if (data.front()->datatype() == Flow::Vector<f32>::type()) {
        return vectorToValue<f32>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<f64>::type()) {
        return vectorToValue<f64>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s8>::type()) {
        return vectorToValue<s8>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u8>::type()) {
        return vectorToValue<u8>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s16>::type()) {
        return vectorToValue<s16>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u16>::type()) {
        return vectorToValue<u16>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s32>::type()) {
        return vectorToValue<s32>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u32>::type()) {
        return vectorToValue<u32>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<s64>::type()) {
        return vectorToValue<s64>(data);
    }
    else if (data.front()->datatype() == Flow::Vector<u64>::type()) {
        return vectorToValue<u64>(data);
    }
    else {
        criticalError("Unsupported input datatype: ") << *data.front()->datatype();
    }

    return {};
}

template<typename T>
Value OnnxForwardNode::vectorToValue(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const {
    // Vector of first time step. Required to get feature size.
    auto* first = dynamic_cast<Flow::Vector<T>*>(data.front().get());
    require(first != nullptr);

    std::vector<Math::FastMatrix<T>> batches(1);    // single "batch"
    batches[0].resize(first->size(), data.size());  // F x T matrix (col-major)
    auto& matrix = batches[0];
    for (size_t t = 0ul; t < data.size(); t++) {
        // Create Flow::Vector from Flow::DataPtr at time t
        auto* vec = dynamic_cast<Flow::Vector<T>*>(data[t].get());
        require(first != nullptr);
        require_eq(vec->size(), matrix.nRows());

        // Store data from matrix column into vector
        std::copy(vec->begin(), vec->end(), &matrix.at(0, t));
    }

    return Value::create(batches, true);
}

void OnnxForwardNode::appendToOutput(Value const& value) {
    int         num_dims = value.numDims();
    std::string dt_name  = value.dataTypeName();
    require_eq(num_dims, 3);
    if (dt_name == "float") {
        appendVectorsToOutput<f32>(value);
    }
    else if (dt_name == "double") {
        appendVectorsToOutput<f64>(value);
    }
    else if (dt_name == "int8") {
        appendVectorsToOutput<s8>(value);
    }
    else if (dt_name == "uint8") {
        appendVectorsToOutput<u8>(value);
    }
    else if (dt_name == "int16") {
        appendVectorsToOutput<s16>(value);
    }
    else if (dt_name == "uint16") {
        appendVectorsToOutput<u16>(value);
    }
    else if (dt_name == "int32") {
        appendVectorsToOutput<s32>(value);
    }
    else if (dt_name == "uint32") {
        appendVectorsToOutput<u32>(value);
    }
    else if (dt_name == "int64") {
        appendVectorsToOutput<s64>(value);
    }
    else if (dt_name == "uint64") {
        appendVectorsToOutput<u64>(value);
    }
    else {
        criticalError("Unsupported output datatype: ") << dt_name;
    }
}

template<typename T>
void OnnxForwardNode::appendVectorsToOutput(Value const& value) {
    require_ge(value.dimSize(2), 0);
    for (size_t t = 0ul; t < static_cast<size_t>(value.dimSize(1)); t++) {
        // Create Flow::Vector from value content and set timestamps
        auto* vec = new Flow::Vector<T>(static_cast<size_t>(value.dimSize(2)));
        value.get<T>(0ul, t, *vec);
        vec->setTimestamp(timestamps_[std::min(t, timestamps_.size() - 1ul)]);

        // Add to output cache
        output_cache_.push_back(vec);
    }
}

}  // namespace Onnx
