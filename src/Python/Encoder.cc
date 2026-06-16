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

#include "Encoder.hh"

#include <algorithm>
#include <stdexcept>

#include <pybind11/gil.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Python {

PythonEncoder::PythonEncoder(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          pyInstance_(),
          inputOffset_(0ul) {
}

void PythonEncoder::reset() {
    Precursor::reset();
    inputOffset_ = 0ul;

    py::gil_scoped_acquire gil;
    py::function           override = py::get_override(this, "reset");
    if (override) {
        override();
    }
}

void PythonEncoder::setInstance(py::object const& instance) {
    py::gil_scoped_acquire gil;
    pyInstance_ = instance;
}

py::array_t<f32> PythonEncoder::bufferedInputsAsArray() const {
    verify(not inputBuffer_.empty());

    ssize_t nTimesteps     = static_cast<ssize_t>(inputBuffer_.size());
    ssize_t featureDimSize = static_cast<ssize_t>(inputBuffer_.front().size());

    py::array_t<f32> inputArray({nTimesteps, featureDimSize});
    f32*             inputData = inputArray.mutable_data();

    for (ssize_t t = 0; t < nTimesteps; ++t) {
        verify(static_cast<ssize_t>(inputBuffer_[t].size()) == featureDimSize);
        std::copy(
                inputBuffer_[t].data(),
                inputBuffer_[t].data() + featureDimSize,
                inputData + t * featureDimSize);
    }

    return inputArray;
}

void PythonEncoder::encode() {
    if (inputBuffer_.empty()) {
        return;
    }

    py::gil_scoped_acquire gil;

    py::object pythonOutputs = encodePythonInputs(bufferedInputsAsArray());
    if (pythonOutputs.is_none()) {
        return;
    }

    for (py::handle pythonOutput : pythonOutputs) {
        if (not py::isinstance<py::sequence>(pythonOutput)) {
            throw std::runtime_error("Python encoder outputs must be tuples of (encoding, input_start, input_end).");
        }

        py::sequence output = py::reinterpret_borrow<py::sequence>(pythonOutput);
        if (output.size() != 3) {
            throw std::runtime_error("Python encoder outputs must be tuples of (encoding, input_start, input_end).");
        }

        py::array_t<f32, py::array::c_style | py::array::forcecast> encodingArray =
                py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(output[0]);
        if (encodingArray.ndim() != 1) {
            throw std::runtime_error("Python encoder output encoding must be one-dimensional.");
        }

        size_t inputStart = output[1].cast<size_t>();
        size_t inputEnd   = output[2].cast<size_t>();
        if (inputStart > inputEnd or inputEnd > inputBuffer_.size()) {
            throw std::runtime_error("Python encoder output span is outside the buffered input range.");
        }

        outputBuffer_.push_back(
                Nn::EncodedSpan{
                        .encoding    = {encodingArray, static_cast<size_t>(encodingArray.size())},
                        .input_start = inputOffset_ + inputStart,
                        .input_end   = inputOffset_ + inputEnd});
    }
}

void PythonEncoder::postEncodeCleanup() {
    inputOffset_ += inputBuffer_.size();
    Precursor::postEncodeCleanup();
}

bool PythonEncoder::canEncode() const {
    py::gil_scoped_acquire gil;
    py::function           override = py::get_override(const_cast<PythonEncoder*>(this), "can_encode");
    if (override) {
        return override(inputBuffer_.size(), expectMoreFeatures_).cast<bool>();
    }
    return Precursor::canEncode();
}

py::object PythonEncoder::encodePythonInputs(py::array const& inputs) {
    PYBIND11_OVERRIDE_PURE_NAME(
            py::object,
            Nn::Encoder,
            "encode",
            encodePythonInputs,
            inputs);
}

}  // namespace Python
