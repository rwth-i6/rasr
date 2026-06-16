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

#ifndef PYTHON_ENCODER_HH
#define PYTHON_ENCODER_HH

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Nn/LabelScorer/Encoder.hh>

namespace py = pybind11;

namespace Python {

/*
 * Trampoline class that exposes Nn::Encoder to python.
 *
 * The native Encoder base class keeps the input/output buffers. Python subclasses
 * implement `encode(inputs)`, where `inputs` is a numpy array containing the
 * currently buffered frames. The return value is an iterable of
 * `(encoding, input_start, input_end)` tuples with input indices relative to the
 * supplied `inputs` array.
 */
class PythonEncoder : public Nn::Encoder {
public:
    using Precursor = Nn::Encoder;

    PythonEncoder(Core::Configuration const& config);
    virtual ~PythonEncoder() = default;

    virtual void reset() override;

    // Keep track of python object as a member to make sure it doesn't get garbage collected.
    void setInstance(py::object const& instance);

    py::object encodePythonInputs(py::array const& inputs);

    size_t numBufferedInputs() const {
        return inputBuffer_.size();
    }

    size_t inputOffset() const {
        return inputOffset_;
    }

    bool expectMoreFeatures() const {
        return expectMoreFeatures_;
    }

protected:
    virtual void encode() override;
    virtual void postEncodeCleanup() override;
    virtual bool canEncode() const override;

private:
    py::array_t<f32> bufferedInputsAsArray() const;

    py::object pyInstance_;
    size_t     inputOffset_;
};

}  // namespace Python

#endif  // PYTHON_ENCODER_HH
