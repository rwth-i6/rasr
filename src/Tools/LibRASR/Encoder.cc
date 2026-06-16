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
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Nn/Module.hh>
#include <Python/Encoder.hh>

// Make it so that a `py::object` can use `Core::Ref` as a holder type instead of the usual `std::unique_ptr`.
// See https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers for official documentation.
PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

void registerPythonEncoder(std::string const& name, py::object const& pyEncoderClass) {
    Nn::Module::instance().encoderFactory().registerEncoder(
            name.c_str(),
            [pyEncoderClass](Core::Configuration const& config, Nn::ModelCache& modelCache) {
                py::gil_scoped_acquire gil;
                // Call constructor of `pyEncoderClass`.
                py::object inst = pyEncoderClass(config);
                inst.cast<Python::PythonEncoder*>()->setInstance(inst);
                return inst.cast<Core::Ref<Nn::Encoder>>();
            });
}

void bindEncoder(py::module_& module) {
    module.def(
            "register_encoder_type",
            &registerPythonEncoder,
            py::arg("name"),
            py::arg("encoder_cls"),
            "Register a custom encoder type in the internal encoder factory of RASR.\n\n"
            "Args:\n"
            "    name: The name under which the encoder type is registered. The same name must be used in the RASR config\n"
            "          in order to make RASR instantiate an encoder of this type later.\n"
            "    encoder_cls: A class that inherits from `librasr.Encoder` and implements `encode`.");

    py::class_<Nn::Encoder, Python::PythonEncoder, Core::Ref<Nn::Encoder>> pyEncoder(
            module,
            "Encoder",
            "Abstract base class for encoders. An encoder receives input features and produces encoded output frames.\n"
            "Python subclasses need to implement `encode(inputs)`, where `inputs` is a float32 numpy array of shape [T, F].\n"
            "`encode` must return an iterable of `(encoding, input_start, input_end)` tuples. `encoding` is a one-dimensional\n"
            "float32-compatible array, and `input_start`/`input_end` are relative to the supplied `inputs` array.");

    pyEncoder.def(
            py::init<Core::Configuration const&>(),
            py::arg("config"),
            "Construct an encoder from a RASR config.");

    pyEncoder.def(
            "reset",
            [](Nn::Encoder& self) { self.reset(); },
            "Reset any internal buffers and flags related to the current segment in order to prepare the encoder for a new segment.\n"
            "Python subclasses that override this method should call `super().reset()` before resetting Python-side state.");

    pyEncoder.def(
            "signal_no_more_features",
            &Nn::Encoder::signalNoMoreFeatures,
            "Signal to the encoder that all features for the current segment have been passed.");

    pyEncoder.def(
            "add_inputs",
            [](Nn::Encoder& self, py::array_t<f32, py::array::c_style | py::array::forcecast> inputs) {
                if (inputs.ndim() != 2) {
                    throw std::runtime_error("Encoder inputs must be a two-dimensional float32-compatible numpy array.");
                }
                self.addInputs(Nn::DataView(inputs, static_cast<size_t>(inputs.size())), static_cast<size_t>(inputs.shape(0)));
            },
            py::arg("inputs"),
            "Feed an array of input features to the encoder.\n\n"
            "Args:\n"
            "    inputs: A numpy array of shape [T, F] containing the input features for `T` time steps.");

    pyEncoder.def(
            "get_next_output",
            [](Nn::Encoder& self) -> py::object {
                auto output = self.getNextOutput();
                if (not output) {
                    return py::none();
                }

                py::array_t<f32> encoding({static_cast<ssize_t>(output->encoding.size())});
                std::copy(
                        output->encoding.data(),
                        output->encoding.data() + output->encoding.size(),
                        encoding.mutable_data());
                return py::make_tuple(encoding, output->input_start, output->input_end);
            },
            "Retrieve the next encoded output frame, or `None` if the encoder is not ready to produce one.");

    pyEncoder.def(
            "encode",
            [](Python::PythonEncoder& self, py::array const& inputs) {
                return self.encodePythonInputs(inputs);
            },
            py::arg("inputs"),
            "Encode the currently buffered input features. Python subclasses must override this method.");

    pyEncoder.def(
            "can_encode",
            [](Python::PythonEncoder& self, size_t numBufferedInputs, bool expectMoreFeatures) {
                return (numBufferedInputs > 0ul) and not expectMoreFeatures;
            },
            py::arg("num_buffered_inputs"),
            py::arg("expect_more_features"),
            "Return whether the encoder is ready to run. Python subclasses may override this method.");

    pyEncoder.def_property_readonly(
            "num_buffered_inputs",
            [](Python::PythonEncoder const& self) { return self.numBufferedInputs(); },
            "Number of input frames currently buffered in the native encoder.");

    pyEncoder.def_property_readonly(
            "input_offset",
            [](Python::PythonEncoder const& self) { return self.inputOffset(); },
            "Absolute segment index of the first frame in the currently buffered input array.");

    pyEncoder.def_property_readonly(
            "expect_more_features",
            [](Python::PythonEncoder const& self) { return self.expectMoreFeatures(); },
            "Whether more input features are expected for the current segment.");
}
