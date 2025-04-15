<<<<<<<< HEAD:src/Onnx/Model.hh

/** Copyright 2020 RWTH Aachen University. All rights reserved.
========
/** Copyright 2025 RWTH Aachen University. All rights reserved.
>>>>>>>> master:src/Nn/LabelScorer/EncoderFactory.cc
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

<<<<<<<< HEAD:src/Onnx/Model.hh
#ifndef ONNX_MODEL_HH
#define ONNX_MODEL_HH

#include "IOSpecification.hh"
#include "Session.hh"
========
#include "EncoderFactory.hh"
>>>>>>>> master:src/Nn/LabelScorer/EncoderFactory.cc

namespace Onnx {

<<<<<<<< HEAD:src/Onnx/Model.hh
/*
 * Wrapper class that glues together session, IOMapping and IOValidator components
 * which are commonly used together when dealing with ONNX models
 */
class Model : public Core::Component {
public:
    Session   session;
    IOMapping mapping;

    Model(const Core::Configuration& config, const std::vector<IOSpecification>& ioSpec);
};

}  // namespace Onnx

#endif  // ONNX_MODEL_HH
========
EncoderFactory::EncoderFactory()
        : choices_(), paramEncoderType("type", &choices_, "Choice from a set of encoder types."), registry_() {}

void EncoderFactory::registerEncoder(const char* name, CreationFunction creationFunction) {
    choices_.addChoice(name, registry_.size());
    registry_.push_back(std::move(creationFunction));
}

Core::Ref<Encoder> EncoderFactory::createEncoder(Core::Configuration const& config) const {
    return registry_.at(paramEncoderType(config))(config);
}

}  // namespace Nn
>>>>>>>> master:src/Nn/LabelScorer/EncoderFactory.cc
