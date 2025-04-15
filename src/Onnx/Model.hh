/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef PYTHON_MODULE_HH
#define PYTHON_MODULE_HH

<<<<<<<< HEAD:src/Python/Module.hh
#include <Core/Singleton.hh>

namespace Python {

class Module_ {
public:
    Module_();
    ~Module_() = default;
};

typedef Core::SingletonHolder<Module_> Module;

}  // namespace Python

#endif  // PYTHON_MODULE_HH
========
#ifndef ONNX_MODEL_HH
#define ONNX_MODEL_HH

#include "IOSpecification.hh"
#include "Session.hh"

namespace Onnx {

/*
 * Wrapper class that ties together Session, IOMapping and IOValidator components
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
>>>>>>>> master:src/Onnx/Model.hh
