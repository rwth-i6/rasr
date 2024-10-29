
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

#include "Model.hh"

namespace Onnx {

Model::Model(const Core::Configuration& config, const std::vector<IOSpecification>& ioSpec)
        : Core::Component(config),
          session(select("session")),
          mapping(select("io-map"), ioSpec) {
    IOValidator validator(select("validator"));
    validator.validate(ioSpec, mapping, session);
}

}  // namespace Onnx
