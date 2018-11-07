/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "TensorMap.hh"

namespace Tensorflow {

const Core::ParameterString TensorInfo::paramParamName("param-name",
                                                       "sprint internal name for this tensor",
                                                       "");

const Core::ParameterString TensorInfo::paramTensorName("tensor-name",
                                                        "name of the placeholder in the tensorflow graph",
                                                        "");

const Core::ParameterString TensorInfo::paramSeqLengthTensorName("seq-length-tensor-name",
                                                                 "name of the placeholder that holds sequence-length info",
                                                                 "");

template<typename Info>
TensorMap<Info>::TensorMap(Core::Configuration const& config)
        : Precursor(config) {
    bool     empty = true;
    unsigned i     = 0;
    // we will allow subconfigs to start at 0 or 1, afterwards they need to be continous
    while (not empty or i < 2) {
        Info info(select(std::string("info-") + std::to_string(i)));
        empty = info.param_name().empty();
        if (not empty) {
            tensor_infos_.insert(std::make_pair(info.param_name(), std::move(info)));
        }
        ++i;
    }
}

template TensorMap<TensorInputInfo>::TensorMap(Core::Configuration const&);
template TensorMap<TensorOutputInfo>::TensorMap(Core::Configuration const&);

}  // namespace Tensorflow
