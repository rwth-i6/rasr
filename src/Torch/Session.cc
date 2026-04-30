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

#include "Session.hh"

#pragma push_macro("ensure")
#undef ensure
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

Session::Session(const std::string& model_path)
        : runner_(model_path) {}

bool Session::run(const std::vector<torch::Tensor>& inputs, std::vector<torch::Tensor>& outputs) {
    outputs = runner_.run(inputs);
    return true;
}

}  // namespace Torch