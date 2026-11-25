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

#ifndef _TORCH_SESSION_HH
#define _TORCH_SESSION_HH

#pragma push_macro("ensure")
#undef ensure
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

/*
 * Simple runtime wrapper for executing an exported Torch model package.
 * Session owns the low-level AOTI model runner and provides a simple positional tensor interface.
 */
class Session {
public:
    Session(const std::string& model_path);

    // Runs the model once with the given positional inputs
    bool run(const std::vector<torch::Tensor>& inputs, std::vector<torch::Tensor>& outputs);

private:
    // Low-level Torch AOTI runtime for the exported model package
    torch::inductor::AOTIModelPackageLoader runner_;
};

}  // namespace Torch

#endif  // _TORCH_SESSION_HH