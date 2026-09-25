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

#ifndef _TORCH_TENSOR_HH
#define _TORCH_TENSOR_HH

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

/*
 * Lightweight wrapper around a torch::Tensor.
 * Stores the tensor in a shared_ptr so views can safely share ownership of the underlying tensor memory.
 */
class Tensor {
public:
    Tensor(torch::Tensor tensor)
            : tensor_(std::make_shared<torch::Tensor>(std::move(tensor))) {}

    // Returns a raw pointer to the tensor data
    const float* data() const {
        return tensor_->data_ptr<float>();
    }

    // Returns the total number of elements in the tensor
    size_t numel() const {
        return tensor_->numel();
    }

    // Returns shared ownership of the wrapped tensor
    const std::shared_ptr<torch::Tensor>& ptr() const {
        return tensor_;
    }

private:
    std::shared_ptr<torch::Tensor> tensor_;
};

}  // namespace Torch

#endif  // _TORCH_TENSOR_HH