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

#ifndef ONNX_SHARED_WEIGHTS_HH
#define ONNX_SHARED_WEIGHTS_HH

#include <cstddef>
#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "Session.hh"

namespace Onnx {

/*
 * Process-wide group of ONNX initializer tensors shared by multiple Ort::Sessions.
 * Sessions opt in by using the same non-empty weight-group name.
 *
 * Initializers are matched by dtype, shape and content, so equal weights are shared
 * even if they have different names in the different models. The shared tensors are
 * handed to ONNX Runtime via `AddInitializer`. ONNX Runtime only uses them if they
 * already reside on the device where it plans to place the initializer, therefore
 * the tensors of a group are created on the device of its execution provider
 * (CPU: memory-mapped from the model files, CUDA: device memory).
 * A group is released once the last session holding it is destroyed.
 *
 * Errors are reported by exceptions, like the ONNX Runtime C++ API used here does.
 */
class SharedWeightGroup {
public:
    struct Statistics {
        size_t newInitializers    = 0ul;
        size_t newBytes           = 0ul;
        size_t sharedInitializers = 0ul;
        size_t sharedBytes        = 0ul;
        size_t groupBytes         = 0ul;
    };

    // Returns the existing group of this name or creates it. All sessions of a group must use the same execution provider.
    static std::shared_ptr<SharedWeightGroup> get(std::string const& name, ExecutionProviderType provider);

    ~SharedWeightGroup();

    // Registers the initializers of the model in the session options, adding those not yet in the group
    Statistics addInitializers(std::string const& modelFile, Ort::SessionOptions& sessionOptions);

    OrtPrepackedWeightsContainer* prepackedWeightsContainer();

    /*
     * Drop the resident pages of memory-mapped initializers. Call this after creating a session:
     * ONNX Runtime copies most weights into pre-packed buffers, so the original pages are mostly not
     * needed anymore. Pages which are still used are read again from the page cache on access.
     */
    void releaseHostPages();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit SharedWeightGroup(ExecutionProviderType provider);
};

}  // namespace Onnx

#endif  // ONNX_SHARED_WEIGHTS_HH
