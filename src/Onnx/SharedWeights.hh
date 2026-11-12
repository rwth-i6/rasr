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
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace Onnx {

class SharedWeightGroup;

enum class SharedWeightDevice {
    cpu,
    cuda
};

struct SharedWeightSessionStats {
    struct InitializerInfo {
        std::string name;
        size_t      bytes = 0;
    };

    size_t                       parsedInitializers     = 0;
    size_t                       skippedInitializers    = 0;
    size_t                       registeredInitializers = 0;
    size_t                       newInitializers        = 0;
    size_t                       reusedInitializers     = 0;
    size_t                       registeredBytes        = 0;
    size_t                       newBytes               = 0;
    size_t                       reusedBytes            = 0;
    size_t                       groupInitializers      = 0;
    size_t                       groupBytes             = 0;
    std::vector<InitializerInfo> largestNewInitializers;
};

/*
 * Process-wide registry for ONNX initializer tensors shared by multiple
 * Ort::Sessions. Sessions opt in by using the same non-empty weight-group name.
 *
 * Initializers are matched by dtype, shape and content, so equal weights are shared
 * even if they have different names in the different models. The shared tensors are
 * handed to ONNX Runtime via `AddInitializer`. ONNX Runtime only uses them if they
 * already reside on the device where it plans to place the initializer, therefore
 * the tensors of a group are created on the device of the sessions using it
 * (CPU: memory-mapped from the model files, CUDA: device memory).
 * A group is released once the last session holding it is destroyed.
 */
class SharedWeightRegistry {
public:
    static SharedWeightRegistry& instance();

    std::shared_ptr<SharedWeightGroup> prepareSessionOptions(std::string const&        groupName,
                                                             std::string const&        modelFile,
                                                             SharedWeightDevice        device,
                                                             Ort::SessionOptions&      sessionOptions,
                                                             SharedWeightSessionStats* stats = nullptr);

private:
    SharedWeightRegistry() = default;
};

class SharedWeightGroup {
public:
    OrtPrepackedWeightsContainer* prepackedWeightsContainer();

    /*
     * Drop the resident pages of memory-mapped CPU initializers. Call this after creating a session:
     * ONNX Runtime copies most weights into pre-packed buffers, so the original pages are mostly not
     * needed anymore. Pages which are still used are read again from the page cache on access.
     */
    void releaseHostPages();

private:
    friend class SharedWeightRegistry;

    struct Impl;
    std::shared_ptr<Impl> impl_;

    explicit SharedWeightGroup(std::shared_ptr<Impl> impl);
};

}  // namespace Onnx

#endif  // ONNX_SHARED_WEIGHTS_HH
