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
#include "PersistentInput.hh"

#include <Core/Application.hh>

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

namespace Onnx {

#ifdef MODULE_CUDA
namespace {

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        Core::Application::us()->criticalError("Onnx::PersistentInput: %s failed: %s", what, cudaGetErrorString(status));
    }
}

}  // namespace
#endif

PersistentInput::PersistentInput()
        : host_(),
          device_(nullptr),
          deviceCapacity_(0ul),
          deviceId_(0),
          uploaded_(false) {
}

PersistentInput::~PersistentInput() {
#ifdef MODULE_CUDA
    // Errors are ignored here: at program exit the CUDA context may already be destroyed
    if (device_ != nullptr) {
        cudaFree(device_);
    }
#endif
}

bool PersistentInput::empty() const {
    return host_.empty();
}

void PersistentInput::set(Value&& value) {
    require(not value.isOnDevice());
    host_     = std::move(value);
    uploaded_ = false;
}

void PersistentInput::clear() {
    host_     = Value();
    uploaded_ = false;
}

Value PersistentInput::view(Session const& session) {
    require(not empty());
    if (not session.usesCuda()) {
        return host_.view();
    }
#ifdef MODULE_CUDA
    if (not uploaded_) {
        upload();
    }
    Ort::MemoryInfo memoryInfo("Cuda", OrtDeviceAllocator, deviceId_, OrtMemTypeDefault);

    auto info  = host_.value_.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    return Value(Ort::Value::CreateTensor(memoryInfo, device_, host_.value_.GetTensorSizeInBytes(), shape.data(), shape.size(), info.GetElementType()));
#else
    defect();  // `usesCuda` is false without MODULE_CUDA
    return Value();
#endif
}

void PersistentInput::upload() {
#ifdef MODULE_CUDA
    size_t bytes = host_.value_.GetTensorSizeInBytes();
    require_gt(bytes, 0ul);
    if (bytes > deviceCapacity_) {
        // cudaFree waits for pending work on the old memory
        if (device_ != nullptr) {
            checkCuda(cudaFree(device_), "Freeing device memory (cudaFree)");
            device_         = nullptr;
            deviceCapacity_ = 0ul;
        }
        // The memory belongs to the current device, which the views refer to
        checkCuda(cudaGetDevice(&deviceId_), "Getting the current CUDA device");
        checkCuda(cudaMalloc(&device_, bytes), "Allocating device memory (cudaMalloc)");
        deviceCapacity_ = bytes;
    }
    // On the shared stream, so the copy is ordered after earlier runs that may still read the old data
    auto stream = static_cast<cudaStream_t>(Session::sharedCudaStream());
    checkCuda(cudaMemcpyAsync(device_, host_.value_.GetTensorRawData(), bytes, cudaMemcpyHostToDevice, stream), "Copying to the device (cudaMemcpyAsync)");
    // Wait for the copy, so the host tensor may be released or changed right away (it may be in pinned memory)
    checkCuda(cudaStreamSynchronize(stream), "Synchronizing the shared CUDA stream");
    uploaded_ = true;
#endif
}

}  // namespace Onnx
