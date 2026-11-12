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
#include "DeviceVector.hh"

#include <algorithm>

#include <cuda_runtime.h>

#include <Core/Application.hh>

namespace Onnx {

void checkCudaStatus(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        Core::Application::us()->criticalError("%s failed: %s", what, cudaGetErrorString(status));
    }
}

/*
 * ==============================
 * ======= DeviceSlotPool =======
 * ==============================
 */

DeviceSlotPool::DeviceSlotPool(size_t slotSize, size_t slotsPerChunk)
        : slotSize_(slotSize),
          slotsPerChunk_(slotsPerChunk) {
    require_gt(slotSize_, 0ul);
    require_gt(slotsPerChunk_, 0ul);
}

DeviceSlotPool::~DeviceSlotPool() {
    // Errors are ignored here: at program exit the CUDA context may already be destroyed
    for (void* chunk : chunks_) {
        cudaFree(chunk);
    }
}

size_t DeviceSlotPool::slotSize() const {
    return slotSize_;
}

float* DeviceSlotPool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (freeSlots_.empty()) {
        void* chunk = nullptr;
        checkCudaStatus(cudaMalloc(&chunk, slotsPerChunk_ * slotSize_ * sizeof(float)), "Allocating device memory for states (cudaMalloc)");
        chunks_.push_back(chunk);
        freeSlots_.reserve(freeSlots_.size() + slotsPerChunk_);
        // In reverse order, so that slots are handed out in increasing address order
        for (size_t i = slotsPerChunk_; i > 0ul; --i) {
            freeSlots_.push_back(static_cast<float*>(chunk) + (i - 1ul) * slotSize_);
        }
    }
    float* slot = freeSlots_.back();
    freeSlots_.pop_back();
    return slot;
}

void DeviceSlotPool::release(float* slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    freeSlots_.push_back(slot);
}

/*
 * ===============================
 * ======= DeviceSlotPools =======
 * ===============================
 */

DeviceSlotPools::DeviceSlotPools(size_t slotsPerChunk)
        : slotsPerChunk_(slotsPerChunk) {
}

std::shared_ptr<DeviceSlotPool> const& DeviceSlotPools::get(size_t slotSize) {
    auto iter = pools_.find(slotSize);
    if (iter == pools_.end()) {
        iter = pools_.emplace(slotSize, std::make_shared<DeviceSlotPool>(slotSize, slotsPerChunk_)).first;
    }
    return iter->second;
}

/*
 * ============================
 * ======= DeviceBuffer =======
 * ============================
 */

DeviceBuffer::DeviceBuffer()
        : data_(nullptr),
          capacity_(0ul) {
}

DeviceBuffer::~DeviceBuffer() {
    // Errors are ignored here: at program exit the CUDA context may already be destroyed
    if (data_ != nullptr) {
        cudaFree(data_);
    }
}

void* DeviceBuffer::allocate(size_t bytes) {
    bytes = std::max(bytes, sizeof(float));
    if (capacity_ < bytes) {
        // Grow by at least half, so that slowly growing sizes don't reallocate in every step
        size_t const capacity = std::max(bytes, capacity_ + capacity_ / 2ul);
        if (data_ != nullptr) {
            // cudaFree waits for pending work on the buffer
            checkCudaStatus(cudaFree(data_), "Freeing device memory (cudaFree)");
            data_     = nullptr;
            capacity_ = 0ul;
        }
        checkCudaStatus(cudaMalloc(&data_, capacity), "Allocating device memory (cudaMalloc)");
        capacity_ = capacity;
    }
    return data_;
}

/*
 * ============================
 * ======= DeviceVector =======
 * ============================
 */

DeviceVector::DeviceVector(std::shared_ptr<DeviceSlotPool> pool, size_t size)
        : pool_(std::move(pool)),
          data_(nullptr),
          size_(size) {
    if (size_ > 0ul) {
        require(pool_);
        require_eq(size_, pool_->slotSize());
        data_ = pool_->acquire();
    }
}

DeviceVector::~DeviceVector() {
    clear();
}

size_t DeviceVector::size() const {
    return size_;
}

std::vector<float> DeviceVector::toHost() const {
    std::vector<float> result(size_);
    if (size_ > 0ul) {
        // The data may still be written on the shared stream
        checkCudaStatus(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream");
        checkCudaStatus(cudaMemcpy(result.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost), "Copying a state from the device (cudaMemcpy)");
    }
    return result;
}

float DeviceVector::get(size_t pos) const {
    require_lt(pos, size_);
    float result = 0.0f;
    // The data may still be written on the shared stream
    checkCudaStatus(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream");
    checkCudaStatus(cudaMemcpy(&result, data_ + pos, sizeof(float), cudaMemcpyDeviceToHost), "Copying a state element from the device (cudaMemcpy)");
    return result;
}

void DeviceVector::uncompress(float* data, size_t size) const {
    require_ge(size, size_);
    auto host = toHost();
    std::copy(host.begin(), host.end(), data);
}

void DeviceVector::uncompress(float* data, Nn::ContiguousBlockInfo const& blockInfo) const {
    require_eq(blockInfo.totalSize(), size_);
    auto host = toHost();
    for (size_t i = 0ul; i < blockInfo.numBlocks(); i++) {
        size_t offset = i * blockInfo.blockSize();
        std::copy(host.begin() + offset, host.begin() + offset + blockInfo.blockSize(), data + blockInfo.blockOffset(i));
    }
}

void DeviceVector::clear() {
    if (data_ != nullptr) {
        pool_->release(data_);
        data_ = nullptr;
    }
    size_ = 0ul;
}

size_t DeviceVector::usedMemory() const {
    return size_ * sizeof(float);
}

float* DeviceVector::deviceData() const {
    return data_;
}

}  // namespace Onnx
