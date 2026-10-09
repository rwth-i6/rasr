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

#include "Session.hh"

namespace Onnx {

namespace {

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        Core::Application::us()->criticalError("%s failed: %s", what, cudaGetErrorString(status));
    }
}

cudaStream_t sharedStream() {
    return static_cast<cudaStream_t>(Session::sharedCudaStream());
}

}  // namespace

/*
 * ===============================
 * ======= DeviceBlockPool =======
 * ===============================
 */

DeviceBlockPool::DeviceBlockPool(size_t blockSize, size_t blocksPerChunk)
        : blockSize_(blockSize),
          blocksPerChunk_(blocksPerChunk) {
    require_gt(blockSize_, 0ul);
    require_gt(blocksPerChunk_, 0ul);
}

DeviceBlockPool::~DeviceBlockPool() {
    // Errors are ignored here: at program exit the CUDA context may already be destroyed
    for (void* chunk : chunks_) {
        cudaFree(chunk);
    }
}

size_t DeviceBlockPool::blockSize() const {
    return blockSize_;
}

float* DeviceBlockPool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (freeBlocks_.empty()) {
        void* chunk = nullptr;
        checkCuda(cudaMalloc(&chunk, blocksPerChunk_ * blockSize_ * sizeof(float)), "Allocating device memory for states (cudaMalloc)");
        chunks_.push_back(chunk);
        freeBlocks_.reserve(freeBlocks_.size() + blocksPerChunk_);
        // In reverse order, so that blocks are handed out in increasing address order
        for (size_t i = blocksPerChunk_; i > 0ul; --i) {
            freeBlocks_.push_back(static_cast<float*>(chunk) + (i - 1ul) * blockSize_);
        }
    }
    float* block = freeBlocks_.back();
    freeBlocks_.pop_back();
    return block;
}

void DeviceBlockPool::release(float* block) {
    std::lock_guard<std::mutex> lock(mutex_);
    freeBlocks_.push_back(block);
}

/*
 * ============================
 * ======= DeviceVector =======
 * ============================
 */

DeviceVector::DeviceVector(std::shared_ptr<DeviceBlockPool> pool, size_t size)
        : pool_(std::move(pool)),
          data_(nullptr),
          size_(size) {
    if (size_ > 0ul) {
        require(pool_);
        require_eq(size_, pool_->blockSize());
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
        checkCuda(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream");
        checkCuda(cudaMemcpy(result.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost), "Copying a state from the device (cudaMemcpy)");
    }
    return result;
}

float DeviceVector::get(size_t pos) const {
    require_lt(pos, size_);
    return toHost()[pos];
}

void DeviceVector::uncompress(float* data, size_t size) const {
    require_ge(size, size_);
    auto host = toHost();
    std::copy(host.begin(), host.end(), data);
}

void DeviceVector::uncompress(float* data, Nn::ContiguousBlockInfo const& block_info) const {
    require_eq(block_info.totalSize(), size_);
    auto host = toHost();
    for (size_t i = 0ul; i < block_info.numBlocks(); i++) {
        size_t offset = i * block_info.blockSize();
        std::copy(host.begin() + offset, host.begin() + offset + block_info.blockSize(), data + block_info.blockOffset(i));
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
