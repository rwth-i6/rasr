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
#ifndef _ONNX_DEVICE_VECTOR_HH
#define _ONNX_DEVICE_VECTOR_HH

#include <memory>
#include <mutex>
#include <vector>

#include <Nn/CompressedVector.hh>

namespace Onnx {

/*
 * Pool of equally sized float blocks in CUDA device memory, allocated in chunks and freed only with the pool.
 * Use the blocks only on `Session::sharedCudaStream()`, so released blocks can be reused right away.
 */
class DeviceBlockPool {
public:
    DeviceBlockPool(size_t blockSize, size_t blocksPerChunk);
    ~DeviceBlockPool();

    DeviceBlockPool(DeviceBlockPool const&)            = delete;
    DeviceBlockPool& operator=(DeviceBlockPool const&) = delete;

    // Number of floats per block
    size_t blockSize() const;

    float* acquire();
    void   release(float* block);

private:
    size_t const blockSize_;
    size_t const blocksPerChunk_;

    std::mutex          mutex_;
    std::vector<void*>  chunks_;
    std::vector<float*> freeBlocks_;
};

/*
 * Uncompressed vector in a `DeviceBlockPool` block, used by the `transformer-gpu` state manager.
 * The host accessors copy from the device (slow, only a fallback).
 */
class DeviceVector : public Nn::CompressedVector<float> {
public:
    // `size` must be 0 (empty vector, `pool` may be null) or the block size of `pool`
    DeviceVector(std::shared_ptr<DeviceBlockPool> pool, size_t size);
    ~DeviceVector() override;

    size_t size() const override;
    float  get(size_t pos) const override;
    void   uncompress(float* data, size_t size) const override;
    void   uncompress(float* data, Nn::ContiguousBlockInfo const& block_info) const override;
    void   clear() override;
    // Device memory used by this vector
    size_t usedMemory() const override;

    // Device pointer of the data (nullptr for an empty vector)
    float* deviceData() const;

private:
    std::shared_ptr<DeviceBlockPool> pool_;
    float*                           data_;
    size_t                           size_;

    // Copy of the data in host memory
    std::vector<float> toHost() const;
};

}  // namespace Onnx

#endif  // _ONNX_DEVICE_VECTOR_HH
