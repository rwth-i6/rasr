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
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <Nn/CompressedVector.hh>

#include "Session.hh"

namespace Onnx {

/*
 * Pool of equally sized float slots in CUDA device memory, allocated in chunks and freed only with the pool.
 * Use the slots only on `Session::sharedCudaStream()`, so released slots can be reused right away.
 */
class DeviceSlotPool {
public:
    DeviceSlotPool(size_t slotSize, size_t slotsPerChunk);
    ~DeviceSlotPool();

    DeviceSlotPool(DeviceSlotPool const&)            = delete;
    DeviceSlotPool& operator=(DeviceSlotPool const&) = delete;

    // Number of floats per slot
    size_t slotSize() const;

    float* acquire();
    void   release(float* slot);

private:
    size_t const slotSize_;
    size_t const slotsPerChunk_;

    std::mutex          mutex_;
    std::vector<void*>  chunks_;
    std::vector<float*> freeSlots_;
};

/*
 * The `DeviceSlotPool`s of all slot sizes that are needed, created on first use.
 */
class DeviceSlotPools {
public:
    explicit DeviceSlotPools(size_t slotsPerChunk);

    std::shared_ptr<DeviceSlotPool> const& get(size_t slotSize);

private:
    size_t const slotsPerChunk_;

    std::unordered_map<size_t, std::shared_ptr<DeviceSlotPool>> pools_;  // by slot size
};

/*
 * CUDA device memory that grows on demand and is reused between calls, e.g. for tensors rebuilt in every step.
 * Use it only on `Session::sharedCudaStream()`: growing frees the old memory, which waits for pending work on it.
 */
class DeviceBuffer {
public:
    DeviceBuffer();
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer const&)            = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    // Device pointer to at least `bytes` bytes, the previous content is lost if the buffer grows
    void* allocate(size_t bytes);

private:
    void*  data_;
    size_t capacity_;  // in bytes
};

// `numFloats` rounded up to a multiple of 256 bytes, for tensors placed one after another in a `DeviceBuffer`
inline size_t alignDeviceFloats(size_t numFloats) {
    constexpr size_t alignment = 64ul;
    return (numFloats + alignment - 1ul) / alignment * alignment;
}

// Abort with a message if `status` is an error
void checkCudaStatus(cudaError_t status, char const* what);

// `Session::sharedCudaStream()` as cudaStream_t
inline cudaStream_t sharedStream() {
    return static_cast<cudaStream_t>(Session::sharedCudaStream());
}

/*
 * Uncompressed vector in a `DeviceSlotPool` slot, used by the GPU state managers.
 * The host accessors copy from the device (slow, only a fallback).
 */
class DeviceVector : public Nn::CompressedVector<float> {
public:
    // `size` must be 0 (empty vector, `pool` may be null) or the slot size of `pool`
    DeviceVector(std::shared_ptr<DeviceSlotPool> pool, size_t size);
    ~DeviceVector() override;

    size_t size() const override;
    float  get(size_t pos) const override;
    void   uncompress(float* data, size_t size) const override;
    void   uncompress(float* data, Nn::ContiguousBlockInfo const& blockInfo) const override;
    void   clear() override;
    // Device memory used by this vector
    size_t usedMemory() const override;

    // Device pointer of the data (nullptr for an empty vector)
    float* deviceData() const;

private:
    std::shared_ptr<DeviceSlotPool> pool_;
    float*                          data_;
    size_t                          size_;

    // Copy of the data in host memory
    std::vector<float> toHost() const;
};

}  // namespace Onnx

#endif  // _ONNX_DEVICE_VECTOR_HH
