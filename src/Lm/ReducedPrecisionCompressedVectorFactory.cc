/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#include "ReducedPrecisionCompressedVectorFactory.hh"

namespace Lm {

size_t ReducedBitsFloatVector::size() const {
    return stream_.size() / bits_per_val_;
}

float ReducedBitsFloatVector::get(size_t pos) const {
    unsigned val;
    stream_.seekg(pos * bits_per_val_);
    stream_.read(bits_per_val_, val);
    val = val << drop_bits_;
    return reinterpret_cast<float&>(val);
}

void ReducedBitsFloatVector::uncompress(float* data, size_t size) const {
    require_ge(size, this->size());
    stream_.seekg(0ul);
    for (size_t i = 0ul; i < this->size(); i++) {
        unsigned val;
        stream_.read(bits_per_val_, val);
        val     = val << drop_bits_;
        data[i] = reinterpret_cast<float&>(val);
    }
}

void ReducedBitsFloatVector::uncompress(float* data, ContiguousBlockInfo const& block_info) const {
    require_eq(block_info.totalSize(), this->size());
    stream_.seekg(0ul);
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        float* block_data = data + block_info.blockOffset(b);
        for (size_t i = 0ul; i < block_info.blockSize(); i++) {
            unsigned val;
            stream_.read(bits_per_val_, val);
            val           = val << drop_bits_;
            block_data[i] = reinterpret_cast<float&>(val);
        }
    }
}

size_t ReducedBitsFloatVector::usedMemory() const {
    return stream_.capacity() / 8;
}

void ReducedBitsFloatVector::store(float const* data, size_t size) {
    stream_.resize(size * bits_per_val_);
    stream_.seekp(0ul);
    stream_.write(bits_per_val_, drop_bits_, reinterpret_cast<unsigned const*>(data), size);
}

void ReducedBitsFloatVector::store(float const* data, ContiguousBlockInfo const& block_info) {
    stream_.resize(block_info.totalSize() * bits_per_val_);
    stream_.seekp(0ul);
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        float const* block_data = data + block_info.blockOffset(b);
        stream_.write(bits_per_val_, drop_bits_, reinterpret_cast<unsigned const*>(block_data), block_info.blockSize());
    }
}

void ReducedBitsFloatVector::clear() {
    stream_.clear();
}

const Core::ParameterInt ReducedPrecisionCompressedVectorFactory::paramDropBits("drop-bits",
                                                                                "How many bits to drop from the mantisse.",
                                                                                0, 0, 24);

CompressedVectorPtr<float> ReducedPrecisionCompressedVectorFactory::compress(float const* data, size_t size, CompressionParameters const* params) const {
    ReducedBitsFloatVector* vec = new ReducedBitsFloatVector(drop_bits_);
    vec->store(data, size);
    return CompressedVectorPtr<float>(vec);
}

CompressedVectorPtr<float> ReducedPrecisionCompressedVectorFactory::compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const {
    ReducedBitsFloatVector* vec = new ReducedBitsFloatVector(drop_bits_);
    vec->store(data, block_info);
    return CompressedVectorPtr<float>(vec);
}

}  // namespace Lm
