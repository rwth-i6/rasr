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
#ifndef _LM_REDUCED_PRECISION_COMPRESSED_VECTOR_FACTORY_HH
#define _LM_REDUCED_PRECISION_COMPRESSED_VECTOR_FACTORY_HH

#include <Core/BitStream.hh>

#include "CompressedVector.hh"

namespace Lm {

class ReducedBitsFloatVector : public CompressedVector<float> {
public:
    ReducedBitsFloatVector(unsigned drop_bits)
            : drop_bits_(drop_bits), bits_per_val_(sizeof(float) * 8 - drop_bits_) {
    }

    virtual size_t size() const;
    virtual float  get(size_t pos) const;
    virtual void   uncompress(float* data, size_t size) const;
    virtual void   uncompress(float* data, ContiguousBlockInfo const& block_info) const;
    virtual size_t usedMemory() const;
    void           store(float const* data, size_t size);
    void           store(float const* data, ContiguousBlockInfo const& block_info);
    void           clear();

private:
    mutable Core::BitStream<unsigned> stream_;
    unsigned                          drop_bits_;
    unsigned                          bits_per_val_;
};

class ReducedPrecisionCompressedVectorFactory : public CompressedVectorFactory<float> {
public:
    using Precursor = CompressedVectorFactory<float>;

    static const Core::ParameterInt paramDropBits;

    ReducedPrecisionCompressedVectorFactory(Core::Configuration const& config)
            : Precursor(config), drop_bits_(paramDropBits(config)) {}
    virtual ~ReducedPrecisionCompressedVectorFactory() = default;

    virtual CompressedVectorPtr<float> compress(float const* data, size_t size, CompressionParameters const* params) const;
    virtual CompressedVectorPtr<float> compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const;

private:
    unsigned drop_bits_;
};

}  // namespace Lm

#endif /* _LM_REDUCED_PRECISION_COMPRESSED_VECTOR_FACTORY_HH */
