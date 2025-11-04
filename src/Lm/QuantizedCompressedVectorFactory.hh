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
#ifndef _LM_QUANTIZED_COMPRESSED_VECTOR_FACTORY_HH
#define _LM_QUANTIZED_COMPRESSED_VECTOR_FACTORY_HH

#include <algorithm>
#include <limits>

#include <Core/BitStream.hh>

#include "CompressedVector.hh"

/**
 * Classes for vector compression using quantization. Quantization is done with unsigned
 * values. No clipping of the range of values is performed, i.e. the min and max value
 * will be restored correctly.
 */

namespace Lm {

class QuantizedFloatVector : public CompressedVector<float> {
public:
    QuantizedFloatVector(float min_val, float max_val, unsigned bits_per_val)
            : min_val_(min_val),
              interval_size_((max_val - min_val) / ((1ul << bits_per_val) - 1ul)),
              bits_per_val_(bits_per_val) {
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
    void uncompress_internal(float* data, size_t size) const;
    void store_internal(float const* data, size_t size);

    mutable Core::BitStream<unsigned> stream_;
    float                             min_val_;
    float                             interval_size_;
    unsigned                          bits_per_val_;
};

struct QuantizedCompressionParameters : public CompressionParameters {
    QuantizedCompressionParameters(float min_val, float max_val)
            : min_val(min_val),
              max_val(max_val) {}
    virtual ~QuantizedCompressionParameters() = default;

    float min_val;
    float max_val;
};

class QuantizedCompressionParameterEstimator : public CompressionParameterEstimator<float> {
public:
    QuantizedCompressionParameterEstimator()
            : min_val_(std::numeric_limits<float>::max()),
              max_val_(-std::numeric_limits<float>::max()) {}
    virtual ~QuantizedCompressionParameterEstimator() = default;

    virtual void                     accumulate(float const* data, size_t size);
    virtual void                     accumulate(float const* data, ContiguousBlockInfo const& block_info);
    virtual CompressionParametersPtr estimate();

private:
    float min_val_;
    float max_val_;
};

class QuantizedCompressedVectorFactory : public CompressedVectorFactory<float> {
public:
    using Precursor = CompressedVectorFactory<float>;

    static const Core::ParameterInt paramBitsPerVal;

    QuantizedCompressedVectorFactory(Core::Configuration const& config)
            : Precursor(config),
              bits_per_val_(paramBitsPerVal(config)) {}
    virtual ~QuantizedCompressedVectorFactory() = default;

    virtual CompressionParameterEstimatorPtr<float> getEstimator() const;
    virtual CompressedVectorPtr<float>              compress(float const* data, size_t size, CompressionParameters const* params) const;
    virtual CompressedVectorPtr<float>              compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const;

private:
    unsigned bits_per_val_;
};

}  // namespace Lm

#endif /* _LM_QUANTIZED_COMPRESSED_VECTOR_FACTORY_HH */
