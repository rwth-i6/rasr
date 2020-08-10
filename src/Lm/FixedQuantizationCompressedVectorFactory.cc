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
#include "FixedQuantizationCompressedVectorFactory.hh"

#include <immintrin.h>

namespace Lm {

// --------------------------- QuantizedFloatVector ---------------------------

template<>
void QuantizedFloatVector8Bits::uncompress_internal(float* data, size_t size, size_t pos) const {
#ifdef __AVX2__
    size_t remain = size % 16ul;
    {
        __m256      scale = _mm256_set1_ps(scale_);
        auto const* src   = data_.data() + pos;
        for (size_t i = 0ul; i < size - remain; i += 16ul) {
            __m128i data_8bit   = _mm_loadu_si128(reinterpret_cast<__m128i const*>(src + i));
            __m256i data_16bit  = _mm256_cvtepi8_epi16(data_8bit);
            __m128i lower       = _mm256_extracti128_si256(data_16bit, 0);
            __m128i upper       = _mm256_extracti128_si256(data_16bit, 1);
            __m256i lower_32bit = _mm256_cvtepi16_epi32(lower);
            __m256i upper_32bit = _mm256_cvtepi16_epi32(upper);
            __m256  lower_float = _mm256_cvtepi32_ps(lower_32bit) * scale;
            __m256  upper_float = _mm256_cvtepi32_ps(upper_32bit) * scale;
            _mm256_storeu_ps(data + i, lower_float);
            _mm256_storeu_ps(data + i + 8, upper_float);
        }
    }
    if (remain > 0) {
        for (size_t i = size - remain; i < size; i++) {
            data[i] = data_[pos + i] * scale_;
        }
    }
#else
    for (size_t i = 0ul; i < size; i++) {
        data[i] = data_[pos + i] * scale_;
    }
#endif
}

template<>
void QuantizedFloatVector8Bits::compress_internal(float const* data, size_t size, size_t pos) {
#ifdef __AVX2__
    size_t remain = size % 16ul;
    {
        __m256 inv_scale = _mm256_set1_ps(1.0f / scale_);
        __m256 min_val   = _mm256_set1_ps(std::numeric_limits<s8>::min());
        __m256 max_val   = _mm256_set1_ps(std::numeric_limits<s8>::max());
        auto*  dst       = data_.data() + pos;
        for (size_t i = 0ul; i < size - remain; i += 16ul) {
            __m256 val_a = _mm256_loadu_ps(data + i);
            val_a *= inv_scale;
            val_a = _mm256_min_ps(val_a, max_val);
            val_a = _mm256_max_ps(val_a, min_val);

            __m256 val_b = _mm256_loadu_ps(data + i + 8ul);
            val_b *= inv_scale;
            val_b = _mm256_min_ps(val_b, max_val);
            val_b = _mm256_max_ps(val_b, min_val);

            __m256i int_val_a       = _mm256_cvtps_epi32(val_a);
            __m128i lower_a         = _mm256_extracti128_si256(int_val_a, 0);
            __m128i upper_a         = _mm256_extracti128_si256(int_val_a, 1);
            __m128i int_val_16bit_a = _mm_packs_epi32(lower_a, upper_a);

            __m256i int_val_b       = _mm256_cvtps_epi32(val_b);
            __m128i lower_b         = _mm256_extracti128_si256(int_val_b, 0);
            __m128i upper_b         = _mm256_extracti128_si256(int_val_b, 1);
            __m128i int_val_16bit_b = _mm_packs_epi32(lower_b, upper_b);

            __m128i int_val_8bit = _mm_packs_epi16(int_val_16bit_a, int_val_16bit_b);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), int_val_8bit);
        }
    }
    if (remain > 0) {
        float inv_scale = 1.0f / scale_;
        float min_val   = std::numeric_limits<s8>::min();
        float max_val   = std::numeric_limits<s8>::max();
        for (size_t i = size - remain; i < size; i++) {
            float v = data[i];
            v *= inv_scale;
            v              = std::min(v, max_val);
            v              = std::max(v, min_val);
            data_[pos + i] = static_cast<s8>(v);
        }
    }
#else
    float inv_scale = 1.0f / scale_;
    float min_val   = std::numeric_limits<s8>::min();
    float max_val   = std::numeric_limits<s8>::max();
    for (size_t i = 0ul; i < size; i++) {
        float v = data[i];
        v *= inv_scale;
        v              = std::min(v, max_val);
        v              = std::max(v, min_val);
        data_[pos + i] = static_cast<s8>(v);
    }
#endif
}

template<>
void QuantizedFloatVector16Bits::uncompress_internal(float* data, size_t size, size_t pos) const {
#ifdef __AVX2__
    size_t remain = size % 8ul;
    {
        __m256      scale = _mm256_set1_ps(scale_);
        auto const* src   = data_.data() + pos;
        for (size_t i = 0ul; i < size - remain; i += 8ul) {
            __m128i data_16bit = _mm_loadu_si128(reinterpret_cast<__m128i const*>(src + i));
            __m256i data_32bit = _mm256_cvtepi16_epi32(data_16bit);
            __m256  data_float = _mm256_cvtepi32_ps(data_32bit);
            data_float *= scale;
            _mm256_storeu_ps(data + i, data_float);
        }
    }
    if (remain > 0) {
        for (size_t i = size - remain; i < size; i++) {
            data[i] = data_[pos + i] * scale_;
        }
    }
#else
    for (size_t i = 0ul; i < size; i++) {
        data[i] = data_[pos + i] * scale_;
    }
#endif
}

template<>
void QuantizedFloatVector16Bits::compress_internal(float const* data, size_t size, size_t pos) {
#ifdef __AVX2__
    size_t remain = size % 8ul;
    {
        __m256 inv_scale = _mm256_set1_ps(1.0f / scale_);
        __m256 min_val   = _mm256_set1_ps(std::numeric_limits<s16>::min());
        __m256 max_val   = _mm256_set1_ps(std::numeric_limits<s16>::max());
        auto*  dst       = data_.data() + pos;
        for (size_t i = 0ul; i < size - remain; i += 8ul) {
            __m256 val = _mm256_loadu_ps(data + i);
            val *= inv_scale;
            val                   = _mm256_min_ps(val, max_val);
            val                   = _mm256_max_ps(val, min_val);
            __m256i int_val       = _mm256_cvtps_epi32(val);
            __m128i lower         = _mm256_extracti128_si256(int_val, 0);
            __m128i upper         = _mm256_extracti128_si256(int_val, 1);
            __m128i int_val_16bit = _mm_packs_epi32(lower, upper);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), int_val_16bit);
        }
    }
    if (remain > 0) {
        float inv_scale = 1.0f / scale_;
        float min_val   = std::numeric_limits<s16>::min();
        float max_val   = std::numeric_limits<s16>::max();
        for (size_t i = size - remain; i < size; i++) {
            float v = data[i];
            v *= inv_scale;
            v              = std::min(v, max_val);
            v              = std::max(v, min_val);
            data_[pos + i] = static_cast<s16>(v);
        }
    }
#else
    float inv_scale = 1.0f / scale_;
    float min_val   = std::numeric_limits<s16>::min();
    float max_val   = std::numeric_limits<s16>::max();
    for (size_t i = 0ul; i < size; i++) {
        float v = data[i];
        v *= inv_scale;
        v              = std::min(v, max_val);
        v              = std::max(v, min_val);
        data_[pos + i] = static_cast<s16>(v);
    }
#endif
}

// --------------------- FixedQuantizationCompressedVectorFactory ---------------------

const Core::ParameterInt FixedQuantizationCompressedVectorFactory::paramBitsPerVal("bits-per-val",
                                                                                   "Number of bits for the quantized value.",
                                                                                   16, 8, 16);

const Core::ParameterFloat FixedQuantizationCompressedVectorFactory::paramEpsilon("epsilon",
                                                                                  "Distance between two quantized values.",
                                                                                  0.001, 0.0);

CompressedVectorPtr<float> FixedQuantizationCompressedVectorFactory::compress(float const* data, size_t size, CompressionParameters const* params) const {
    if (bits_per_val_ == 16) {
        QuantizedFloatVector16Bits* vec = new QuantizedFloatVector16Bits(epsilon_);
        vec->compress(data, size);
        return CompressedVectorPtr<float>(vec);
    }
    else if (bits_per_val_ == 8) {
        QuantizedFloatVector8Bits* vec = new QuantizedFloatVector8Bits(epsilon_);
        vec->compress(data, size);
        return CompressedVectorPtr<float>(vec);
    }
    defect();
}

CompressedVectorPtr<float> FixedQuantizationCompressedVectorFactory::compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const {
    if (bits_per_val_ == 16) {
        QuantizedFloatVector16Bits* vec = new QuantizedFloatVector16Bits(epsilon_);
        vec->compress(data, block_info);
        return CompressedVectorPtr<float>(vec);
    }
    else if (bits_per_val_ == 8) {
        QuantizedFloatVector8Bits* vec = new QuantizedFloatVector8Bits(epsilon_);
        vec->compress(data, block_info);
        return CompressedVectorPtr<float>(vec);
    }
    defect();
}

}  // namespace Lm
