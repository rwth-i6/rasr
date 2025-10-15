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
#include "TFQuantizedBlasNceSoftmaxAdapter.hh"

#include <immintrin.h>

#include "FixedQuantizationCompressedVectorFactory.hh"

float quantized_dot_16bit(size_t size, float scale, s16 const* a, s16 const* b) {
#ifdef __AVX2__
    size_t  i         = 0ul;
    size_t  remainder = size % 16ul;
    __m256i acc       = _mm256_set1_epi32(0);
    for (; i < size - remainder; i += 16ul) {
        __m256i val_a = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(a + i));
        __m256i val_b = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(b + i));
        acc += _mm256_madd_epi16(val_a, val_b);
    }
    __m128i lower = _mm256_extracti128_si256(acc, 0);
    __m128i upper = _mm256_extracti128_si256(acc, 1);
    __m128i s     = _mm_hadd_epi32(lower, upper);
    s             = _mm_hadd_epi32(s, s);
    s             = _mm_hadd_epi32(s, s);
    s32 sum       = _mm_extract_epi32(s, 0);
    for (; i < size; i++) {
        sum += static_cast<s32>(a[i]) * static_cast<s32>(b[i]);
    }
    return sum * scale;
#else
    s32 sum = 0ul;
    for (size_t i = 0ul; i < size; i++) {
        sum += static_cast<s32>(a[i]) * static_cast<s32>(b[i]);
    }
    return sum * scale;
#endif
}

namespace Lm {

template<>
const Core::ParameterFloat TFQuantizedBlasNceSoftmaxAdapter16Bit::paramNNOutputEpsilon(
        "nn-output-epsilon",
        "if the nn-output vector is not quantized, use this scale for quantization",
        0.001, 0.0);

template<>
const Core::ParameterFloat TFQuantizedBlasNceSoftmaxAdapter16Bit::paramWeightsBiasEpsilon(
        "weights-bias-epsilon",
        "if the nn-output vector is not quantized, use this scale for quantization",
        0.001, 0.0);

template<>
Score TFQuantizedBlasNceSoftmaxAdapter16Bit::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    std::vector<s16>                  nn_output;
    s16 const*                        data;
    float                             scale;
    QuantizedFloatVector16Bits const* vec = dynamic_cast<QuantizedFloatVector16Bits const*>(nn_out.get());

    if (vec != nullptr) {
        data  = vec->data().data();
        scale = vec->scale();
    }
    else {
        float inv_scale = 1.0 / nnOutputEpsilon_;
        float min_val   = std::numeric_limits<s16>::min();
        float max_val   = std::numeric_limits<s16>::max();
        nn_output.resize(nn_out->size());
        std::vector<float> float_out(nn_out->size());
        nn_out->uncompress(float_out.data(), float_out.size());
        for (size_t i = 0ul; i < nn_output.size(); i++) {
            float val    = std::round(float_out[i] * inv_scale);
            val          = std::max(val, min_val);
            val          = std::min(val, max_val);
            nn_output[i] = static_cast<s16>(val);
        }
        data  = nn_output.data();
        scale = nnOutputEpsilon_;
    }

    return quantized_dot_16bit(nn_out->size(), weightsBiasEpsilon_ * scale, &weights_(0, output_idx), data) + bias_[output_idx];
}

}  // namespace Lm
