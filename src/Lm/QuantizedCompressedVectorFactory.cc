#include "QuantizedCompressedVectorFactory.hh"

namespace Lm {

// --------------------------- QuantizedFloatVector ---------------------------

size_t QuantizedFloatVector::size() const {
    return stream_.size() / bits_per_val_;
}

float QuantizedFloatVector::get(size_t pos) const {
    unsigned val;
    stream_.seekg(pos * bits_per_val_);
    stream_.read(bits_per_val_, val);
    return min_val_ + val * interval_size_;
}

void QuantizedFloatVector::uncompress(float* data, size_t size) const {
    require_ge(size, this->size());
    stream_.seekg(0ul);
    uncompress_internal(data, size);
}

void QuantizedFloatVector::uncompress(float* data, ContiguousBlockInfo const& block_info) const {
    require_eq(block_info.totalSize(), this->size());
    stream_.seekg(0ul);
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        uncompress_internal(data + block_info.blockOffset(b), block_info.blockSize());
    }
}

size_t QuantizedFloatVector::usedMemory() const {
    return stream_.capacity() / 8;
}

void QuantizedFloatVector::store(float const* data, size_t size) {
    stream_.resize(size * bits_per_val_);
    stream_.seekp(0ul);
    store_internal(data, size);
}

void QuantizedFloatVector::store(float const* data, ContiguousBlockInfo const& block_info) {
    stream_.resize(block_info.totalSize() * bits_per_val_);
    stream_.seekp(0ul);
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        store_internal(data + block_info.blockOffset(b), block_info.blockSize());
    }
}

void QuantizedFloatVector::clear() {
    stream_.clear();
}

void QuantizedFloatVector::uncompress_internal(float* data, size_t size) const {
    for (size_t i = 0ul; i < size; i++) {
        unsigned val;
        stream_.read(bits_per_val_, val);
        data[i] = min_val_ + val * interval_size_;
    }
}

__attribute__((optimize("unroll-loops"))) void QuantizedFloatVector::store_internal(float const* data, size_t size) {
    float interval_inverse = 1.0f / interval_size_;
    float adj_min_val      = interval_inverse * min_val_ - 0.5;
#if defined(__AVX__)
    std::vector<unsigned> temp(size + 7);  // we need 32byte aligned data
    size_t                i                  = 0ul;
    __m256                m_interval_inverse = _mm256_broadcast_ss(&interval_inverse);
    __m256                m_adj_min_val      = _mm256_broadcast_ss(&adj_min_val);
    unsigned*             temp_data          = reinterpret_cast<unsigned*>(temp.data());
    temp_data += ((32 - (reinterpret_cast<uintptr_t>(temp_data) % 32)) % 32) / sizeof(unsigned);
    for (; i < (size - (size % 8)); i += 8) {
        __m256  raw         = _mm256_loadu_ps(data + i);
        __m256  quant_float = raw * m_interval_inverse - m_adj_min_val;
        __m256i quantized   = _mm256_cvttps_epi32(quant_float);
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp_data + i), quantized);
    }
    for (size_t j = 0ul; j < size % 8; j++, i++) {
        temp_data[i] = static_cast<int>(data[i] * interval_inverse - adj_min_val);
    }
    stream_.write(bits_per_val_, 0, temp_data, size);
#elif defined(__SSE3__)
    std::vector<unsigned> temp(size);
    size_t                i                  = 0ul;
    __m128                m_interval_inverse = _mm_load1_ps(&interval_inverse);
    __m128                m_adj_min_val      = _mm_load1_ps(&adj_min_val);
    unsigned*             temp_data          = temp.data();
    for (; i < (size - (size % 4)); i += 4) {
        __m128  raw         = _mm_loadu_ps(data + i);
        __m128  quant_float = raw * m_interval_inverse - m_adj_min_val;
        __m128i quantized   = _mm_cvttps_epi32(quant_float);
        _mm_store_si128(reinterpret_cast<__m128i*>(temp_data + i), quantized);
    }
    for (size_t j = 0ul; j < size % 4; j++, i++) {
        temp[i] = static_cast<int>(data[i] * interval_inverse - adj_min_val);
    }
    stream_.write(bits_per_val_, 0, temp.data(), temp.size());
#else
    std::vector<unsigned> temp(size);
    for (size_t i = 0ul; i < size; i++) {
        temp[i] = static_cast<int>(data[i] * interval_inverse - adj_min_val);
    }
    stream_.write(bits_per_val_, 0, temp.data(), temp.size());
#endif
}

// ------------------ QuantizedCompressionParameterEstimator ------------------

void QuantizedCompressionParameterEstimator::accumulate(float const* data, size_t size) {
    if (size > 0) {
        auto minmax = std::minmax_element(data, data + size);
        min_val_    = std::min(*minmax.first, min_val_);
        max_val_    = std::max(*minmax.second, max_val_);
    }
}

void QuantizedCompressionParameterEstimator::accumulate(float const* data, ContiguousBlockInfo const& block_info) {
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        accumulate(data + block_info.blockOffset(b), block_info.blockSize());
    }
}

CompressionParametersPtr QuantizedCompressionParameterEstimator::estimate() {
    return CompressionParametersPtr(new QuantizedCompressionParameters(min_val_, max_val_));
}

// --------------------- QuantizedCompressedVectorFactory ---------------------

const Core::ParameterInt QuantizedCompressedVectorFactory::paramBitsPerVal("bits-per-val",
                                                                           "Number of bits for the quantized value.",
                                                                           16, 1, 32);

CompressionParameterEstimatorPtr<float> QuantizedCompressedVectorFactory::getEstimator() const {
    return CompressionParameterEstimatorPtr<float>(new QuantizedCompressionParameterEstimator());
}

CompressedVectorPtr<float> QuantizedCompressedVectorFactory::compress(float const* data, size_t size, CompressionParameters const* params) const {
    QuantizedCompressionParameters const* qparams = dynamic_cast<QuantizedCompressionParameters const*>(params);
    require(qparams != nullptr);
    QuantizedFloatVector* vec = new QuantizedFloatVector(qparams->min_val, qparams->max_val, bits_per_val_);
    vec->store(data, size);
    return CompressedVectorPtr<float>(vec);
}

CompressedVectorPtr<float> QuantizedCompressedVectorFactory::compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const {
    QuantizedCompressionParameters const* qparams = dynamic_cast<QuantizedCompressionParameters const*>(params);
    require(qparams != nullptr);
    QuantizedFloatVector* vec = new QuantizedFloatVector(qparams->min_val, qparams->max_val, bits_per_val_);
    vec->store(data, block_info);
    return CompressedVectorPtr<float>(vec);
}

}  // namespace Lm
