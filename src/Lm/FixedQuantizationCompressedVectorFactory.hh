#ifndef _LM_FIXED_QUANTIZATION_COMPRESSED_VECTOR_FACTORY_HH
#define _LM_FIXED_QUANTIZATION_COMPRESSED_VECTOR_FACTORY_HH

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

template<typename T>
class QuantizedFloatVectorFixedBits : public CompressedVector<float> {
public:
    QuantizedFloatVectorFixedBits(float scale);

    virtual size_t        size() const;
    virtual float         get(size_t pos) const;
    virtual void          uncompress(float* data, size_t size) const;
    virtual void          uncompress(float* data, ContiguousBlockInfo const& block_info) const;
    virtual size_t        usedMemory() const;
    void                  store(float const* data, size_t size);
    void                  store(float const* data, ContiguousBlockInfo const& block_info);
    void                  clear();
    float                 scale() const;
    std::vector<T> const& data() const;

private:
    void uncompress_internal(float* data, size_t size, size_t pos) const;
    void store_internal(float const* data, size_t size, size_t pos);

    mutable std::vector<T> data_;
    float                  scale_;
};

using QuantizedFloatVector16Bits = QuantizedFloatVectorFixedBits<s16>;
using QuantizedFloatVector8Bits = QuantizedFloatVectorFixedBits<s8>;

class FixedQuantizationCompressedVectorFactory : public CompressedVectorFactory<float> {
public:
    using Precursor = CompressedVectorFactory<float>;

    static const Core::ParameterInt   paramBitsPerVal;
    static const Core::ParameterFloat paramEpsilon;

    FixedQuantizationCompressedVectorFactory(Core::Configuration const& config);
    virtual ~FixedQuantizationCompressedVectorFactory() = default;

    virtual CompressedVectorPtr<float> compress(float const* data, size_t size, CompressionParameters const* params) const;
    virtual CompressedVectorPtr<float> compress(float const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const;

private:
    unsigned bits_per_val_;
    float    epsilon_;
};

// inline implementations

template<typename T>
QuantizedFloatVectorFixedBits<T>::QuantizedFloatVectorFixedBits(float scale) : scale_(scale) {
}

template<typename T>
size_t QuantizedFloatVectorFixedBits<T>::size() const {
    return data_.size();
}

template<typename T>
float QuantizedFloatVectorFixedBits<T>::get(size_t pos) const {
    return data_[pos] * scale_;
}

template<typename T>
void QuantizedFloatVectorFixedBits<T>::uncompress(float* data, size_t size) const {
    require_ge(size, this->size());
    uncompress_internal(data, size, 0ul);
}

template<typename T>
void QuantizedFloatVectorFixedBits<T>::uncompress(float* data, ContiguousBlockInfo const& block_info) const {
    require_eq(block_info.totalSize(), this->size());
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        uncompress_internal(data + block_info.blockOffset(b), block_info.blockSize(), b * block_info.blockSize());
    }
}

template<typename T>
size_t QuantizedFloatVectorFixedBits<T>::usedMemory() const {
    return data_.capacity() * sizeof(typename decltype(data_)::value_type);
}

template<typename T>
void QuantizedFloatVectorFixedBits<T>::store(float const* data, size_t size) {
    data_.resize(size);
    store_internal(data, size, 0ul);
}

template<typename T>
void QuantizedFloatVectorFixedBits<T>::store(float const* data, ContiguousBlockInfo const& block_info) {
    data_.resize(block_info.totalSize());
    for (size_t b = 0ul; b < block_info.numBlocks(); b++) {
        store_internal(data + block_info.blockOffset(b), block_info.blockSize(), b * block_info.blockSize());
    }
}

template<typename T>
void QuantizedFloatVectorFixedBits<T>::clear() {
    data_.clear();
}

template<typename T>
float QuantizedFloatVectorFixedBits<T>::scale() const {
    return scale_;
}

template<typename T>
std::vector<T> const& QuantizedFloatVectorFixedBits<T>::data() const {
    return data_;
}

inline FixedQuantizationCompressedVectorFactory::FixedQuantizationCompressedVectorFactory(Core::Configuration const& config)
        : Precursor(config), bits_per_val_(paramBitsPerVal(config)), epsilon_(paramEpsilon(config)) {
    switch (bits_per_val_) {
        case 8:
        case 16:
            break;
        default:
            error("Only 8 and 16 bits are supported for fixed-quantization");
    }
}

}  // namespace Lm

#endif /* _LM_FIXED_QUANTIZATION_COMPRESSED_VECTOR_FACTORY_HH */
