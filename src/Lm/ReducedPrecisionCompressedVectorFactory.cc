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

size_t ReducedBitsFloatVector::usedMemory() const {
    return stream_.capacity() / 8;
}

void ReducedBitsFloatVector::store(float const* data, size_t size) {
    stream_.resize(size * bits_per_val_);
    stream_.seekp(0ul);
    stream_.write(bits_per_val_, drop_bits_, reinterpret_cast<unsigned const*>(data), size);
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

}  // namespace Lm
