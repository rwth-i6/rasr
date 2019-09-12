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
    virtual size_t usedMemory() const;
    void           store(float const* data, size_t size);
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

private:
    unsigned drop_bits_;
};

}  // namespace Lm

#endif /* _LM_REDUCED_PRECISION_COMPRESSED_VECTOR_FACTORY_HH */
