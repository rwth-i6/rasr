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
    QuantizedFloatVector(float min_val, float max_val, unsigned bits_per_val) : min_val_(min_val),
                                                                                interval_size_((max_val - min_val) / ((1 << bits_per_val) - 1)),
                                                                                bits_per_val_(bits_per_val) {
    }

    virtual size_t size() const;
    virtual float get(size_t pos) const;
    virtual void uncompress(float* data, size_t size) const;
    virtual size_t usedMemory() const;
    void store(float const* data, size_t size);
    void clear();

private:
    mutable Core::BitStream<unsigned> stream_;
    float min_val_;
    float interval_size_;
    unsigned bits_per_val_;
};

struct QuantizedCompressionParameters : public CompressionParameters {
    QuantizedCompressionParameters(float min_val, float max_val) : min_val(min_val), max_val(max_val) {}
    virtual ~QuantizedCompressionParameters() = default;

    float min_val;
    float max_val;
};

class QuantizedCompressionParameterEstimator : public CompressionParameterEstimator<float> {
public:
    QuantizedCompressionParameterEstimator() : min_val_(std::numeric_limits<float>::max()),
                                               max_val_(-std::numeric_limits<float>::max()) {}
    virtual ~QuantizedCompressionParameterEstimator() = default;

    virtual void accumulate(float const* data, size_t size);
    virtual CompressionParametersPtr estimate();
private:
    float min_val_;
    float max_val_;
};

class QuantizedCompressedVectorFactory : public CompressedVectorFactory<float> {
public:
    using Precursor = CompressedVectorFactory<float>;

    static const Core::ParameterInt paramBitsPerVal;

    QuantizedCompressedVectorFactory(Core::Configuration const& config) : Precursor(config), bits_per_val_(paramBitsPerVal(config)) {}
    virtual ~QuantizedCompressedVectorFactory() = default;

    virtual CompressionParameterEstimatorPtr<float> getEstimator() const;
    virtual CompressedVectorPtr<float> compress(float const* data, size_t size, CompressionParameters const* params) const;
private:
    unsigned bits_per_val_;
};

}  // namespace Lm

#endif /* _LM_QUANTIZED_COMPRESSED_VECTOR_FACTORY_HH */
