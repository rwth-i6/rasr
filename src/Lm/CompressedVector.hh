#ifndef _LM_COMPRESSED_VECTOR_HH
#define _LM_COMPRESSED_VECTOR_HH

#include <memory>

#include <Core/Component.hh>

namespace Lm {

// abstract base classes

template<typename T>
class CompressedVector {
public:
    CompressedVector()          = default;
    virtual ~CompressedVector() = default;

    virtual size_t size() const                           = 0;
    virtual T      get(size_t pos) const                  = 0;
    virtual void   uncompress(T* data, size_t size) const = 0;
    virtual void   clear();
    virtual size_t usedMemory() const = 0;
};
template<typename T>
using CompressedVectorPtr = std::unique_ptr<CompressedVector<T>>;

struct CompressionParameters {
public:
    CompressionParameters()          = default;
    virtual ~CompressionParameters() = default;
};
using CompressionParametersPtr = std::unique_ptr<CompressionParameters>;

template<typename U>
class CompressionParameterEstimator {
public:
    CompressionParameterEstimator()          = default;
    virtual ~CompressionParameterEstimator() = default;

    virtual void                     accumulate(U const* data, size_t size) {}
    virtual CompressionParametersPtr estimate() {
        return CompressionParametersPtr();
    }
};
template<typename U>
using CompressionParameterEstimatorPtr = std::unique_ptr<CompressionParameterEstimator<U>>;

template<typename T>
class CompressedVectorFactory : public Core::Component {
public:
    using Precursor = Core::Component;

    CompressedVectorFactory(Core::Configuration const& config)
            : Precursor(config) {}
    virtual ~CompressedVectorFactory() = default;

    virtual CompressionParameterEstimatorPtr<T> getEstimator() const {
        return CompressionParameterEstimatorPtr<T>(new CompressionParameterEstimator<T>());
    }
    virtual CompressedVectorPtr<T> compress(T const* data, size_t size, CompressionParameters const* params) const = 0;
};
template<typename T>
using CompressedVectorFactoryPtr = std::unique_ptr<CompressedVectorFactory<T>>;

}  // namespace Lm

#endif /* _LM_COMPRESSED_VECTOR_HH */
