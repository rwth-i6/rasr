#ifndef _LM_COMPRESSED_VECTOR_HH
#define _LM_COMPRESSED_VECTOR_HH

#include <memory>
#include <valarray>

#include <Core/Component.hh>

namespace Lm {

// utility class

class ContiguousBlockInfo {
public:
    ContiguousBlockInfo(std::gslice const& slice);
    ~ContiguousBlockInfo() = default;

    std::valarray<size_t> const& sizes() const;
    size_t                       totalSize() const;
    size_t                       numBlocks() const;
    size_t                       blockSize() const;

    size_t blockOffset(size_t idx) const;

private:
    size_t                start_;
    std::valarray<size_t> sizes_;
    std::valarray<size_t> strides_;
    size_t                totalSize_;
    size_t                numBlocks_;
    size_t                blockSize_;
    size_t                firstIdxDim_;
};

// abstract base classes

template<typename T>
class CompressedVector {
public:
    CompressedVector()          = default;
    virtual ~CompressedVector() = default;

    virtual size_t size() const                                                     = 0;
    virtual T      get(size_t pos) const                                            = 0;
    virtual void   uncompress(T* data, size_t size) const                           = 0;
    virtual void   uncompress(T* data, ContiguousBlockInfo const& block_info) const = 0;
    virtual void   clear()                                                          = 0;
    virtual size_t usedMemory() const                                               = 0;
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
    virtual void                     accumulate(U const* data, ContiguousBlockInfo const& block_info) {}
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
    virtual CompressedVectorPtr<T> compress(T const* data, size_t size, CompressionParameters const* params) const                           = 0;
    virtual CompressedVectorPtr<T> compress(T const* data, ContiguousBlockInfo const& block_info, CompressionParameters const* params) const = 0;
};
template<typename T>
using CompressedVectorFactoryPtr = std::unique_ptr<CompressedVectorFactory<T>>;

// inline implementations

inline std::valarray<size_t> const& ContiguousBlockInfo::sizes() const {
    return sizes_;
}

inline size_t ContiguousBlockInfo::totalSize() const {
    return totalSize_;
}

inline size_t ContiguousBlockInfo::numBlocks() const {
    return numBlocks_;
}

inline size_t ContiguousBlockInfo::blockSize() const {
    return blockSize_;
}

}  // namespace Lm

#endif /* _LM_COMPRESSED_VECTOR_HH */
