#ifndef _LM_DUMMY_COMPRESSED_VECTOR_FACTORY_HH
#define _LM_DUMMY_COMPRESSED_VECTOR_FACTORY_HH

#include "CompressedVector.hh"

namespace Lm {

template<typename T>
class UncompressedVector : public CompressedVector<T> {
public:
    virtual size_t size() const {
        return data_.size();
    }

    virtual T get(size_t pos) const {
        return data_.at(pos);
    }

    virtual void uncompress(T* data, size_t size) const {
        require_ge(size, this->size());
        std::copy(data_.begin(), data_.end(), data);
    }

    virtual size_t usedMemory() const {
        return data_.capacity() * sizeof(T);
    }

    void store(T const* data, size_t size) {
        data_.resize(size);
        std::copy(data, data + size, data_.begin());
    }

    void clear() {
        data_.clear();
    }

private:
    std::vector<T> data_;
};

template<typename T>
class DummyCompressedVectorFactory : public CompressedVectorFactory<T> {
public:
    using Precursor = CompressedVectorFactory<T>;

    DummyCompressedVectorFactory(Core::Configuration const& config)
            : Precursor(config) {}
    virtual ~DummyCompressedVectorFactory() = default;

    virtual CompressedVectorPtr<T> compress(T const* data, size_t size, CompressionParameters const* params) const {
        UncompressedVector<T>* vec = new UncompressedVector<T>();
        vec->store(data, size);
        return CompressedVectorPtr<T>(vec);
    }
};

}  // namespace Lm

#endif /* _LM_DUMMY_COMPRESSED_VECTOR_FACTORY_HH */
