#ifndef CORE_BITSTREAM_HH
#define CORE_BITSTREAM_HH

#include <functional>
#include <vector>

#include <emmintrin.h>
#include <tmmintrin.h>

#include "Assertions.hh"

namespace Core {

template<typename T>
class BitStream {
public:
    BitStream();
    ~BitStream() = default;

    unsigned read(unsigned bits, T& val);
    unsigned write(unsigned bits, T val);
    unsigned write(unsigned bits, unsigned shift, T const* ary, size_t size);

    template<typename U>
    unsigned write_transformed(unsigned bits, U const* ary, size_t size, std::function<T(U)> transform);

    size_t tellg() const;
    void   seekg(size_t pos);
    size_t tellp() const;
    void   seekp(size_t pos);

    size_t size() const;
    size_t capacity() const;
    void   resize(size_t new_size);
    void   clear();

private:
    size_t         posg_;
    size_t         posp_;
    size_t         size_;
    std::vector<T> store_;

#if defined(__SSE3__) && defined(__SSSE3__)
    unsigned write_16_bits_aligned(T const* ary, size_t size);
    unsigned write_8_bits_aligned(T const* ary, size_t size);
#endif
};

// inline implementations

namespace {

template<typename T>
constexpr size_t bitsizeof() {
    return sizeof(T) * 8ul;
}

}  // namespace

template<typename T>
BitStream<T>::BitStream()
        : posg_(0ul), posp_(0ul), size_(0ul), store_() {
}

template<typename T>
unsigned BitStream<T>::read(unsigned bits, T& val) {
    require_le(bits, bitsizeof<T>());
    bits = static_cast<unsigned>(std::min<size_t>(size_ - posg_, bits));

    val = 0;

    unsigned total_read_bits = 0;
    while (bits > 0) {
        unsigned idx       = posg_ / bitsizeof<T>();
        unsigned skip_bits = posg_ % bitsizeof<T>();
        unsigned read_bits = static_cast<unsigned>(std::min<size_t>(bits, bitsizeof<T>() - skip_bits));

        T temp = store_[idx];
        temp   = temp >> skip_bits;
        temp &= ~(T(-1) << read_bits);
        val |= temp << total_read_bits;

        posg_ += read_bits;
        total_read_bits += read_bits;
        bits -= read_bits;
    }

    return total_read_bits;
}

template<typename T>
unsigned BitStream<T>::write(unsigned bits, T val) {
    if (bits == 0u) {
        return 0u;
    }
    require_le(bits, bitsizeof<T>());

    unsigned idx = posp_ / bitsizeof<T>();
    if (idx >= store_.size()) {
        store_.push_back(0);
    }
    unsigned skip_bits  = posp_ % bitsizeof<T>();
    unsigned write_bits = static_cast<unsigned>(std::min<size_t>(bits, bitsizeof<T>() - skip_bits));

    T write_bits_mask = ~(T(-1) << write_bits);
    T shifted_val     = (val & write_bits_mask) << skip_bits;
    T shifted_mask    = (T(-1) >> (bitsizeof<T>() - skip_bits)) | ~(write_bits_mask << skip_bits);
    store_[idx]       = (store_[idx] & shifted_mask) | shifted_val;

    bits -= write_bits;
    posp_ += write_bits;
    size_ = std::max(posp_, size_);
    return write_bits + write(bits, val >> write_bits);
}

template<typename T>
unsigned BitStream<T>::write(unsigned bits, unsigned shift, T const* ary, size_t size) {
    require(ary != nullptr or size == 0ul);
    if (bits == 0u or size == 0) {
        return 0u;
    }

    const unsigned last_idx = (posp_ + size * bits - 1) / bitsizeof<T>();
    if (last_idx >= store_.size()) {
        store_.resize(last_idx + 1, 0);
    }

#if defined(__SSE3__) && defined(__SSSE3__)
    if (shift == 0 and sizeof(T) == 4 and (reinterpret_cast<uintptr_t>(store_.data() + posp_ / (bits * sizeof(T))) % 16 == 0)) {
        if (bits == 8) {
            return write_8_bits_aligned(ary, size);
        }
        else if (bits == 16) {
            return write_16_bits_aligned(ary, size);
        }
    }
#endif

    require_le(bits, bitsizeof<T>());

    unsigned idx             = posp_ / bitsizeof<T>();
    unsigned skip_bits       = posp_ % bitsizeof<T>();
    const T  write_bits_mask = ~(T(-1) << bits);
    T        shifted_mask    = (T(-1) >> (bitsizeof<T>() - skip_bits)) | ~(write_bits_mask << skip_bits);
    T        write_buffer    = store_[idx] & shifted_mask;
    for (size_t i = 0ul; i < size; i++) {
        const unsigned write_bits = static_cast<unsigned>(std::min<size_t>(bits, bitsizeof<T>() - skip_bits));
        T              val        = ary[i] >> shift;
        write_buffer |= (val & write_bits_mask) << skip_bits;
        skip_bits += write_bits;
        if (skip_bits >= bitsizeof<T>()) {
            store_[idx] = write_buffer;
            idx += 1;
            write_buffer = T(0);
            skip_bits    = 0;
            if (write_bits < bits) {
                val          = val >> write_bits;
                write_buffer = val;
                skip_bits    = bits - write_bits;
            }
        }
    }
    if (skip_bits > 0) {
        store_[idx] = write_buffer;
    }
    posp_ += bits * size;
    size_ = std::max<size_t>(posp_, size_);
    return bits * size;
}

template<typename T>
template<typename U>
unsigned BitStream<T>::write_transformed(unsigned bits, U const* ary, size_t size, std::function<T(U)> transform) {
    if (bits == 0u) {
        return 0u;
    }
    require_le(bits, bitsizeof<T>());

    const unsigned last_idx = (posp_ + size * bits - 1) / bitsizeof<T>();
    if (last_idx >= store_.size()) {
        store_.resize(last_idx + 1, 0);
    }

    unsigned idx             = posp_ / bitsizeof<T>();
    unsigned skip_bits       = posp_ % bitsizeof<T>();
    const T  write_bits_mask = ~(T(-1) << bits);
    T        shifted_mask    = (T(-1) >> (bitsizeof<T>() - skip_bits)) | ~(write_bits_mask << skip_bits);
    T        write_buffer    = store_[idx] & shifted_mask;
    for (size_t i = 0ul; i < size; i++) {
        const unsigned write_bits = static_cast<unsigned>(std::min<size_t>(bits, bitsizeof<T>() - skip_bits));
        T              val        = transform(ary[i]);
        write_buffer |= (val & write_bits_mask) << skip_bits;
        skip_bits += write_bits;
        if (skip_bits >= bitsizeof<T>()) {
            store_[idx] = write_buffer;
            idx += 1;
            write_buffer = T(0);
            skip_bits    = 0;
            if (write_bits < bits) {
                val          = val >> write_bits;
                write_buffer = val;
                skip_bits    = bits - write_bits;
            }
        }
    }
    if (skip_bits > 0) {
        store_[idx] = write_buffer;
    }
    posp_ += bits * size;
    size_ = std::max<size_t>(posp_, size_);
    return bits * size;
}

template<typename T>
size_t BitStream<T>::tellg() const {
    return posg_;
}

template<typename T>
void BitStream<T>::seekg(size_t pos) {
    require_le(pos, size_);
    posg_ = pos;
}

template<typename T>
size_t BitStream<T>::tellp() const {
    return posp_;
}

template<typename T>
void BitStream<T>::seekp(size_t pos) {
    require_le(pos, size_);
    posp_ = pos;
}

template<typename T>
size_t BitStream<T>::size() const {
    return size_;
}

template<typename T>
size_t BitStream<T>::capacity() const {
    return store_.capacity() * sizeof(T) * 8;
}

template<typename T>
void BitStream<T>::resize(size_t new_size) {
    size_                        = new_size;
    size_t bits_per_storage_slot = sizeof(T) * 8;
    store_.resize((new_size + bits_per_storage_slot - 1) / bits_per_storage_slot);
    posg_ = std::min(posg_, new_size);
    posp_ = std::min(posp_, new_size);
}

template<typename T>
void BitStream<T>::clear() {
    store_.clear();
    posg_ = 0ul;
    posp_ = 0ul;
}

#if defined(__SSE3__) && defined(__SSSE3__)
template<typename T>
unsigned BitStream<T>::write_16_bits_aligned(T const* ary, size_t size) {
    unsigned  idx          = posp_ / bitsizeof<T>();
    T*        out          = store_.data() + idx;
    size_t    i            = 0ul;
    const int zero_out     = 0xFFFFFFFF;
    const int lower_2bytes = 0x05040100;
    const int upper_2bytes = 0x0D0C0908;
    __m128i   mask1        = _mm_set_epi32(zero_out, zero_out, upper_2bytes, lower_2bytes);
    __m128i   mask2        = _mm_set_epi32(upper_2bytes, lower_2bytes, zero_out, zero_out);
    for (; i < (size - size % 8); i += 8) {
        __m128i val1      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 0));
        __m128i val2      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 4));
        __m128i shuffled1 = _mm_shuffle_epi8(val1, mask1);
        __m128i shuffled2 = _mm_shuffle_epi8(val2, mask2);
        __m128i mix       = _mm_or_si128(shuffled1, shuffled2);
        _mm_store_si128(reinterpret_cast<__m128i*>(out + (i / 2)), mix);
    }
    if (size & 0x04) {
        __m128i val      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i));
        __m128i shuffled = _mm_shuffle_epi8(val, mask1);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + (i / 2)), shuffled);
        i += 4;
    }
    if (size & 0x02) {
        out[i / 2] = (ary[i] & 0xFFFF) | ((ary[i + 1] & 0xFFFF) << 16);
        i += 2;
    }
    if (size & 0x01) {
        out[i / 2] = ary[i] & 0xFFFF;
    }
    posp_ += 16 * size;
    return 16 * size;
}

template<typename T>
unsigned BitStream<T>::write_8_bits_aligned(T const* ary, size_t size) {
    unsigned  idx        = posp_ / bitsizeof<T>();
    T*        out        = store_.data() + idx;
    size_t    i          = 0ul;
    const int zero_out   = 0xFFFFFFFF;
    const int lower_byte = 0x0C080400;
    __m128i   mask1      = _mm_set_epi32(zero_out, zero_out, zero_out, lower_byte);
    __m128i   mask2      = _mm_set_epi32(zero_out, zero_out, lower_byte, zero_out);
    __m128i   mask3      = _mm_set_epi32(zero_out, lower_byte, zero_out, zero_out);
    __m128i   mask4      = _mm_set_epi32(lower_byte, zero_out, zero_out, zero_out);
    for (; i < (size - size % 16); i += 16) {
        __m128i val1      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 0));
        __m128i val2      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 4));
        __m128i val3      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 8));
        __m128i val4      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 12));
        __m128i shuffled1 = _mm_shuffle_epi8(val1, mask1);
        __m128i shuffled2 = _mm_shuffle_epi8(val2, mask2);
        __m128i shuffled3 = _mm_shuffle_epi8(val3, mask3);
        __m128i shuffled4 = _mm_shuffle_epi8(val4, mask4);
        __m128i mix1      = _mm_or_si128(shuffled1, shuffled2);
        __m128i mix2      = _mm_or_si128(shuffled3, shuffled4);
        __m128i mix3      = _mm_or_si128(mix1, mix2);
        _mm_store_si128(reinterpret_cast<__m128i*>(out + (i / 4)), mix3);
    }
    if (size & 0x08) {
        __m128i val1      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 0));
        __m128i val2      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i + 4));
        __m128i shuffled1 = _mm_shuffle_epi8(val1, mask1);
        __m128i shuffled2 = _mm_shuffle_epi8(val2, mask2);
        __m128i mix       = _mm_or_si128(shuffled1, shuffled2);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + (i / 4)), mix);
        i += 8;
    }
    if (size & 0x04) {
        __m128i val      = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ary + i));
        __m128i shuffled = _mm_shuffle_epi8(val, mask1);
        T       temp[4];
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&temp), shuffled);
        out[i / 4] = temp[0];
        i += 4;
    }
    if (size & 0x03) {
        T buf = T(0);
        for (size_t j = 0; j < size % 4; j++, i++) {
            buf |= (ary[i] & 0xFF) << (j * 8);
        }
        out[i / 4] = buf;
    }
    posp_ += 8 * size;
    return 8 * size;
}
#endif

}  // namespace Core

#endif /* CORE_BITSTREAM_HH */
