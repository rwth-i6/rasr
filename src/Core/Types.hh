/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#ifndef _CORE_TYPES_HH
#define _CORE_TYPES_HH

#include <complex>
#include <cstring>
#include <string>
#include <vector>
#include <stdint.h>

typedef int8_t   s8;
typedef uint8_t  u8;
typedef int16_t  s16;
typedef uint16_t u16;
typedef int32_t  s32;
typedef uint32_t u32;
typedef int64_t  s64;
typedef uint64_t u64;

typedef float  f32;
typedef double f64;

namespace Core {

/** Static information about elementary types. */
template<class T>
struct Type {
    /** Name to be used to represent data type. */
    //static const char *name; // don't declare here to force compiler error on non-defined type

    /** Largest representable value of data type. */
    //static const T max;

    /**
     * Smallest representable value of data type.
     * Note that unlike std::numeric_limits<>::min this is the most negative
     * value also for floating point types.
     */
    //static const T min;

    /**
     * The difference between the smallest value greater than one and one.
     */
    //static const T epsilon;

    /**
     * Smallest representable value greater than zero.
     * For all integer types this is one.  For floating point
     * types this is the same as std::numeric_limits<>::min or
     * FLT_MIN / DBL_MIN.
     */
    //static const T delta;
};

template<>
struct Type<s8> {
    static constexpr const char* name = "s8";
    static constexpr s8          max  = 127;
    static constexpr s8          min  = -128;
};

template<>
struct Type<u8> {
    static constexpr const char* name = "u8";
    static constexpr u8          max  = 255U;
    static constexpr u8          min  = 0U;
};

template<>
struct Type<u16> {
    static constexpr const char* name = "u16";
    static constexpr u16         max  = 65535U;
    static constexpr u16         min  = 0U;
};

template<>
struct Type<s16> {
    static constexpr const char* name = "s16";
    static constexpr s16         max  = 32767;
    static constexpr s16         min  = -32768;
};

template<>
struct Type<u32> {
    static constexpr const char* name = "u32";
    static constexpr u32         max  = 4294967295U;
    static constexpr u32         min  = 0U;
};

template<>
struct Type<s32> {
    static constexpr const char* name    = "s32";
    static constexpr s32         max     = 2147483647;
    static constexpr s32         min     = -2147483647 - 1;  // gcc warns about too large int when -2147483648
    static constexpr s32         epsilon = 1;
    static constexpr s32         delta   = 1;
};

#if defined(HAS_64BIT)
template<>
struct Type<u64> {
    static constexpr const char* name = "u64";
    static constexpr u64         max  = 18446744073709551615U;
    static constexpr u64         min  = 0U;
};

template<>
struct Type<s64> {
    static constexpr const char* name = "s64";
    static constexpr s64         max  = 9223372036854775807LL;
    static constexpr s64         min  = -9223372036854775807LL - 1;
};
#endif

#ifdef OS_darwin
// I don't quite understand why these are different types
// than any other numeric types.
// It seems as if
//   ssize_t == long and long != s64 and long != s32 .

template<>
struct Type<size_t> : Type<u64> {
    static constexpr const char* name = "size_t";
};

template<>
struct Type<ssize_t> : Type<s64> {
    static constexpr const char* name = "ssize_t";
};
#endif

template<>
struct Type<f32> {
    static constexpr const char* name    = "f32";
    static constexpr f32         max     = +3.40282347e+38F;
    static constexpr f32         min     = -3.40282347e+38F;
    static constexpr f32         epsilon = 1.19209290e-07F;
    static constexpr f32         delta   = 1.17549435e-38F;
};

template<>
struct Type<f64> {
    static constexpr const char* name    = "f64";
    static constexpr f64         max     = +1.7976931348623157e+308;
    static constexpr f64         min     = -1.7976931348623157e+308;
    static constexpr f64         epsilon = 2.2204460492503131e-16;
    static constexpr f64         delta   = 2.2250738585072014e-308;
};

/**
 *  Use this class for naming your basic classes.
 *  Creating new names: by specialization.
 *  @see example Matrix.hh
 */
template<typename T>
class NameHelper : public std::string {
public:
    NameHelper()
            : std::string(Type<T>::name) {}
};

template<>
class NameHelper<std::string> : public std::string {
public:
    NameHelper()
            : std::string("string") {}
};

template<>
class NameHelper<bool> : public std::string {
public:
    NameHelper()
            : std::string("bool") {}
};

template<typename T>
class NameHelper<std::complex<T>> : public std::string {
public:
    NameHelper()
            : std::string(std::string("complex-") + NameHelper<T>()) {}
};

template<typename T>
class NameHelper<std::vector<T>> : public std::string {
public:
    NameHelper()
            : std::string(std::string("vector-") + NameHelper<T>()) {}
};

/**
 * Change endianess of a block of data.
 * The size argument is given as a template parameter, so the
 * compiler can unroll the loop.
 * @param buf pointer to an array of data
 * @param size size of the element data type in bytes.
 * @param count number of elements
 */
template<size_t size>
void swapEndianess(void* buf, size_t count = 1);
template<>
inline void swapEndianess<1>(void* buf, size_t count) {}

}  // namespace Core

namespace std {

template<>
inline bool equal(std::vector<u8>::const_iterator b1, std::vector<u8>::const_iterator e1, std::vector<u8>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(u8) * (e1 - b1)) == 0);
}
template<>
inline bool equal(std::vector<s8>::const_iterator b1, std::vector<s8>::const_iterator e1, std::vector<s8>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(s8) * (e1 - b1)) == 0);
}

template<>
inline bool equal(std::vector<u16>::const_iterator b1, std::vector<u16>::const_iterator e1, std::vector<u16>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(u16) * (e1 - b1)) == 0);
}
template<>
inline bool equal(std::vector<s16>::const_iterator b1, std::vector<s16>::const_iterator e1, std::vector<s16>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(s16) * (e1 - b1)) == 0);
}

template<>
inline bool equal(std::vector<u32>::const_iterator b1, std::vector<u32>::const_iterator e1, std::vector<u32>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(u32) * (e1 - b1)) == 0);
}
template<>
inline bool equal(std::vector<s32>::const_iterator b1, std::vector<s32>::const_iterator e1, std::vector<s32>::const_iterator b2) {
    return (memcmp(&(*b1), &(*b2), sizeof(s32) * (e1 - b1)) == 0);
}

}  // namespace std

#endif  // _CORE_TYPES_HH
