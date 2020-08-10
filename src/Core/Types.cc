/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
// $Id$

#include <Core/Types.hh>

namespace Core {

constexpr const char* Type<s8>::name;
constexpr s8          Type<s8>::max;
constexpr s8          Type<s8>::min;

constexpr const char* Type<u8>::name;
constexpr u8          Type<u8>::max;
constexpr u8          Type<u8>::min;

constexpr const char* Type<u16>::name;
constexpr u16         Type<u16>::max;
constexpr u16         Type<u16>::min;

constexpr const char* Type<s16>::name;
constexpr s16         Type<s16>::max;
constexpr s16         Type<s16>::min;

constexpr const char* Type<u32>::name;
constexpr u32         Type<u32>::max;
constexpr u32         Type<u32>::min;

constexpr const char* Type<s32>::name;
constexpr s32         Type<s32>::max;
constexpr s32         Type<s32>::min;
constexpr s32         Type<s32>::epsilon;
constexpr s32         Type<s32>::delta;

constexpr const char* Type<u64>::name;
constexpr u64         Type<u64>::max;
constexpr u64         Type<u64>::min;

constexpr const char* Type<s64>::name;
constexpr s64         Type<s64>::max;
constexpr s64         Type<s64>::min;

#ifdef OS_darwin
constexpr const char* Type<size_t>::name;
constexpr const char* Type<ssize_t>::name;
static_assert(sizeof(size_t) == sizeof(Type<size_t>::max), "expected 64bit size_t");
static_assert(sizeof(ssize_t) == sizeof(Type<ssize_t>::max), "expected 64bit ssize_t");
#endif

constexpr const char* Type<f32>::name;
constexpr f32         Type<f32>::max;
constexpr f32         Type<f32>::min;
constexpr f32         Type<f32>::epsilon;
constexpr f32         Type<f32>::delta;

constexpr const char* Type<f64>::name;
constexpr f64         Type<f64>::max;
constexpr f64         Type<f64>::min;
constexpr f64         Type<f64>::epsilon;
constexpr f64         Type<f64>::delta;

template<size_t size>
void swapEndianess(void* buf, size_t count) {
    char* b = (char*)buf;
    for (size_t j = 0; j < size / 2; ++j)
        for (size_t i = 0; i < count; ++i)
            std::swap(b[i * size + j], b[i * size + size - j - 1]);
}

template void swapEndianess<2>(void* buf, size_t count);
template void swapEndianess<4>(void* buf, size_t count);
template void swapEndianess<8>(void* buf, size_t count);

}  // namespace Core
