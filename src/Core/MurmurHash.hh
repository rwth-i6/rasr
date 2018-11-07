//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _CORE_MURMURHASH3_HH
#define _CORE_MURMURHASH3_HH

#include <stdint.h>

//-----------------------------------------------------------------------------

namespace Core {

void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out);
void MurmurHash3_x86_128(const void* key, int len, uint32_t seed, void* out);
void MurmurHash3_x64_128(const void* key, int len, uint32_t seed, void* out);

// convinience function, not part of the original MurmurHash collection
uint64_t MurmurHash3_x64_64(const void* key, const int len, const uint32_t seed);

}  // namespace Core

//-----------------------------------------------------------------------------

#endif  // _CORE_MURMURHASH3_HH
