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
#ifndef _CORE_HASH_HH
#define _CORE_HASH_HH

#include <cstdint>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Core {

// Auxiliary function to merge multiple hashes into one via the boost way
// See https://www.boost.org/doc/libs/1_43_0/doc/html/hash/reference.html#boost.hash_combine
inline size_t combineHashes(size_t hash1, size_t hash2) {
    if (hash1 == 0ul) {
        return hash2;
    }
    if (hash2 == 0ul) {
        return hash1;
    }
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
}

template<class Key>
struct StandardValueHash {
    inline uint32_t operator()(Key a) const {
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0xfd7046c5) + (a << 3);
        return a;
    }
};

template<class T>
struct SetHash {
    size_t operator()(const std::set<T>& set) const {
        size_t a = set.size();
        a        = (a ^ 0xc761c23c) ^ (a >> 19);
        a        = (a + 0xfd7046c5) + (a << 3);
        for (typename std::set<T>::const_iterator it = set.begin(); it != set.end(); ++it)
            a += (*it << a) + a * *it + (*it ^ 0xb711a53c);
        return a;
    }
};

template<class T>
struct PointerHash {
    size_t operator()(const T* p) const noexcept {
        return reinterpret_cast<size_t>(p);
    }
};

struct StringHash {
    size_t operator()(const char* s) const noexcept {
        size_t result = 0;
        while (*s)
            result = 5 * result + size_t(*s++);
        return result;
    }
    size_t operator()(const std::string& s) const noexcept {
        return (*this)(s.c_str());
    }
};

struct StringEquality {
    bool operator()(const char* s, const char* t) const {
        return (s == t) || (std::strcmp(s, t) == 0);
    }
    bool operator()(const std::string& s, const std::string& t) const {
        return (s == t);
    }
};

class StringHashSet : public std::unordered_set<std::string, StringHash, StringEquality> {};

template<typename T>
class StringHashMap : public std::unordered_map<std::string, T, StringHash, StringEquality> {};

template<typename T_Key, typename T_Value, typename HashFcn = std::hash<T_Key>, typename EqualKey = std::equal_to<T_Key>>
class HashMap : public std::unordered_map<T_Key, T_Value, HashFcn, EqualKey> {};
}  // namespace Core

#endif  // _CORE_HASH_HH
