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

#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Core {

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

struct StringEquality : std::binary_function<const char*, const char*, bool> {
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
