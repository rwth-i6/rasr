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
#ifndef CORE_EXTENSIONS_HH
#define CORE_EXTENSIONS_HH

#include <functional>
// Check for libstdc++ (GNU GCC STL).
#ifdef __GLIBCXX__
#include <ext/functional>
#endif

#include "Hash.hh"

namespace Core {

// Check for libstdc++ (GNU GCC STL).
// This is almost the only lib which implements these SGI extensions.
#ifdef __GLIBCXX__
using __gnu_cxx::binary_compose;
using __gnu_cxx::identity;
using __gnu_cxx::select1st;
using __gnu_cxx::select2nd;

#else

template<typename T>
struct identity : public std::unary_function<T, T> {
    const T& operator()(const T& v) const {
        return v;
    }
    T& operator()(T& v) const {
        return v;
    }
};

template<typename BinaryFun, typename UnaryFun1, typename UnaryFun2 = UnaryFun1>
struct binary_compose
        : public std::unary_function<typename UnaryFun1::argument_type, typename BinaryFun::result_type> {
    typename BinaryFun::result_type operator()(const typename UnaryFun1::argument_type& x) const {
        return f(g(x), h(x));
    }

    BinaryFun f;
    UnaryFun1 g;
    UnaryFun2 h;
};

template<typename pair_type>
struct select1st
        : public std::unary_function<const pair_type&, const typename pair_type::first_type&> {
    const typename pair_type::first_type& operator()(const pair_type& v) const {
        return v.first;
    }
};

template<typename pair_type>
struct select2nd
        : public std::unary_function<const pair_type&, const typename pair_type::second_type&> {
    const typename pair_type::second_type& operator()(const pair_type& v) const {
        return v.second;
    }
};

#endif

}  // namespace Core

#endif  // CORE_EXTENSIONS_HH
