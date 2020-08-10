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
#ifndef _CORE_TLIST_HH
#define _CORE_TLIST_HH

#include <type_traits>
#include <utility>
#include <stddef.h>

namespace Core {

/*
 * This introduces a meta template list, implemented via variadic templates.
 * Some reference:
 *   http://en.cppreference.com/w/cpp/language/parameter_pack
 * C++ standard:
 *   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf
 * See Boost.MPL for much more of this magic.
 *
 * Usage example:
 *
 * typedef TList<s32, f32> MyTypeList;
 *
 * struct MyHandler {
 *    template<typename T>
 *    void handle(int extra) {
 *        cout << T(extra) << " ";
 *    }
 * };
 *
 * MyTypeList(MyHandler(), 42); // prints "42 42.0 "
 *
 * cout << MyTypeList::contains<s32>::value; // prints "1"
 * cout << MyTypeList::contains<u8>::value; // prints "0"
 *
 */

template<typename... Elems>
struct TList {
    // This matches no template args.
    static const size_t size = 0;

    template<typename Handler, typename... HandlerArgs>
    static Handler&& forEach(Handler&& handler, HandlerArgs...) {
        return std::forward<Handler>(handler);
    }

    template<typename OtherElem>
    struct contains : std::false_type {};
};

template<typename FirstElem, typename... OtherElems>
struct TList<FirstElem, OtherElems...> {
    // This mathes one or more template args.
    typedef FirstElem            FirstElemT;
    typedef TList<OtherElems...> TSubListT;
    static const size_t          size = 1 + sizeof...(OtherElems);

    template<typename Handler, typename... HandlerArgs>
    static Handler&& forEach(Handler&& handler, HandlerArgs... args) {
        handler.template handle<FirstElem>(args...);
        return std::forward<Handler>(
                TSubListT::forEach(handler, args...));
    }

    template<typename OtherElem>
    struct contains : std::integral_constant<bool, std::is_same<FirstElem, OtherElem>::value ||
                                                           TSubListT::template contains<OtherElem>::value> {};
};

}  // namespace Core

#endif  // TLIST_HH
