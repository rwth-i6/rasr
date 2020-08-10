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
#ifndef _CORE_TYPECASTS_HH
#define _CORE_TYPECASTS_HH

#include <type_traits>
#include <string.h>

#include "TList.hh"
#include "Types.hh"

namespace Core {

typedef TList<s8, u8, s16, u16, s32, u32, s64, u64, f32, f64> NumberTypeList;

/*
 * Use xmlMatchTagName() in XmlElement types which contain numeric types, such as XmlVectorElement,
 * if you want to allow casts between different numeric types.
 *
 * Requirements:
 *    - Its a template class with one template parameter which is the type, such as f32.
 *    - It has the typedef Predecessor which is its base class.
 *    - It has a static attribute tagName, which is an object with a function c_str().
 */

template<template<typename> class XmlElemThisType, typename T>
inline typename std::enable_if<!NumberTypeList::contains<T>::value, bool>::type
        xmlMatchTagName(const XmlElemThisType<T>& elem, const char* name) {
    return elem.XmlElemThisType<T>::Predecessor::matches(name);
}

template<template<typename> class XmlElemThisType, typename T>
struct XmlMatchTagNameHandler;

// This is the specialisation to cast between types in NumberTypeList.
template<template<typename> class XmlElemThisType, typename T>
inline typename std::enable_if<NumberTypeList::contains<T>::value, bool>::type
        xmlMatchTagName(const XmlElemThisType<T>& elem, const char* name) {
    if (elem.XmlElemThisType<T>::Predecessor::matches(name))
        return true;
    return NumberTypeList::forEach(XmlMatchTagNameHandler<XmlElemThisType, T>(), elem, name).result;
}

template<template<typename> class XmlElemThisType, typename T>
struct XmlMatchTagNameHandler {
    typedef XmlElemThisType<T>                XmlElemType;
    typedef typename XmlElemType::Predecessor XmlElemBase;

    bool result;
    XmlMatchTagNameHandler()
            : result(false) {}

    // We don't need to check if T == T2.
    template<typename T2>
    typename std::enable_if<std::is_same<T, T2>::value, void>::type
            handle(const XmlElemThisType<T>&, const char*) {}

    template<typename T2>
    typename std::enable_if<!std::is_same<T, T2>::value, void>::type
            handle(const XmlElemType& elem, const char* name) {
        if (result)
            return;

        const auto& tagName = XmlElemThisType<T2>::tagName;
        if (strcmp(name, tagName.c_str()) != 0)
            return;

        // We have a match.
        const_cast<XmlElemType&>(elem).parser()->warning("casting %s to %s", name, elem.name());
        result = true;
    }
};

}  // namespace Core

#endif  // TYPECASTS_HH
