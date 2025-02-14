/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef CALLBACK_REGISTRY_HH
#define CALLBACK_REGISTRY_HH

#undef ensure  // macro duplication in pybind11/numpy.h
#include <Core/Singleton.hh>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class CallbackRegistry_ {
public:
    void         registerCallback(const std::string& name, py::function callback);
    bool         hasCallback(const std::string& name) const;
    py::function get_callback(const std::string& name);

private:
    std::unordered_map<std::string, py::function> callbacks_;
};

typedef Core::SingletonHolder<CallbackRegistry_> CallbackRegistry;

#endif
