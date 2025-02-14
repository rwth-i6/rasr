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
#include "CallbackRegistry.hh"

void CallbackRegistry_::registerCallback(const std::string& name, py::function callback) {
    callbacks_[name] = callback;
}

bool CallbackRegistry_::hasCallback(const std::string& name) const {
    return callbacks_.find(name) != callbacks_.end();
}

py::function CallbackRegistry_::get_callback(const std::string& name) {
    if (not hasCallback(name)) {
        throw std::runtime_error("Python callback '" + name + "' not registered.");
    }
    return callbacks_[name];
}
