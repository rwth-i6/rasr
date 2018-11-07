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
#ifndef _TENSORFLOW_MODULE_HH
#define _TENSORFLOW_MODULE_HH

#include <memory>

#include <Core/Singleton.hh>

#include "GraphLoader.hh"

namespace Tensorflow {

class Module_ {
public:
    static const Core::Choice          choiceGraphLoader;
    static const Core::ParameterChoice paramGraphLoader;

    Module_();
    ~Module_() = default;

    std::unique_ptr<GraphLoader> createGraphLoader(Core::Configuration const& config);
};

typedef Core::SingletonHolder<Module_> Module;

}  // namespace Tensorflow

#endif  // _TENSORFLOW_MODULE_HH
