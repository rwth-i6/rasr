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
#include "Module.hh"
#include <Core/Configuration.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#include "GenericPythonLabelScorer.hh"
#include "LimitedCtxPythonLabelScorer.hh"

namespace Python {

Module_::Module_() {
#ifdef MODULE_NN
    // Feed the feature at the current step together with a (fixed-size) history into a python callback
    Nn::Module::instance().labelScorerFactory().registerLabelScorer(
            "limited-ctx-python",
            [](Core::Configuration const& config) {
                return Core::ref(new LimitedCtxPythonLabelScorer(config));
            });
    Nn::Module::instance().labelScorerFactory().registerLabelScorer(
            "generic-python",
            [](Core::Configuration const& config) {
                return Core::ref(new GenericPythonLabelScorer(config));
            });
#endif
};

}  // namespace Python
