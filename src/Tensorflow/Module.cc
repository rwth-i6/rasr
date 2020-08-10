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
#include "Module.hh"

#include <Flow/Registry.hh>

#include "MetaGraphLoader.hh"
#include "TensorflowForwardNode.hh"
#include "VanillaGraphLoader.hh"

namespace {
enum GraphLoaderChoice {
    graphLoaderNotGiven,
    graphLoaderVanilla,
    graphLoaderMeta
};
}

namespace Tensorflow {

const Core::Choice Module_::choiceGraphLoader(
        "vanilla", GraphLoaderChoice::graphLoaderVanilla,
        "meta", GraphLoaderChoice::graphLoaderMeta,
        Core::Choice::endMark());
const Core::ParameterChoice Module_::paramGraphLoader("type", &choiceGraphLoader, "graph-loader to use", GraphLoaderChoice::graphLoaderNotGiven);

Module_::Module_() {
    Flow::Registry::Instance& registry = Flow::Registry::instance();

    registry.registerFilter<TensorflowForwardNode>();
    registry.registerFilter<TensorflowOverlappingForwardNode>();
}

std::unique_ptr<GraphLoader> Module_::createGraphLoader(Core::Configuration const& config) {
    GraphLoaderChoice choice = static_cast<GraphLoaderChoice>(paramGraphLoader(config));
    switch (choice) {
        case graphLoaderVanilla:
            return std::unique_ptr<GraphLoader>(new VanillaGraphLoader(config));
        case graphLoaderMeta:
            return std::unique_ptr<GraphLoader>(new MetaGraphLoader(config));
        case graphLoaderNotGiven:
        default:
            return std::unique_ptr<GraphLoader>();
    }
}

}  // namespace Tensorflow
