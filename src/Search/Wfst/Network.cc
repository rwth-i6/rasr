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
#include <Core/Application.hh>
#include <OpenFst/Scale.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/Network.hh>
#include <fst/arcsort.h>

namespace Search {
namespace Wfst {

const Core::ParameterString StaticNetwork::paramNetworkFile_(
        "file", "search network to load", "");
const Core::ParameterFloat StaticNetwork::paramScale_(
        "scale", "weight scaling factor", 1.0);

StaticNetwork::StaticNetwork(const Core::Configuration& c)
        : Precursor(c) {}

bool StaticNetwork::init() {
    logMemoryUsage();
    Core::Component::log("reading network: %s", paramNetworkFile_(config).c_str());
    f_ = OpenFst::VectorFst::Read(paramNetworkFile_(config));
    logMemoryUsage();
    verify(f_);
    if (!f_->Properties(FstLib::kILabelSorted, false)) {
        warning("input automaton is not sorted by input.");
        log("sorting automaton");
        FstLib::ArcSort(f_, FstLib::StdILabelCompare());
    }
    f32 scale = paramScale_(config);
    if (scale != 1.0) {
        log("scaling weights: %f", scale);
        OpenFst::scaleWeights(f_, scale);
    }
    return true;
}

}  // namespace Wfst
}  // namespace Search
