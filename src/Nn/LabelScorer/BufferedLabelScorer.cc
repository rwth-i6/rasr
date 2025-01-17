/** Copyright 2024 RWTH Aachen University. All rights reserved.
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

#include "BufferedLabelScorer.hh"

namespace Nn {

BufferedLabelScorer::BufferedLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          inputBuffer_(),
          featureSize_(Core::Type<size_t>::max),
          expectMoreFeatures_(true) {
}

void BufferedLabelScorer::reset() {
    inputBuffer_.clear();
    featureSize_        = Core::Type<size_t>::max;
    expectMoreFeatures_ = true;
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    expectMoreFeatures_ = false;
}

void BufferedLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = featureSize;
    }
    else if (featureSize_ != featureSize) {
        error() << "Label scorer received incompatible feature size " << featureSize << "; was set to " << featureSize_ << " before.";
    }

    inputBuffer_.push_back(input);
}

}  // namespace Nn
