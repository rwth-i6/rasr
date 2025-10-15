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
#include "Search.hh"

using namespace Search;

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c) {}

Speech::ModelCombination::Mode SearchAlgorithm::modelCombinationNeeded() const {
    return Speech::ModelCombination::complete;
}

void SearchAlgorithm::getCurrentBestSentencePartial(Traceback& result) const {
}

void SearchAlgorithm::getPartialSentence(Traceback&) {
}

SearchAlgorithm::PruningRef SearchAlgorithm::describePruning() {
    return SearchAlgorithm::PruningRef();
}

bool SearchAlgorithm::relaxPruning(f32 factor, f32 offset) {
    return false;
}

void SearchAlgorithm::resetPruning(SearchAlgorithm::PruningRef pruning) {
}

Core::Ref<const LatticeAdaptor> SearchAlgorithm::getPartialWordLattice() {
    return Core::Ref<const LatticeAdaptor>();
}
