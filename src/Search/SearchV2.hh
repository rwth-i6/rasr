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
#ifndef SEARCH_V2_HH
#define SEARCH_V2_HH

#include <Am/AcousticModel.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/Types.hh>
#include <Search/LatticeAdaptor.hh>
#include <Search/Types.hh>
#include <Speech/Feature.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Types.hh>

#include "Traceback.hh"

namespace Search {

/*
 * Abstract base class for search algorithms that can work in an online or offline manner.
 * In contrast to the previous `SearchAlgorithm` this receives features instead of `Mm::FeatureScorers`
 * and the algorithm is supposed to perform scoring internally (usually with a `Nn::LabelScorer`).
 *
 * The "workflow" usually happens as follows:
 *  1. Check which `Speech::ModelCombination` (and possibly `Am::AcousticModel`) are needed by the search algorithm via
 *     `requiredModelCombination` (and `requiredAcousticModel`).
 *  2. Create appropriate `Speech::ModelCombination` and provide it to the search algorithm via `setModelCombination`.
 *  3. Signal segment start via `enterSegment`.
 *  4. Pass audio features via `putFeature` or `putFeatures`.
 *  (5. Call `decodeStep` or `decodeManySteps` to run the next search step(s) given the currently available features.)
 *  (6. Optionally retrieve intermediate results via `getCurrentBestTraceback` or `getCurrentBestWordLattice`.)
 *  7. Call `finishSegment` to signal that all features have been passed and finalize the search with all the segment features.
 *  8. Retrieve the final result via `getCurrentBestTraceback` or `getCurrentBestWordLattice`.
 *  9. Call `reset` to clean up any buffered features, hypotheses, flags etc. from the previous segment and prepare the algorithm for the next one.
 *  (10. Optionally also reset search statistics via `resetStatistics`).
 *  11. Continue again at step 3.
 */
class SearchAlgorithmV2 : public virtual Core::Component {
public:
    SearchAlgorithmV2(Core::Configuration const&)
            : Core::Component(config) {}
    virtual ~SearchAlgorithmV2() = default;

    // Check which parts of the `Speech::ModelCombination` are required for the search and need to be set via `setModelCombination`
    virtual Speech::ModelCombination::Mode requiredModelCombination() const = 0;

    // Check which parts of the `Am::AcousticModel` are required for the search (only in the case that an `AcousticModel` is needed in the ModelCombination).
    virtual Am::AcousticModel::Mode requiredAcousticModel() const {
        return Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTying | Am::AcousticModel::noStateTransition;
    }

    // Pass a `Speech::ModelCombination` that matches the requirements set by `requiredModelCombination` (and `requiredAcousticModel`) to the search.
    virtual bool setModelCombination(Speech::ModelCombination const& modelCombination) = 0;

    // Cleans up buffers, hypotheses, flags etc. from the previous segment recognition.
    virtual void reset() = 0;

    // Return the first trace of all hypotheses. Needed for computing partial trace.
    virtual Core::Ref<LatticeTrace> getRootTrace() const = 0;

    // Signal the beginning of a new audio segment.
    virtual void enterSegment(Bliss::SpeechSegment const* = nullptr) = 0;

    // Signal that all features of the current segment have been passed.
    virtual void finishSegment() = 0;

    // Pass a single feature vector.
    virtual void putFeature(Nn::DataView const& feature) = 0;

    // Pass feature vectors for multiple time steps.
    virtual void putFeatures(Nn::DataView const& features, size_t nTimesteps) = 0;

    // Return the current best traceback. May contain unstable results.
    virtual Core::Ref<const Traceback> getCurrentBestTraceback() const = 0;

    // Similar to `getCurrentBestTraceback` but return the lattice instead of just single-best traceback.
    virtual Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const = 0;

    // Return common prefix of all active traces.
    virtual Core::Ref<LatticeTrace> getCommonPrefix() const = 0;

    // Try to decode one more step. Return bool indicates whether a step could be made.
    virtual bool decodeStep() = 0;

    // Decode as much as possible given the currently available features. Return number of successfull steps.
    virtual unsigned decodeManySteps() {
        unsigned count = 0u;
        while (decodeStep()) {
            count++;
        }

        return count;
    }
};

}  // namespace Search

#endif  // SEARCH_V2_HH
