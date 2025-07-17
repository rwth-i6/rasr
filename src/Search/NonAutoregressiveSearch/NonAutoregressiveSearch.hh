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

#ifndef NON_AUTOREGRESSIVE_SEARCH_HH
#define NON_AUTOREGRESSIVE_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

namespace Search {

class NonAutoregressiveSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt  paramBlankLabelIndex;
    static const Core::ParameterBool paramCollapseRepeatedLabels;
    static const Core::ParameterBool paramCacheCleanupInterval;
    static const Core::ParameterBool paramLogStepwiseStatistics;

    NonAutoregressiveSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                            finishSegment() override;
    void                            putFeature(Nn::DataView const& feature) override;
    void                            putFeatures(Nn::DataView const& features, size_t nTimesteps) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    bool                            decodeStep() override;

protected:
    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        Nn::ScoringContextRef scoringContext;  // Context to compute scores based on this hypothesis
        Score                 score;           // Full score of hypothesis Core::Ref<LatticeTrace> trace;           // Associated trace for traceback or lattice building off of hypothesis

        LabelHypothesis();

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging.
         */
        std::string toString() const;
    };

private:
    bool           useBlank_;
    Nn::LabelIndex blankLabelIndex_;
    bool           collapseRepeatedLabels_;
    bool           logStepwiseStatistics_;
    size_t         cacheCleanupInterval_;

    Core::Channel debugChannel_;

    Core::Ref<Nn::LabelScorer> labelScorer_;
    Bliss::LexiconRef          lexicon_;

    void resetStatistics();
    void logStatistics() const;
};

}  // namespace Search

#endif  // NON_AUTOREGRESSIVE_SEARCH_HH
