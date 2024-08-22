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

#ifndef GREEDY_SEARCH_HH
#define GREEDY_SEARCH_HH

#include <Bliss/CorpusDescription.hh>
#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Nn/LabelScorer.hh>
#include <Nn/Module.hh>
#include <Search/SearchV2.hh>

namespace Search {

// Bare-bones search algorithm without lexicon, LM, transition model, beam or pruning.
// Given a vocab file that maps strings to label indices, pick the label index with
// maximum probability at each position.
class GreedyTimeSyncSearch : public SearchAlgorithmV2 {
    struct LabelHypothesis {
        Core::Ref<Nn::LabelHistory> history;
        std::vector<Nn::LabelIndex> labelSeq;
        Score                       score;
        Traceback                   traceback;

        LabelHypothesis()
                : history(nullptr), labelSeq(), score(0.0), traceback() {}
    };

public:
    static const Core::ParameterBool   paramUseBlank;
    static const Core::ParameterBool   paramAllowLabelLoop;
    static const Core::ParameterInt    paramBlankLabelIndex;
    static const Core::ParameterString paramVocabFile;

    GreedyTimeSyncSearch(const Core::Configuration&);

    // Inherited methods

    void                            reset() override;
    void                            enterSegment() override;
    void                            enterSegment(Bliss::SpeechSegment const*) override;
    void                            finishSegment() override;
    void                            finalize() override;
    void                            addFeature(Core::Ref<const Speech::Feature>) override;
    Core::Ref<const Traceback>      stablePartialTraceback() override;
    Core::Ref<const Traceback>      recentStablePartialTraceback() override;
    Core::Ref<const Traceback>      unstablePartialTraceback() const override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getPartialWordLattice() override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    void                            resetStatistics() override;
    void                            logStatistics() const override;

private:
    // Try to decode one more timeframe. Return bool indicates whether next timeframe was
    // able to be decoded or not.
    bool decodeStep();

    // Decode as much as possible given the currently available features
    void decodeMore();

    void parseVocabFile(const std::string& filename);

    bool useBlank_;
    bool allowLabelLoop_;

    Nn::LabelIndex blankLabelIndex_;

    Core::Ref<Nn::LabelScorer>                      labelScorer_;
    Nn::LabelIndex                                  numClasses_;
    std::unordered_map<std::string, Nn::LabelIndex> vocabMap_;
    LabelHypothesis                                 hyp_;
    size_t                                          currentStep_;

    Core::Ref<Traceback> previousPassedTraceback_;
};

}  // namespace Search
#endif  // GREEDY_SEARCH_HH
