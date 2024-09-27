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
#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Nn/LabelScorer.hh>
#include <Nn/Module.hh>
#include <Nn/Types.hh>
#include <Search/SearchV2.hh>
#include <Speech/ModelCombination.hh>
#include "Search/Types.hh"

namespace Search {

// Bare-bones search algorithm without lexicon, LM, transition model, beam or pruning.
// Given a lexicon only containing labels (without lemmas), pick the label index with
// maximum probability at each position.
class GreedyTimeSyncSearch : public SearchAlgorithmV2 {
    struct HypothesisExtension {
        const Bliss::Lemma*             lemma;
        Nn::LabelIndex                  label;
        Score                           score;
        Search::TimeframeIndex          timestep;
        Nn::LabelScorer::TransitionType transitionType;

        HypothesisExtension()
                : lemma(), label(), score(Core::Type<Score>::max), timestep(), transitionType() {}

        HypothesisExtension(const Bliss::Lemma* lemma, Nn::LabelIndex label, Score score, Search::TimeframeIndex timestep, Nn::LabelScorer::TransitionType transitionType)
                : lemma(lemma), label(label), score(score), timestep(timestep), transitionType(transitionType) {}
    };

    struct LabelHypothesis {
        Core::Ref<Nn::LabelHistory> history;
        Nn::LabelIndex              currentLabel;
        Score                       score;
        Traceback                   traceback;

        LabelHypothesis()
                : history(), currentLabel(Core::Type<Nn::LabelIndex>::max), score(0.0), traceback() {}

        void reset();
        void extend(const HypothesisExtension& extension, Core::Ref<Nn::LabelScorer> labelScorer);
    };

public:
    static const Core::ParameterBool paramUseBlank;
    static const Core::ParameterBool paramAllowLabelLoop;
    static const Core::ParameterInt  paramBlankLabelIndex;

    GreedyTimeSyncSearch(const Core::Configuration&);

    // Inherited methods

    Speech::ModelCombination::Mode  modelCombinationNeeded() const override;
    bool                            setModelCombination(const Speech::ModelCombination& modelCombination) override;
    void                            reset() override;
    void                            enterSegment() override;
    void                            enterSegment(Bliss::SpeechSegment const*) override;
    void                            finishSegment() override;
    void                            addFeature(Nn::FeatureVectorRef) override;
    void                            addFeature(Core::Ref<const Speech::Feature>) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    void                            resetStatistics() override;
    void                            logStatistics() const override;
    bool                            decodeStep() override;

private:
    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    bool useBlank_;
    bool allowLabelLoop_;

    Nn::LabelIndex blankLabelIndex_;

    Core::Ref<Nn::LabelScorer> labelScorer_;
    Nn::LabelIndex             numClasses_;
    Bliss::LexiconRef          lexicon_;
    LabelHypothesis            hyp_;
};

}  // namespace Search
#endif  // GREEDY_SEARCH_HH
