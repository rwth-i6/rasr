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
#ifndef _SPEECH_MODEL_COMBINATION_HH
#define _SPEECH_MODEL_COMBINATION_HH

#include <Am/AcousticModel.hh>
#include <Am/AcousticModelAdaptor.hh>
#include <Bliss/Lexicon.hh>
#include <Core/ReferenceCounting.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Mc/Component.hh>
#include <Nn/LabelScorer/LabelScorer.hh>

namespace Speech {

/** Combination of a lexicon, an acoustic model, a label scorer and a language model.
 *  It supports the creation and initialization of these four mutually dependent objects.
 *
 *  Usage:
 *    - Create a ModelCombination object locally to create the four parts:
 *       lexicon, acoustic model, label scorer and/or language model.
 *    - The ModelCombination can be directly created by passing references to the lexicon,
 *      acoustic model and language model.
 *    - Alternatively, it is possible to set a Mode indicating which components are required
 *      by setting useLexicon, useAcousticModel, useLanguageModel and/or useLabelScorer.
 *      In this case, the ModelCombination will create the relevant parts from the config.
 *      (A Mode for the acoustic model and a lexicon reference can optionally be passed as well.)
 *    - Store the references to those parts which you will use later.
 *    - When the local ModelCombination object is destructed, the unreferenced parts get freed as well.
 */
class ModelCombination : public Mc::Component, public Core::ReferenceCounted {
public:
    typedef u32       Mode;
    static const Mode complete;  // Includes lexicon, AM and LM but NOT label scorer; named 'complete' for legacy reasons.
    static const Mode useLexicon;
    static const Mode useAcousticModel;
    static const Mode useLanguageModel;
    static const Mode useLabelScorer;

    static const Core::ParameterFloat paramPronunciationScale;
    static const Core::ParameterInt   paramNumLabelScorers;

protected:
    Bliss::LexiconRef                       lexicon_;
    Mm::Score                               pronunciationScale_;
    Core::Ref<Am::AcousticModel>            acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>      languageModel_;
    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers_;

private:
    void setPronunciationScale(Mm::Score scale) {
        pronunciationScale_ = scale;
    }

protected:
    virtual void distributeScaleUpdate(const Mc::ScaleUpdate& scaleUpdate);

public:
    ModelCombination(const Core::Configuration&,
                     Mode                    = complete,
                     Am::AcousticModel::Mode = Am::AcousticModel::complete,
                     Bliss::LexiconRef       = Bliss::LexiconRef());

    ModelCombination(const Core::Configuration&,
                     Bliss::LexiconRef,
                     Core::Ref<Am::AcousticModel>,
                     Core::Ref<Lm::ScaledLanguageModel>);

    virtual ~ModelCombination();

    Mm::Score pronunciationScale() const {
        return pronunciationScale_ * scale();
    }

    void setLexicon(Bliss::LexiconRef);

    Bliss::LexiconRef lexicon() const {
        return lexicon_;
    }

    void setAcousticModel(Core::Ref<Am::AcousticModel>);

    Core::Ref<Am::AcousticModel> acousticModel() const {
        return acousticModel_;
    }

    void setLanguageModel(Core::Ref<Lm::ScaledLanguageModel>);

    Core::Ref<Lm::ScaledLanguageModel> languageModel() const {
        return languageModel_;
    }

    void setLabelScorer(Core::Ref<Nn::LabelScorer> ls, size_t index = 0ul);

    Core::Ref<Nn::LabelScorer> labelScorer(size_t index = 0ul) const {
        verify(index < labelScorers_.size());
        return labelScorers_[index];
    }

    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers() const {
        return labelScorers_;
    }

    void getDependencies(Core::DependencySet&) const;
};

typedef Core::Ref<ModelCombination> ModelCombinationRef;

}  // namespace Speech

namespace Core {

template<>
class NameHelper<Speech::ModelCombinationRef> : public std::string {
public:
    NameHelper()
            : std::string("flow-model-combination-ref") {}
};

}  // namespace Core

#endif  // _SPEECH_MODEL_COMBINATION_HH
