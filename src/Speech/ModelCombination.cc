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
#include "ModelCombination.hh"
#include <Am/Module.hh>
#include <Lm/Module.hh>
#include <Nn/Module.hh>
#include <string>

using namespace Speech;

const ModelCombination::Mode ModelCombination::complete         = 0x3;
const ModelCombination::Mode ModelCombination::useLexicon       = 0x0;
const ModelCombination::Mode ModelCombination::useAcousticModel = 0x1;
const ModelCombination::Mode ModelCombination::useLanguageModel = 0x2;
const ModelCombination::Mode ModelCombination::useLabelScorers  = 0x4;

//======================================================================================

const Core::ParameterFloat ModelCombination::paramPronunciationScale(
        "pronunciation-scale", "scaling exponent for lemma pronunciation probabilities", 0.0);

const Core::ParameterInt ModelCombination::paramNumLabelScorers(
        "num-label-scorers", "Number of label scorers in model combination", 1, 1);

ModelCombination::ModelCombination(const Core::Configuration& c)
        : Core::Component(c),
          Mc::Component(c),
          labelScorers_(paramNumLabelScorers(c)) {
    setPronunciationScale(paramPronunciationScale(c));
}

ModelCombination::ModelCombination(const Core::Configuration& c,
                                   Mode                       mode,
                                   Am::AcousticModel::Mode    acousticModelMode,
                                   Bliss::LexiconRef          lexicon)
        : Core::Component(c),
          Mc::Component(c),
          labelScorers_(paramNumLabelScorers(c)) {
    setPronunciationScale(paramPronunciationScale(c));
    build(mode, acousticModelMode, lexicon);
}

ModelCombination::ModelCombination(const Core::Configuration&         c,
                                   Bliss::LexiconRef                  lexicon,
                                   Core::Ref<Am::AcousticModel>       acousticModel,
                                   Core::Ref<Lm::ScaledLanguageModel> languageModel)
        : Core::Component(c), Mc::Component(c), pronunciationScale_(0), labelScorers_(paramNumLabelScorers(c)) {
    setPronunciationScale(paramPronunciationScale(c));
    setLexicon(lexicon);
    setAcousticModel(acousticModel);
    setLanguageModel(languageModel);
}

ModelCombination::~ModelCombination() {}

void ModelCombination::build(Mode                    mode,
                             Am::AcousticModel::Mode acousticModelMode,
                             Bliss::LexiconRef       lexicon) {
    if (lexicon) {
        setLexicon(lexicon);
        log() << "Set lexicon in ModelCombination";
    }
    else {
        log() << "Create lexicon in ModelCombination";
        setLexicon(Bliss::Lexicon::create(select("lexicon")));
    }
    if (!lexicon_) {
        criticalError("failed to initialize the lexicon");
    }

    if (mode & useAcousticModel) {
        log() << "Create acoustic model in ModelCombination";
        setAcousticModel(Am::Module::instance().createAcousticModel(
                select("acoustic-model"), lexicon_, acousticModelMode));
        if (!acousticModel_) {
            criticalError("failed to initialize the acoustic model");
        }
    }

    if (mode & useLanguageModel) {
        log() << "Create language model in ModelCombination";
        setLanguageModel(Lm::Module::instance().createScaledLanguageModel(select("lm"), lexicon_));
        if (!languageModel_) {
            criticalError("failed to initialize language model");
        }
    }

    if (mode & useLabelScorers) {
        log() << "Create label scorers in ModelCombination";
        if (labelScorers_.size() == 1) {
            setLabelScorer(Nn::Module::instance().createLabelScorer(select("label-scorer")));
        }
        else {
            for (size_t idx = 0ul; idx < labelScorers_.size(); ++idx) {
                setLabelScorer(Nn::Module::instance().createLabelScorer(select("label-scorer-" + std::to_string(idx + 1))), idx);
            }
        }
    }
}

void ModelCombination::setLexicon(Bliss::LexiconRef lexicon) {
    lexicon_ = lexicon;
}

void ModelCombination::setAcousticModel(Core::Ref<Am::AcousticModel> acousticModel) {
    acousticModel_ = acousticModel;
    if (acousticModel_)
        acousticModel_->setParentScale(scale());
}

void ModelCombination::setLanguageModel(Core::Ref<Lm::ScaledLanguageModel> languageModel) {
    languageModel_ = languageModel;
    if (languageModel_)
        languageModel_->setParentScale(scale());
}

void ModelCombination::distributeScaleUpdate(const Mc::ScaleUpdate& scaleUpdate) {
    if (lexicon_) {
        Mm::Score scale;
        if (scaleUpdate.findScale(config.prepareResourceName(fullName(), paramPronunciationScale.name()), scale)) {
            log("Pronunciation scale set to %f.", scale);
            setPronunciationScale(scale);
        }
    }
    if (acousticModel_)
        acousticModel_->updateScales(scaleUpdate);
    if (languageModel_)
        languageModel_->updateScales(scaleUpdate);
}

void ModelCombination::getDependencies(Core::DependencySet& dependencies) const {
    Core::DependencySet d;
    if (lexicon_) {
        d.add("lexicon", lexicon_->getDependency());
        d.add("pronunciation scale", pronunciationScale_);
    }
    if (acousticModel_)
        acousticModel_->getDependencies(d);
    if (languageModel_)
        languageModel_->getDependencies(d);

    dependencies.add(name(), d);
    Mc::Component::getDependencies(dependencies);
}
