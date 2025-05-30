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

#include "Align.hh"

#include <Lm/CheatingSegmentLm.hh>
#include <Lm/CombineLm.hh>
#include <Search/Module.hh>
#include <Search/TreeTimesyncBeamSearch/TreeTimesyncBeamSearch.hh>

#include <Bliss/CorpusDescription.hh>
#include <Lm/Module.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Speech/ModelCombination.hh>

namespace py = pybind11;

Aligner::Aligner(const Core::Configuration& c)
        : Core::Component(c),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV2(select("search-algorithm"))) {
    if (not searchAlgorithm_->requiredModelCombination() & Speech::ModelCombination::useLanguageModel) {
        error() << "Search algorithm must support a word-level language model";
    }

    Speech::ModelCombination modelCombination(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel());

    // Replace language model in modelCombination with a CombineLm containing a primary cheating Lm together with the usual Lm

    Core::Configuration cheatingLmConfig;
    cheatingLmConfig.set("infinity-score", "1e9");
    cheatingLmConfig.set("skip-threshold", "$[1e9 - 1]");
    auto cheatingLm = Core::ref(new Lm::CheatingSegmentLm(cheatingLmConfig, modelCombination.lexicon()));
    cheatingLm->load();
    Core::Ref<Lm::ScaledLanguageModel> scaledCheatingLm = Core::ref(new Lm::LanguageModelScaling(config, cheatingLm));

    Core::Configuration combineLmConfig;
    combineLmConfig.set("lookahead-lm", "1");
    combineLmConfig.set("recombination-lm", "2");
    auto                               combineLm       = Core::ref(new Lm::CombineLanguageModel(combineLmConfig, modelCombination.lexicon(), {scaledCheatingLm, modelCombination.languageModel()}));
    Core::Ref<Lm::ScaledLanguageModel> scaledCombineLm = Core::ref(new Lm::LanguageModelScaling(config, combineLm));

    modelCombination.setLanguageModel(scaledCombineLm);
    searchAlgorithm_->setModelCombination(modelCombination);
}

void Aligner::putFeatures(py::array_t<f32> const& features) {
    size_t T = 0ul;
    size_t F = 0ul;
    if (features.ndim() == 3) {
        if (features.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << features.shape(0) << "; should be 1";
        }
        T = features.shape(1);
        F = features.shape(2);
    }
    else if (features.ndim() == 2) {
        T = features.shape(0);
        F = features.shape(1);
    }
    else {
        error() << "Received feature tensor of invalid dim " << features.ndim() << "; should be 2 or 3";
    }

    searchAlgorithm_->putFeatures({features, T * F}, T);
}

Traceback Aligner::getBestTraceback() {
    searchAlgorithm_->decodeManySteps();

    auto                       traceback = searchAlgorithm_->getCurrentBestTraceback();
    std::vector<TracebackItem> result;
    result.reserve(traceback->size());

    u32 prevTime = 0;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        if (not it->pronunciation or not it->pronunciation->lemma()) {
            continue;
        }
        result.push_back({
                it->pronunciation->lemma()->symbol(),
                it->score.acoustic,
                it->score.lm,
                prevTime,
                it->time,
        });
        prevTime = it->time;
    }
    return result;
}

Traceback Aligner::alignSegment(py::array_t<f32> const& features, std::string const& orth) {
    searchAlgorithm_->reset();

    Bliss::Corpus        corpus;
    Bliss::Recording     recording(&corpus);
    Bliss::SpeechSegment segment(&recording);
    segment.setOrth(orth);

    searchAlgorithm_->enterSegment(&segment);
    putFeatures(features);
    searchAlgorithm_->finishSegment();
    return getBestTraceback();
}
