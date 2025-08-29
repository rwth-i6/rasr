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
#include <Nn/LabelScorer/DataView.hh>
#include <Speech/ModelCombination.hh>
#include <stdexcept>
#include <utility>

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

std::vector<s32> ctcAlignment(py::array_t<f32> const& scores, py::array_t<s32> const& targets, s32 blankId) {
    size_t T = 0ul;
    size_t V = 0ul;
    size_t S = 0ul;
    if (scores.ndim() == 3) {
        if (scores.shape(0) != 1) {
            throw std::invalid_argument("Received scores tensor with non-trivial batch dimension");
        }
        T = scores.shape(1);
        V = scores.shape(2);
    }
    else if (scores.ndim() == 2) {
        T = scores.shape(0);
        V = scores.shape(1);
    }
    else {
        throw std::invalid_argument("Received scores tensor with invalid number of dimensions");
    }

    if (targets.ndim() == 2) {
        if (targets.shape(0) != 1) {
            throw std::invalid_argument("Received target tensor with non-trivial batch dimension");
        }
        S = targets.shape(1);
    }
    else if (targets.ndim() == 1) {
        S = targets.shape(0);
    }
    else {
        throw std::invalid_argument("Received target tensor with invalid number of dimensions");
    }

    Nn::DataView scoresView(scores, T * V);
    Nn::DataView targetsView(targets, S);

    size_t const     L = 2 * S + 1;
    std::vector<s32> fsa(L, blankId);
    for (size_t s = 0ul; s < S; ++s) {
        fsa[2 * s + 1] = targetsView[s];
    }
    f32 const           inf = std::numeric_limits<f32>::infinity();
    std::vector<f32>    alphaPrev(L, inf);
    std::vector<f32>    alphaCur(L, inf);
    std::vector<size_t> backPtr(T * L, 0);

    alphaPrev[0] = 0.0f;

    for (size_t t = 0ul; t < T; ++t) {
        for (size_t s = 0; s < L; ++s) {
            f32 best = alphaPrev[s];  // loop
            u32 prev = s;

            if (s > 0) {
                f32 v = alphaPrev[s - 1];  // forward
                if (v < best) {
                    best = v;
                    prev = s - 1;
                }
            }

            if (s > 1 and fsa[s] != fsa[s - 2]) {
                f32 v = alphaPrev[s - 2];  // skip
                if (v < best) {
                    best = v;
                    prev = s - 2;
                }
            }
            alphaCur[s]        = best + scoresView[t * V + fsa[s]];
            backPtr[t * L + s] = prev;
        }
        std::swap(alphaPrev, alphaCur);
    }

    std::vector<s32> result(T);
    size_t           s = alphaPrev[L - 2] < alphaPrev[L - 1] ? L - 2 : L - 1;
    result[T - 1]      = fsa[s];

    for (size_t t = T - 1; t > 0; --t) {
        s             = backPtr[t * L + s];
        result[t - 1] = fsa[s];
    }

    return result;
}

std::vector<s32> rnaAlignment(py::array_t<f32> const& scores, py::array_t<s32> const& targets, s32 blankId) {
    size_t T = 0ul;
    size_t V = 0ul;
    size_t S = 0ul;
    if (scores.ndim() == 3) {
        if (scores.shape(0) != 1) {
            throw std::invalid_argument("Received scores tensor with non-trivial batch dimension");
        }
        T = scores.shape(1);
        V = scores.shape(2);
    }
    else if (scores.ndim() == 2) {
        T = scores.shape(0);
        V = scores.shape(1);
    }
    else {
        throw std::invalid_argument("Received scores tensor with invalid number of dimensions");
    }

    if (targets.ndim() == 2) {
        if (targets.shape(0) != 1) {
            throw std::invalid_argument("Received target tensor with non-trivial batch dimension");
        }
        S = targets.shape(1);
    }
    else if (targets.ndim() == 1) {
        S = targets.shape(0);
    }
    else {
        throw std::invalid_argument("Received target tensor with invalid number of dimensions");
    }

    Nn::DataView scoresView(scores, T * V);
    Nn::DataView targetsView(targets, S);

    std::vector<f32>    alphaPrev(S + 1, std::numeric_limits<f32>::infinity());
    std::vector<f32>    alphaCur(S + 1, std::numeric_limits<f32>::infinity());
    std::vector<size_t> backPtr(T * (S + 1), 0);

    alphaPrev[0] = 0.0f;

    for (size_t t = 0ul; t < T; ++t) {
        for (size_t s = 0; s < S + 1; ++s) {
            f32 best     = alphaPrev[s];  // blank
            u32 emission = blankId;
            u32 prev     = s;

            if (s > 0) {
                f32 v = alphaPrev[s - 1];  // label
                if (v < best) {
                    best     = v;
                    emission = targetsView[s - 1];
                    prev     = s - 1;
                }
            }

            alphaCur[s]              = best + scoresView[t * V + emission];
            backPtr[t * (S + 1) + s] = prev;
        }
        std::swap(alphaPrev, alphaCur);
    }

    std::vector<s32> result(T);
    size_t           s = S;
    size_t           sPrev;

    for (size_t t = T; t > 0; --t) {
        sPrev = backPtr[(t - 1) * (S + 1) + s];
        if (s == sPrev) {
            result[t - 1] = blankId;
        }
        else {
            result[t - 1] = targetsView[s - 1];
        }
        s = sPrev;
    }

    return result;
}
