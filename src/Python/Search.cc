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

#include "Search.hh"

#include <Flf/Map.hh>
#include <Flf/Module.hh>
#include <Flf/NBest.hh>
#include <Flf/RecognizerV2.hh>
#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>

namespace py = pybind11;

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          latticeHandler_(Flf::Module::instance().createLatticeHandler(config)),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV2(select("search-algorithm"))),
          lexicon_(new Flf::Lexicon(select("lexicon"))),
          modelCombination_(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel(), lexicon_) {
    latticeHandler_->setLexicon(lexicon_);
    searchAlgorithm_->setModelCombination(modelCombination_);
}

void SearchAlgorithm::enterSegment() {
    searchAlgorithm_->enterSegment();
}

void SearchAlgorithm::finishSegment() {
    searchAlgorithm_->finishSegment();
}

void SearchAlgorithm::putFeature(py::array_t<f32> const& feature) {
    size_t F = 0ul;
    if (feature.ndim() == 2) {
        if (feature.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << feature.shape(0) << "; should be 1";
        }
        F = feature.shape(1);
    }
    else if (feature.ndim() == 1) {
        F = feature.shape(0);
    }
    else {
        error() << "Received feature vector of invalid dim " << feature.ndim() << "; should be 1 or 2";
    }

    searchAlgorithm_->putFeature({feature, F});
}

void SearchAlgorithm::putFeatures(py::array_t<f32> const& features) {
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

Traceback SearchAlgorithm::searchTracebackToPythonTraceback(Core::Ref<Search::Traceback const> traceback) {
    std::vector<TracebackItem> result;
    result.reserve(traceback->size());

    u32 prevTime = 0;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        if (not it->pronunciation or not it->pronunciation->lemma()) {
            continue;
        }
        result.push_back({
                .lemma     = it->pronunciation->lemma()->symbol(),
                .amScore   = it->score.acoustic,
                .lmScore   = it->score.lm,
                .startTime = prevTime,
                .endTime   = it->time,
        });
        prevTime = it->time;
    }
    return result;
}

Traceback SearchAlgorithm::getCurrentBestTraceback() {
    searchAlgorithm_->decodeManySteps();
    return searchTracebackToPythonTraceback(searchAlgorithm_->getCurrentBestTraceback());
}

Traceback SearchAlgorithm::getCommonPrefix() {
    searchAlgorithm_->decodeManySteps();
    return searchTracebackToPythonTraceback(searchAlgorithm_->getCommonPrefix()->performTraceback());
}

std::vector<Traceback> SearchAlgorithm::getCurrentNBestList(size_t nBestSize) {
    searchAlgorithm_->decodeManySteps();

    auto lattice = searchAlgorithm_->getCurrentBestWordLattice();

    // Use Flf functions to convert search lattice to n-best lattice
    auto flfLattice   = convertSearchLatticeToFlf(lexicon_, lattice, latticeHandler_.get(), "", modelCombination_.languageModel()->scale());
    auto mapLattice   = Flf::mapInput(flfLattice, Flf::MapToLemma);
    auto nBestLattice = Flf::nbest(mapLattice, nBestSize, true);

    Fsa::ConstAlphabetRef alphabet   = nBestLattice->getInputAlphabet();
    auto                  semiring   = nBestLattice->semiring();
    auto                  boundaries = nBestLattice->getBoundaries();

    auto amId = semiring->id("am");
    auto lmId = semiring->id("lm");

    std::vector<Traceback> result;

    auto initialState = nBestLattice->getState(nBestLattice->initialStateId());

    // The nbest lattice looks roughly like this where each outgoing arc from the initial state goes into
    // a linear path which can be followed until the end:
    //  /-----\
    // o ------o
    //  \-----/
    for (auto arcIter = initialState->begin(); arcIter != initialState->end(); ++arcIter) {
        Traceback  tb;
        u32        prevTime = 0;
        Flf::Score amScore  = 0;
        Flf::Score lmScore  = 0;

        auto arc = arcIter;

        while (true) {
            auto nextState = nBestLattice->getState(arc->target());
            auto endTime   = boundaries->time(nextState);

            if (arc->input() != Fsa::Epsilon) {
                auto label = alphabet->symbol(arc->input());

                amScore += arc->score(amId);
                lmScore += arc->score(lmId);

                tb.push_back({.lemma     = label,
                              .amScore   = amScore,
                              .lmScore   = lmScore,
                              .startTime = prevTime,
                              .endTime   = endTime});
            }

            prevTime = endTime;

            if (nextState->hasArcs()) {
                arc = nextState->begin();
            }
            else {
                break;
            }
        }

        result.push_back(tb);
    }

    return result;
}

Traceback SearchAlgorithm::recognizeSegment(py::array_t<f32> const& features) {
    enterSegment();
    putFeatures(features);
    finishSegment();
    return getCurrentBestTraceback();
}

std::vector<Traceback> SearchAlgorithm::recognizeSegmentNBest(py::array_t<f32> const& features, size_t nBestSize) {
    enterSegment();
    putFeatures(features);
    finishSegment();
    return getCurrentNBestList(nBestSize);
}



// Foward-Backward Search

Search::ForwardBackwardSearch const* SearchAlgorithm::requireForwardBackwardSearch(Search::SearchAlgorithmV2 const* searchAlgorithm) {
    auto const* fbSearch = dynamic_cast<Search::ForwardBackwardSearch const*>(searchAlgorithm);

    if (!fbSearch) {
        throw std::runtime_error(
                "The configured search algorithm is not ForwardBackwardSearch. "
                "Use a search-algorithm type that creates Search::ForwardBackwardSearch "
                "before calling forward-backward Python methods.");
    }

    return fbSearch;
}

py::array_t<double> SearchAlgorithm::gammasToNumpy(std::vector<std::vector<double>> const& gammas) {
    size_t numFrames = gammas.size();
    size_t numLabels = 0ul;

    for (auto const& frameGammas : gammas) {
        numLabels = std::max(numLabels, frameGammas.size());
    }

    py::array_t<double> result({numFrames, numLabels});
    auto resultMutable = result.mutable_unchecked<2>();

    for (size_t t = 0ul; t < numFrames; ++t) {
        for (size_t label = 0ul; label < numLabels; ++label) {
            resultMutable(t, label) = 0.0;
        }
    }

    for (size_t t = 0ul; t < numFrames; ++t) {
        for (size_t label = 0ul; label < gammas[t].size(); ++label) {
            resultMutable(t, label) = gammas[t][label];
        }
    }

    return result;
}

py::dict SearchAlgorithm::getForwardBackwardResult() {

    auto const* fbSearch = requireForwardBackwardSearch(searchAlgorithm_.get());

    py::dict result;
    result["partition_cost"] = static_cast<double>(fbSearch->partitionCost());
    result["log_likelihood"] = static_cast<double>(fbSearch->logLikelihood());
    result["label_gammas"] = gammasToNumpy(fbSearch->labelGammas());

    return result;
}

py::dict SearchAlgorithm::recognizeSegmentForwardBackward(py::array_t<f32> const& features) {
    enterSegment();
    putFeatures(features);
    finishSegment();
    return getForwardBackwardResult();
}
