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

#include <Flf/Best.hh>
#include <Flf/Convert.hh>
#include <Flf/Draw.hh>
#include <Flf/FlfCore/Semiring.hh>
#include <Flf/FlfCore/Types.hh>
#include <Flf/FwdBwd.hh>
#include <Flf/Map.hh>
#include <Flf/Module.hh>
#include <Flf/NBest.hh>
#include <Flf/RecognizerV2.hh>
#include <Flf/TimeframeConfusionNetwork.hh>
#include <Flf/TimeframeConfusionNetworkBuilder.hh>
#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>
#include "Fsa/Types.hh"

namespace py = pybind11;

const Core::ParameterBool SearchAlgorithm::paramConfidenceScores(
        "add-confidence-scores",
        "Include confidence scores in the traceback items",
        false);

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          addConfidenceScores_(paramConfidenceScores(config)),
          latticeHandler_(Flf::Module::instance().createLatticeHandler(config)),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV2(select("search-algorithm"))),
          lexicon_(new Flf::Lexicon(select("lexicon"))),
          modelCombination_(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel(), lexicon_) {
    Flf::Module::instance().setLexicon(lexicon_.get());
    searchAlgorithm_->setModelCombination(modelCombination_);
}

void SearchAlgorithm::reset() {
    searchAlgorithm_->reset();
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
                .lemma           = it->pronunciation->lemma()->symbol(),
                .amScore         = it->score.acoustic,
                .lmScore         = it->score.lm,
                .confidenceScore = std::nullopt,
                .startTime       = prevTime,
                .endTime         = it->time,
        });
        prevTime = it->time;
    }
    return result;
}

Traceback SearchAlgorithm::getTracebackWithConfidence() {
    auto lattice = searchAlgorithm_->getCurrentBestWordLattice();

    auto flfLattice = convertSearchLatticeToFlf(lattice, latticeHandler_.get(), "", modelCombination_.languageModel()->scale());

    auto semiring         = flfLattice->semiring();
    auto confidenceId     = semiring->size();
    auto extendedSemiring = Flf::appendSemiring(semiring, 0.0, "confidence");
    flfLattice            = Flf::offsetSemiring(flfLattice, extendedSemiring, 0);

    auto                fwdBwdBuilder = Flf::FwdBwdBuilder::create(select("fb"));
    Flf::ConstFwdBwdRef fwdBwd;
    std::tie(flfLattice, fwdBwd) = fwdBwdBuilder->build(flfLattice);
    auto cn                      = Flf::buildFramePosteriorCn(flfLattice, fwdBwd);
    auto confidenceLattice       = Flf::extendByFCnConfidence(flfLattice, cn, confidenceId, Flf::RescoreModeInPlaceCache);

    auto mapLattice        = Flf::mapInput(confidenceLattice, Flf::MapToLemma);
    auto singleBestLattice = Flf::nbest(mapLattice, 1, true);
    // Flf::drawDot(singleBestLattice, "nBestLattice.dot");
    // singleBestLattice = Flf::best(mapLattice, Flf::BellmanFord);
    // Flf::drawDot(singleBestLattice, "singleLattice.dot");

    Fsa::ConstAlphabetRef alphabet   = singleBestLattice->getInputAlphabet();
    auto                  boundaries = singleBestLattice->getBoundaries();

    auto amId = extendedSemiring->id("am");
    auto lmId = extendedSemiring->id("lm");

    Traceback result;

    auto initialState = singleBestLattice->getState(singleBestLattice->initialStateId());

    auto        nextState       = initialState;
    auto const* arc             = initialState->getArc(0);
    u32         prevTime        = 0;
    Flf::Score  amScore         = 0;
    Flf::Score  lmScore         = 0;
    Flf::Score  confidenceScore = 0;

    while (true) {
        nextState    = singleBestLattice->getState(arc->target());
        auto endTime = boundaries->time(nextState->id());

        if (arc->input() != Fsa::Epsilon) {
            auto label = alphabet->symbol(arc->input());

            amScore += arc->score(amId);
            lmScore += arc->score(lmId);
            confidenceScore = arc->score(confidenceId);

            result.push_back({.lemma           = label,
                              .amScore         = amScore,
                              .lmScore         = lmScore,
                              .confidenceScore = confidenceScore,
                              .startTime       = prevTime,
                              .endTime         = endTime});
        }

        prevTime = endTime;
        if (nextState->hasArcs()) {
            arc = nextState->getArc(0);
        }
        else {
            break;
        }
    }

    return result;
}

Traceback SearchAlgorithm::getCurrentBestTraceback() {
    searchAlgorithm_->decodeManySteps();

    if (addConfidenceScores_) {
        return getTracebackWithConfidence();
    }
    else {
        return searchTracebackToPythonTraceback(searchAlgorithm_->getCurrentBestTraceback());
    }
}

Traceback SearchAlgorithm::getCurrentStableTraceback() {
    searchAlgorithm_->decodeManySteps();
    return searchTracebackToPythonTraceback(searchAlgorithm_->getCurrentStableTraceback());
}

std::vector<Traceback> SearchAlgorithm::getCurrentNBestList(size_t nBestSize) {
    searchAlgorithm_->decodeManySteps();

    auto lattice = searchAlgorithm_->getCurrentBestWordLattice();

    auto flfLattice   = convertSearchLatticeToFlf(lattice, latticeHandler_.get(), "", modelCombination_.languageModel()->scale());
    auto mapLattice   = Flf::mapInput(flfLattice, Flf::MapToLemma);
    auto nBestLattice = Flf::nbest(mapLattice, nBestSize, true);

    Fsa::ConstAlphabetRef alphabet   = nBestLattice->getInputAlphabet();
    auto                  semiring   = nBestLattice->semiring();
    auto                  boundaries = nBestLattice->getBoundaries();

    auto amId = semiring->id("am");
    auto lmId = semiring->id("lm");

    std::vector<Traceback> result;

    auto initialState = nBestLattice->getState(nBestLattice->initialStateId());

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

                tb.push_back({label,
                              amScore,
                              lmScore,
                              prevTime,
                              endTime});
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
    reset();
    enterSegment();
    putFeatures(features);
    finishSegment();
    return getCurrentBestTraceback();
}

std::vector<Traceback> SearchAlgorithm::recognizeSegmentNBest(py::array_t<f32> const& features, size_t nBestSize) {
    reset();
    enterSegment();
    putFeatures(features);
    finishSegment();
    return getCurrentNBestList(nBestSize);
}
