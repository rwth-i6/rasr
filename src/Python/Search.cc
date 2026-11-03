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

namespace {

Traceback searchTracebackToPythonTraceback(Core::Ref<Search::Traceback const> traceback) {
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

std::vector<Traceback> searchLatticeToNBestList(Core::Ref<Search::LatticeAdaptor const> lattice,
                                                Flf::LexiconRef                         lexicon,
                                                Flf::LatticeHandler*                    latticeHandler,
                                                Speech::ModelCombination const&         modelCombination,
                                                size_t                                  nBestSize) {
    // Use Flf functions to convert search lattice to n-best lattice
    auto flfLattice   = convertSearchLatticeToFlf(lexicon, lattice, latticeHandler, "", modelCombination.languageModel()->scale());
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

// Wrap a feature array of shape [F] or [1, F] into a DataView
Nn::DataView featureToDataView(Core::Component const& component, py::array_t<f32> const& feature) {
    size_t F = 0ul;
    if (feature.ndim() == 2) {
        if (feature.shape(0) != 1) {
            component.error() << "Received feature tensor with non-trivial batch dimension " << feature.shape(0) << "; should be 1";
        }
        F = feature.shape(1);
    }
    else if (feature.ndim() == 1) {
        F = feature.shape(0);
    }
    else {
        component.error() << "Received feature vector of invalid dim " << feature.ndim() << "; should be 1 or 2";
    }

    return {feature, F};
}

// Wrap an array of features of shape [T, F] or [1, T, F] into a DataView; returns it together with the number of timesteps T
std::pair<Nn::DataView, size_t> featuresToDataView(Core::Component const& component, py::array_t<f32> const& features) {
    size_t T = 0ul;
    size_t F = 0ul;
    if (features.ndim() == 3) {
        if (features.shape(0) != 1) {
            component.error() << "Received feature tensor with non-trivial batch dimension " << features.shape(0) << "; should be 1";
        }
        T = features.shape(1);
        F = features.shape(2);
    }
    else if (features.ndim() == 2) {
        T = features.shape(0);
        F = features.shape(1);
    }
    else {
        component.error() << "Received feature tensor of invalid dim " << features.ndim() << "; should be 2 or 3";
    }

    return {Nn::DataView(features, T * F), T};
}

}  // namespace

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          latticeHandler_(Flf::Module::instance().createLatticeHandler(config)),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV2(select("search-algorithm"))),
          lexicon_(new Flf::Lexicon(select("lexicon"))),
          modelCombination_(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel(), lexicon_) {
    latticeHandler_->setLexicon(lexicon_);
    searchAlgorithm_->setModelCombination(modelCombination_);
}

SearchAlgorithm::~SearchAlgorithm() {
    closeSegmentLog();
}

Speech::ModelCombination& SearchAlgorithm::modelCombination() {
    return modelCombination_;
}

void SearchAlgorithm::enterSegment(std::string const& name) {
    if (segmentOpen_) {
        warning() << "enter_segment was called while segment " << segmentIndex_ << " is still open; closing it implicitly";
        closeSegmentLog();
    }

    auto open = Core::XmlOpen("segment") + Core::XmlAttribute("index", segmentIndex_);
    if (not name.empty()) {
        open += Core::XmlAttribute("name", name);
    }
    clog() << open;
    segmentOpen_ = true;

    searchAlgorithm_->enterSegment();
}

void SearchAlgorithm::finishSegment() {
    // Run the search first so that its statistics are logged inside the segment element
    searchAlgorithm_->finishSegment();

    if (not segmentOpen_) {
        warning() << "finish_segment was called without a matching enter_segment";
        return;
    }
    closeSegmentLog();
}

void SearchAlgorithm::putFeature(py::array_t<f32> const& feature) {
    searchAlgorithm_->putFeature(featureToDataView(*this, feature));
}

void SearchAlgorithm::putFeatures(py::array_t<f32> const& features) {
    auto [dataView, T] = featuresToDataView(*this, features);
    searchAlgorithm_->putFeatures(dataView, T);
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
    return searchLatticeToNBestList(searchAlgorithm_->getCurrentBestWordLattice(), lexicon_, latticeHandler_.get(), modelCombination_, nBestSize);
}

Traceback SearchAlgorithm::recognizeSegment(py::array_t<f32> const& features, std::string const& name) {
    enterSegment(name);
    putFeatures(features);
    finishSegment();
    return getCurrentBestTraceback();
}

std::vector<Traceback> SearchAlgorithm::recognizeSegmentNBest(py::array_t<f32> const& features, size_t nBestSize, std::string const& name) {
    enterSegment(name);
    putFeatures(features);
    finishSegment();
    return getCurrentNBestList(nBestSize);
}

void SearchAlgorithm::closeSegmentLog() {
    if (not segmentOpen_) {
        return;
    }
    clog() << Core::XmlClose("segment");
    segmentOpen_ = false;
    ++segmentIndex_;
}

BatchedSearchAlgorithm::BatchedSearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          latticeHandler_(Flf::Module::instance().createLatticeHandler(config)),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV3(select("search-algorithm"))),
          lexicon_(new Flf::Lexicon(select("lexicon"))),
          modelCombination_(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel(), lexicon_) {
    latticeHandler_->setLexicon(lexicon_);
    searchAlgorithm_->setModelCombination(modelCombination_);
}

Speech::ModelCombination& BatchedSearchAlgorithm::modelCombination() {
    return modelCombination_;
}

size_t BatchedSearchAlgorithm::maxActiveSegments() const {
    return searchAlgorithm_->maxActiveSegments();
}

size_t BatchedSearchAlgorithm::numActiveSegments() const {
    return searchAlgorithm_->numActiveSegments();
}

BatchedSearchAlgorithm::SegmentHandle BatchedSearchAlgorithm::addSegment(std::string const& name) {
    return searchAlgorithm_->addSegment(name);
}

void BatchedSearchAlgorithm::signalNoMoreFeatures(SegmentHandle segment) {
    searchAlgorithm_->signalNoMoreFeatures(segment);
}

void BatchedSearchAlgorithm::putFeature(SegmentHandle segment, py::array_t<f32> const& feature) {
    searchAlgorithm_->putFeature(segment, featureToDataView(*this, feature));
}

void BatchedSearchAlgorithm::putFeatures(SegmentHandle segment, py::array_t<f32> const& features) {
    auto [dataView, T] = featuresToDataView(*this, features);
    searchAlgorithm_->putFeatures(segment, dataView, T);
}

bool BatchedSearchAlgorithm::decodeStep() {
    return searchAlgorithm_->decodeStep();
}

unsigned BatchedSearchAlgorithm::decodeManySteps() {
    return searchAlgorithm_->decodeManySteps();
}

bool BatchedSearchAlgorithm::isFinished(SegmentHandle segment) const {
    return searchAlgorithm_->isFinished(segment);
}

void BatchedSearchAlgorithm::removeSegment(SegmentHandle segment) {
    searchAlgorithm_->removeSegment(segment);
}

std::vector<BatchedSearchAlgorithm::SegmentHandle> BatchedSearchAlgorithm::segments() const {
    return searchAlgorithm_->segments();
}

Traceback BatchedSearchAlgorithm::getCurrentBestTraceback(SegmentHandle segment) {
    return searchTracebackToPythonTraceback(searchAlgorithm_->getCurrentBestTraceback(segment));
}

Traceback BatchedSearchAlgorithm::getCommonPrefix(SegmentHandle segment) {
    return searchTracebackToPythonTraceback(searchAlgorithm_->getCommonPrefix(segment)->performTraceback());
}

std::vector<Traceback> BatchedSearchAlgorithm::getCurrentNBestList(SegmentHandle segment, size_t nBestSize) {
    return searchLatticeToNBestList(searchAlgorithm_->getCurrentBestWordLattice(segment), lexicon_, latticeHandler_.get(), modelCombination_, nBestSize);
}

std::vector<Traceback> BatchedSearchAlgorithm::recognizeSegments(std::vector<py::array_t<f32>> const& features, std::vector<std::string> const& names) {
    std::vector<Traceback> results(features.size());
    recognizeSegments(features, names, [this, &results](size_t index, SegmentHandle segment) {
        results[index] = getCurrentBestTraceback(segment);
    });
    return results;
}

std::vector<std::vector<Traceback>> BatchedSearchAlgorithm::recognizeSegmentsNBest(std::vector<py::array_t<f32>> const& features, size_t nBestSize, std::vector<std::string> const& names) {
    std::vector<std::vector<Traceback>> results(features.size());
    recognizeSegments(features, names, [this, &results, nBestSize](size_t index, SegmentHandle segment) {
        results[index] = getCurrentNBestList(segment, nBestSize);
    });
    return results;
}

void BatchedSearchAlgorithm::recognizeSegments(std::vector<py::array_t<f32>> const& features, std::vector<std::string> const& names, std::function<void(size_t, SegmentHandle)> const& storeResult) {
    if (not searchAlgorithm_->segments().empty()) {
        error() << "recognize_segments can only be used if no other segments are present in the search";
    }
    if (not names.empty() and names.size() != features.size()) {
        error() << "Got " << names.size() << " segment names for " << features.size() << " segments";
    }

    size_t batchSize = searchAlgorithm_->maxActiveSegments();

    std::vector<std::pair<SegmentHandle, size_t>> activeSegments;  // Handle and input index of all segments in the search
    size_t                                        nextInput = 0ul;
    while (nextInput < features.size() or not activeSegments.empty()) {
        // Fill up the batch with new segments
        while (nextInput < features.size() and activeSegments.size() < batchSize) {
            auto segment = addSegment(names.empty() ? std::string() : names[nextInput]);
            putFeatures(segment, features[nextInput]);
            searchAlgorithm_->signalNoMoreFeatures(segment);
            activeSegments.emplace_back(segment, nextInput);
            ++nextInput;
        }

        bool anythingChanged = searchAlgorithm_->decodeStep();

        // Collect the results of finished segments and remove them from the search
        for (auto it = activeSegments.begin(); it != activeSegments.end();) {
            if (searchAlgorithm_->isFinished(it->first)) {
                storeResult(it->second, it->first);
                searchAlgorithm_->removeSegment(it->first);
                it = activeSegments.erase(it);
            }
            else {
                ++it;
            }
        }

        if (not anythingChanged) {
            error() << "Search made no progress although all features of the active segments were passed";
            break;
        }
    }
}
