/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "RecognizerV3.hh"

#include <Core/XmlStream.hh>
#include <Speech/Module.hh>

#include "Lexicon.hh"
#include "Module.hh"
#include "RecognizerV2.hh"

namespace Flf {

NodeRef createRecognizerNodeV3(std::string const& name, Core::Configuration const& config) {
    return NodeRef(new RecognizerNodeV3(name, config));
}

ConstSegmentRef detachedSegment(Bliss::SpeechSegment const* blissSegment) {
    // Fill the attributes via a temporary segment and copy them without the Bliss segment pointer
    Segment    full(blissSegment);
    SegmentRef segment(new Segment());
    if (full.hasAudioFilename()) {
        segment->setAudioFilename(full.audioFilename());
    }
    segment->setStartTime(full.startTime());
    segment->setEndTime(full.endTime());
    segment->setTrack(full.track());
    segment->setOrthography(full.orthography());
    if (full.hasSpeakerId()) {
        segment->setSpeakerId(full.speakerId());
    }
    if (full.hasConditionId()) {
        segment->setConditionId(full.conditionId());
    }
    if (full.hasRecordingId()) {
        segment->setRecordingId(full.recordingId());
    }
    segment->setSegmentId(full.segmentId());
    return segment;
}

RecognizerNodeV3::RecognizerNodeV3(std::string const& name, Core::Configuration const& config)
        : Node(name, config),
          pendingSegments_(),
          latticeResultBuffer_(),
          segmentResultBuffer_(),
          latticeHandler_(Flf::Module::instance().createLatticeHandler(config)),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV3(select("search-algorithm"))),
          modelCombination_(),
          totalRecognitionTime_(0.0),
          totalSignalDuration_(0.0) {
    latticeHandler_->setLexicon(Lexicon::us());
    Core::Configuration featureExtractionConfig(config, "feature-extraction");
    DataSourceRef       dataSource = DataSourceRef(Speech::Module::instance().createDataSource(featureExtractionConfig));
    featureExtractor_              = SegmentwiseFeatureExtractorRef(new SegmentwiseFeatureExtractor(featureExtractionConfig, dataSource));
}

void RecognizerNodeV3::init(std::vector<std::string> const& arguments) {
    modelCombination_ = Core::ref(new Speech::ModelCombination(
            config,
            searchAlgorithm_->requiredModelCombination(),
            searchAlgorithm_->requiredAcousticModel(),
            Lexicon::us()));
    searchAlgorithm_->setModelCombination(*modelCombination_);
    log() << "Decode up to " << maxActiveSegments() << " segments at the same time";
}

void RecognizerNodeV3::sync() {
    latticeResultBuffer_.reset();
    segmentResultBuffer_.reset();
}

void RecognizerNodeV3::finalize() {
    if (not pendingSegments_.empty()) {
        warning() << pendingSegments_.size() << " segments were not processed completely";
    }
    if (totalSignalDuration_ > 0.0) {
        clog() << Core::XmlOpen("flf-recognizer-total-time") + Core::XmlAttribute("unit", "milliseconds") << totalRecognitionTime_ << Core::XmlClose("flf-recognizer-total-time");
        clog() << Core::XmlOpen("flf-recognizer-total-rtf") << (totalRecognitionTime_ / totalSignalDuration_) << Core::XmlClose("flf-recognizer-total-rtf");
    }
}

ConstSegmentRef RecognizerNodeV3::sendSegment(Port to) {
    if (not segmentResultBuffer_) {
        criticalError("recognizer-v3 has no result; it has to be driven by the batched corpus processor");
    }
    return segmentResultBuffer_;
}

ConstLatticeRef RecognizerNodeV3::sendLattice(Port to) {
    if (not segmentResultBuffer_) {
        criticalError("recognizer-v3 has no result; it has to be driven by the batched corpus processor");
    }
    return latticeResultBuffer_;
}

size_t RecognizerNodeV3::maxActiveSegments() const {
    return searchAlgorithm_->maxActiveSegments();
}

size_t RecognizerNodeV3::numActiveSegments() const {
    return searchAlgorithm_->numActiveSegments();
}

void RecognizerNodeV3::addSegment(Bliss::SpeechSegment const* segment) {
    auto timerStart = std::chrono::steady_clock::now();

    auto handle = searchAlgorithm_->addSegment(segment->fullName());

    featureExtractor_->enterSegment(segment);
    DataSourceRef dataSource = featureExtractor_->extractor();
    dataSource->initialize(const_cast<Bliss::SpeechSegment*>(segment));

    FeatureRef feature;
    Time       startTimestamp = 0;
    Time       endTimestamp   = 0;
    if (dataSource->getData(feature)) {
        startTimestamp = feature->timestamp().startTime();
        do {
            searchAlgorithm_->putFeature(handle, feature->mainStream());
            endTimestamp = feature->timestamp().endTime();
        } while (dataSource->getData(feature));
    }

    searchAlgorithm_->signalNoMoreFeatures(handle);
    dataSource->finalize();
    featureExtractor_->leaveSegment(segment);

    pendingSegments_.push_back({.handle          = handle,
                                .segment         = detachedSegment(segment),
                                .name            = segment->fullName(),
                                .referenceOrth   = segment->orth(),
                                .finished        = false,
                                .lattice         = ConstLatticeRef(),
                                .traceback       = Core::Ref<Search::Traceback const>(),
                                .startTime       = timerStart,
                                .recognitionTime = 0.0,
                                .signalDuration  = (endTimestamp - startTimestamp) * 1000.0});

    totalRecognitionTime_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timerStart).count();
    totalSignalDuration_ += pendingSegments_.back().signalDuration;
}

bool RecognizerNodeV3::hasPendingSegments() const {
    return not pendingSegments_.empty();
}

bool RecognizerNodeV3::nextResultReady() const {
    return not pendingSegments_.empty() and pendingSegments_.front().finished;
}

void RecognizerNodeV3::decodeStep() {
    auto timerStart = std::chrono::steady_clock::now();

    bool anythingChanged = searchAlgorithm_->decodeStep();
    collectFinishedSegments();

    totalRecognitionTime_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timerStart).count();

    if (not anythingChanged and not nextResultReady()) {
        criticalError("Search made no progress although all features of the active segments were passed");
    }
}

ConstSegmentRef RecognizerNodeV3::prepareNextResult() {
    verify(nextResultReady());
    auto pendingSegment = std::move(pendingSegments_.front());
    pendingSegments_.pop_front();

    logResult(pendingSegment);

    latticeResultBuffer_ = pendingSegment.lattice;
    segmentResultBuffer_ = pendingSegment.segment;
    return segmentResultBuffer_;
}

void RecognizerNodeV3::collectFinishedSegments() {
    for (auto& pendingSegment : pendingSegments_) {
        if (pendingSegment.finished or not searchAlgorithm_->isFinished(pendingSegment.handle)) {
            continue;
        }
        pendingSegment.finished  = true;
        pendingSegment.traceback = searchAlgorithm_->getCurrentBestTraceback(pendingSegment.handle);
        pendingSegment.lattice   = convertSearchLatticeToFlf(
                Lexicon::us(),
                searchAlgorithm_->getCurrentBestWordLattice(pendingSegment.handle),
                latticeHandler_.get(),
                pendingSegment.segment->segmentId(),
                modelCombination_->languageModel()->scale());
        pendingSegment.recognitionTime = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pendingSegment.startTime).count();
        searchAlgorithm_->removeSegment(pendingSegment.handle);
    }
}

void RecognizerNodeV3::logResult(PendingSegment const& pendingSegment) {
    Core::XmlWriter& os(clog());
    os << Core::XmlOpen("layer") + Core::XmlAttribute("name", name) + Core::XmlAttribute("segment", pendingSegment.name);

    if (not pendingSegment.referenceOrth.empty()) {
        os << Core::XmlOpen("orth") + Core::XmlAttribute("source", "reference")
           << pendingSegment.referenceOrth
           << Core::XmlClose("orth");
    }

    os << Core::XmlOpen("traceback");
    pendingSegment.traceback->write(os, modelCombination_->lexicon()->phonemeInventory());
    os << Core::XmlClose("traceback");

    os << Core::XmlOpen("orth") + Core::XmlAttribute("source", "recognized");
    for (auto const& tracebackItem : *pendingSegment.traceback) {
        if (tracebackItem.pronunciation and tracebackItem.pronunciation->lemma()) {
            os << tracebackItem.pronunciation->lemma()->preferredOrthographicForm() << Core::XmlBlank();
        }
    }
    os << Core::XmlClose("orth");

    // Latency of the segment: includes the decoding of the other segments that were decoded at the same time
    os << Core::XmlOpen("flf-recognizer-latency") + Core::XmlAttribute("unit", "milliseconds") << pendingSegment.recognitionTime << Core::XmlClose("flf-recognizer-latency");
    os << Core::XmlClose("layer");
}

}  // namespace Flf
