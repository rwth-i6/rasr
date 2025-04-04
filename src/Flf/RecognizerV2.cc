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
#include "RecognizerV2.hh"
#include <Core/XmlStream.hh>
#include <Fsa/Types.hh>
#include <Speech/ModelCombination.hh>
#include <chrono>
#include "LatticeHandler.hh"
#include "Module.hh"

namespace Flf {

NodeRef createRecognizerNodeV2(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new RecognizerNodeV2(name, config));
}

RecognizerNodeV2::RecognizerNodeV2(const std::string& name, const Core::Configuration& config)
        : Node(name, config),
          latticeResultBuffer_(),
          segmentResultBuffer_(),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithmV2(select("search-algorithm"))),
          modelCombination_() {
    Core::Configuration featureExtractionConfig(config, "feature-extraction");
    DataSourceRef       dataSource = DataSourceRef(Speech::Module::instance().createDataSource(featureExtractionConfig));
    featureExtractor_              = SegmentwiseFeatureExtractorRef(new SegmentwiseFeatureExtractor(featureExtractionConfig, dataSource));
}

void RecognizerNodeV2::recognizeSegment(const Bliss::SpeechSegment* segment) {
    if (!segment->orth().empty()) {
        clog() << Core::XmlOpen("orth") + Core::XmlAttribute("source", "reference")
               << segment->orth()
               << Core::XmlClose("orth");
    }

    // Initialize recognizer and feature extractor
    searchAlgorithm_->reset();
    searchAlgorithm_->enterSegment();

    featureExtractor_->enterSegment(segment);
    DataSourceRef dataSource = featureExtractor_->extractor();

    auto timerStart = std::chrono::steady_clock::now();

    dataSource->initialize(const_cast<Bliss::SpeechSegment*>(segment));
    FeatureRef feature;
    dataSource->getData(feature);
    Time startTime = feature->timestamp().startTime();
    Time endTime;

    // Loop over features and perform recognition
    do {
        searchAlgorithm_->putFeature(*feature->mainStream());
        endTime = feature->timestamp().endTime();
    } while (dataSource->getData(feature));

    searchAlgorithm_->finishSegment();
    dataSource->finalize();
    featureExtractor_->leaveSegment(segment);

    // Result processing and logging
    auto traceback = searchAlgorithm_->getCurrentBestTraceback();

    auto lattice         = buildLattice(searchAlgorithm_->getCurrentBestWordLattice(), segment->name());
    latticeResultBuffer_ = lattice;
    segmentResultBuffer_ = SegmentRef(new Flf::Segment(segment));

    Core::XmlWriter& os(clog());
    os << Core::XmlOpen("traceback");
    traceback->write(os, modelCombination_->lexicon()->phonemeInventory());
    os << Core::XmlClose("traceback");

    os << Core::XmlOpen("orth") + Core::XmlAttribute("source", "recognized");
    for (auto const& tracebackItem : *traceback) {
        if (tracebackItem.pronunciation and tracebackItem.pronunciation->lemma()) {
            os << tracebackItem.pronunciation->lemma()->preferredOrthographicForm() << Core::XmlBlank();
        }
    }
    os << Core::XmlClose("orth");

    auto   timerEnd       = std::chrono::steady_clock::now();
    double duration       = std::chrono::duration<double, std::milli>(timerEnd - timerStart).count();
    double signalDuration = (endTime - startTime) * 1000.;  // convert duration to ms

    clog() << Core::XmlOpen("flf-recognizer-time") + Core::XmlAttribute("unit", "milliseconds") << duration << Core::XmlClose("flf-recognizer-time");
    clog() << Core::XmlOpen("flf-recognizer-rtf") << (duration / signalDuration) << Core::XmlClose("flf-recognizer-rtf");
}

void RecognizerNodeV2::work() {
    clog() << Core::XmlOpen("layer") + Core::XmlAttribute("name", name);
    recognizeSegment(static_cast<const Bliss::SpeechSegment*>(requestData(0)));
    clog() << Core::XmlClose("layer");
}

ConstLatticeRef RecognizerNodeV2::buildLattice(Core::Ref<const Search::LatticeAdaptor> latticeAdaptor, std::string segmentName) {
    auto lmScale = modelCombination_->languageModel()->scale();

    auto semiring = Semiring::create(Fsa::SemiringTypeTropical, 2);
    semiring->setKey(0, "am");
    semiring->setScale(0, 1.0);
    semiring->setKey(1, "lm");
    semiring->setScale(1, lmScale);

    auto                sentenceEndLabel        = Fsa::Epsilon;
    const Bliss::Lemma* specialSentenceEndLemma = modelCombination_->lexicon()->specialLemma("sentence-end");
    if (specialSentenceEndLemma and specialSentenceEndLemma->nPronunciations() > 0) {
        sentenceEndLabel = specialSentenceEndLemma->pronunciations().first->id();
    }

    Flf::LatticeHandler* handler = Flf::Module::instance().createLatticeHandler(config);
    handler->setLexicon(Lexicon::us());
    if (latticeAdaptor->empty()) {
        return ConstLatticeRef();
    }
    ::Lattice::ConstWordLatticeRef             lattice    = latticeAdaptor->wordLattice(handler);
    Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
    Fsa::ConstAutomatonRef                     amFsa      = lattice->part(::Lattice::WordLattice::acousticFsa);
    Fsa::ConstAutomatonRef                     lmFsa      = lattice->part(::Lattice::WordLattice::lmFsa);
    require_(Fsa::isAcyclic(amFsa) && Fsa::isAcyclic(lmFsa));

    StaticBoundariesRef flfBoundaries = StaticBoundariesRef(new StaticBoundaries);
    StaticLatticeRef    flfLattice    = StaticLatticeRef(new StaticLattice);
    flfLattice->setType(Fsa::TypeAcceptor);
    flfLattice->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
    flfLattice->setInputAlphabet(modelCombination_->lexicon()->lemmaPronunciationAlphabet());
    flfLattice->setSemiring(semiring);
    flfLattice->setDescription(Core::form("recog(%s)", segmentName.c_str()));
    flfLattice->setBoundaries(ConstBoundariesRef(flfBoundaries));
    flfLattice->setInitialStateId(0);

    Time timeOffset = (*boundaries)[amFsa->initialStateId()].time();

    Fsa::Stack<Fsa::StateId>   stateStack;
    Core::Vector<Fsa::StateId> stateIdMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
    stateIdMap[amFsa->initialStateId()] = 0;
    stateStack.push_back(amFsa->initialStateId());
    Fsa::StateId nextStateId = 2;
    Time         finalTime   = 0;
    while (not stateStack.isEmpty()) {
        Fsa::StateId stateId = stateStack.pop();
        verify(stateId < stateIdMap.size());
        const ::Lattice::WordBoundary& boundary((*boundaries)[stateId]);
        Fsa::ConstStateRef             amFsaState = amFsa->getState(stateId);
        Fsa::ConstStateRef             lmFsaState = lmFsa->getState(stateId);
        State*                         flfState   = new State(stateIdMap[stateId]);
        flfLattice->setState(flfState);
        flfBoundaries->set(flfState->id(), Boundary(boundary.time() - timeOffset,
                                                    Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
        if (amFsaState->isFinal()) {
            auto scores = semiring->create();
            scores->set(0, amFsaState->weight());
            if (lmScale) {
                scores->set(1, static_cast<Score>(lmFsaState->weight()) / lmScale);
            }
            else {
                scores->set(1, 0.0);
            }
            flfState->newArc(1, scores, sentenceEndLabel);
            finalTime = std::max(finalTime, boundary.time() - timeOffset);
        }
        for (Fsa::State::const_iterator amArc = amFsaState->begin(), lmArc = lmFsaState->begin(); (amArc != amFsaState->end()) && (lmArc != lmFsaState->end()); ++amArc, ++lmArc) {
            stateIdMap.grow(amArc->target(), Fsa::InvalidStateId);
            if (stateIdMap[amArc->target()] == Fsa::InvalidStateId) {
                stateIdMap[amArc->target()] = nextStateId++;
                stateStack.push(amArc->target());
            }
            Fsa::ConstStateRef targetAmState = amFsa->getState(amArc->target());
            Fsa::ConstStateRef targetLmState = amFsa->getState(lmArc->target());

            auto scores = semiring->create();
            scores->set(0, amArc->weight());

            if (lmScale) {
                scores->set(1, static_cast<Score>(lmArc->weight()) / lmScale);
            }
            else {
                scores->set(1, 0);
            }

            if (targetAmState->isFinal() and targetLmState->isFinal() and amArc->input() == Fsa::Epsilon) {
                scores->add(0, Score(targetAmState->weight()));
                if (lmScale) {
                    scores->add(1, Score(targetLmState->weight()) / lmScale);
                }
                flfState->newArc(1, scores, sentenceEndLabel);
            }
            else {
                flfState->newArc(stateIdMap[amArc->target()], scores, amArc->input());
            }
        }
    }
    State* finalState = new State(1);
    finalState->setFinal(semiring->clone(semiring->one()));
    flfLattice->setState(finalState);
    flfBoundaries->set(finalState->id(), Boundary(finalTime));
    return flfLattice;
}

void RecognizerNodeV2::init(std::vector<std::string> const& arguments) {
    modelCombination_ = Core::ref(new Speech::ModelCombination(
            config,
            searchAlgorithm_->requiredModelCombination(),
            searchAlgorithm_->requiredAcousticModel(),
            Lexicon::us()));
    searchAlgorithm_->setModelCombination(*modelCombination_);
    if (not connected(0)) {
        criticalError("Speech segment at port 0 required");
    }
}

void RecognizerNodeV2::sync() {
    latticeResultBuffer_.reset();
    segmentResultBuffer_.reset();
}

void RecognizerNodeV2::finalize() {
    searchAlgorithm_->reset();
}

ConstSegmentRef RecognizerNodeV2::sendSegment(RecognizerNodeV2::Port to) {
    if (not segmentResultBuffer_) {
        work();
    }
    return segmentResultBuffer_;
}

ConstLatticeRef RecognizerNodeV2::sendLattice(RecognizerNodeV2::Port to) {
    if (not latticeResultBuffer_) {
        work();
    }
    return latticeResultBuffer_;
}

}  // namespace Flf
