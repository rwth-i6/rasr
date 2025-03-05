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
#include <Speech/ModelCombination.hh>
#include <chrono>
#include "Core/XmlStream.hh"
#include "LatticeHandler.hh"
#include "Module.hh"

namespace Flf {

NodeRef createRecognizerNodeV2(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new RecognizerNodeV2(name, config));
}

RecognizerNodeV2::RecognizerNodeV2(const std::string& name, const Core::Configuration& config)
        : Node(name, config),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithm(select("search-algorithm"))),
          modelCombination_(config) {
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
    dataSource->initialize(const_cast<Bliss::SpeechSegment*>(segment));
    FeatureRef feature;
    dataSource->getData(feature);
    Time startTime = feature->timestamp().startTime();
    Time endTime;

    auto timerStart = std::chrono::steady_clock::now();

    // Loop over features and perform recognition
    do {
        searchAlgorithm_->putFeature(*feature->mainStream());
        endTime = feature->timestamp().endTime();
    } while (dataSource->getData(feature));

    searchAlgorithm_->finishSegment();
    searchAlgorithm_->decodeManySteps();
    dataSource->finalize();
    featureExtractor_->leaveSegment(segment);

    // Result processing and logging
    auto traceback = searchAlgorithm_->getCurrentBestTraceback();

    auto lattice  = buildLattice(searchAlgorithm_->getCurrentBestWordLattice(), segment->name());
    resultBuffer_ = std::make_pair(lattice, SegmentRef(new Flf::Segment(segment)));

    Core::XmlWriter& os(clog());
    os << Core::XmlOpen("traceback");
    traceback->write(os, modelCombination_.lexicon()->phonemeInventory());
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
    auto semiring = Semiring::create(Fsa::SemiringTypeTropical, 2);
    semiring->setKey(0, "am");
    semiring->setScale(0, 1.0);
    semiring->setKey(1, "lm");
    semiring->setScale(1, modelCombination_.languageModel()->scale());

    auto                sentenceEndLabel        = Fsa::Epsilon;
    const Bliss::Lemma* specialSentenceEndLemma = modelCombination_.lexicon()->specialLemma("sentence-end");
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

    StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
    StaticLatticeRef    s = StaticLatticeRef(new StaticLattice);
    s->setType(Fsa::TypeAcceptor);
    s->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
    s->setInputAlphabet(modelCombination_.lexicon()->lemmaPronunciationAlphabet());
    s->setSemiring(semiring);
    s->setDescription(Core::form("recog(%s)", segmentName.c_str()));
    s->setBoundaries(ConstBoundariesRef(b));
    s->setInitialStateId(0);

    Time timeOffset = (*boundaries)[amFsa->initialStateId()].time();

    Fsa::Stack<Fsa::StateId>   stateStack;
    Core::Vector<Fsa::StateId> sidMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
    sidMap[amFsa->initialStateId()] = 0;
    stateStack.push_back(amFsa->initialStateId());
    Fsa::StateId nextSid   = 2;
    Time         finalTime = 0;
    while (not stateStack.isEmpty()) {
        Fsa::StateId sid = stateStack.pop();
        verify(sid < sidMap.size());
        const ::Lattice::WordBoundary& boundary((*boundaries)[sid]);
        Fsa::ConstStateRef             amSr = amFsa->getState(sid);
        Fsa::ConstStateRef             lmSr = lmFsa->getState(sid);
        State*                         sp   = new State(sidMap[sid]);
        s->setState(sp);
        b->set(sp->id(), Boundary(boundary.time() - timeOffset,
                                  Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
        if (amSr->isFinal()) {
            auto scores = semiring->create();
            scores->set(0, amSr->weight());
            scores->set(1, static_cast<Score>(lmSr->weight()) / semiring->scale(1));
            sp->newArc(1, scores, sentenceEndLabel);
            finalTime = std::max(finalTime, boundary.time() - timeOffset);
        }
        for (Fsa::State::const_iterator am_a = amSr->begin(), lm_a = lmSr->begin(); (am_a != amSr->end()) && (lm_a != lmSr->end()); ++am_a, ++lm_a) {
            sidMap.grow(am_a->target(), Fsa::InvalidStateId);
            if (sidMap[am_a->target()] == Fsa::InvalidStateId) {
                sidMap[am_a->target()] = nextSid++;
                stateStack.push(am_a->target());
            }
            Fsa::ConstStateRef targetAmSr = amFsa->getState(am_a->target());
            Fsa::ConstStateRef targetLmSr = amFsa->getState(lm_a->target());
            if (targetAmSr->isFinal() && targetLmSr->isFinal()) {
                if (am_a->input() == Fsa::Epsilon) {
                    auto scores = semiring->create();
                    scores->set(0, am_a->weight());
                    scores->set(1, static_cast<Score>(lm_a->weight()) / semiring->scale(1));
                    scores->add(0, Score(targetAmSr->weight()));
                    scores->add(1, Score(targetLmSr->weight()) / semiring->scale(1));
                    sp->newArc(1, scores, sentenceEndLabel);
                }
                else {
                    auto scores = semiring->create();
                    scores->set(0, am_a->weight());
                    scores->set(1, static_cast<Score>(lm_a->weight()) / semiring->scale(1));
                    sp->newArc(sidMap[am_a->target()], scores, am_a->input());
                }
            }
            else {
                auto scores = semiring->create();
                scores->set(0, am_a->weight());
                scores->set(1, static_cast<Score>(lm_a->weight()) / semiring->scale(1));
                sp->newArc(sidMap[am_a->target()], scores, am_a->input());
            }
        }
    }
    State* sp = new State(1);
    sp->setFinal(semiring->clone(semiring->one()));
    s->setState(sp);
    b->set(sp->id(), Boundary(finalTime));
    return s;
}

void RecognizerNodeV2::init(std::vector<std::string> const& arguments) {
    modelCombination_.build(searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel(), Lexicon::us());
    searchAlgorithm_->setModelCombination(modelCombination_);
    if (not connected(0)) {
        criticalError("Speech segment at port 1 required");
    }
}

void RecognizerNodeV2::sync() {
    resultBuffer_.first.reset();
    resultBuffer_.second.reset();
}

void RecognizerNodeV2::finalize() {
    searchAlgorithm_->reset();
}

ConstSegmentRef RecognizerNodeV2::sendSegment(RecognizerNodeV2::Port to) {
    if (!resultBuffer_.second) {
        work();
    }
    return resultBuffer_.second;
}

ConstLatticeRef RecognizerNodeV2::sendLattice(RecognizerNodeV2::Port to) {
    if (!resultBuffer_.first) {
        work();
    }
    return resultBuffer_.first;
}

}  // namespace Flf
