#include "Search.hh"
#include <Flf/LatticeHandler.hh>
#include <Flf/Module.hh>
#include <Flow/Data.hh>
#include <Fsa/tBest.hh>
#include <Nn/Types.hh>

#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>

namespace py = pybind11;

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithm(select("search-algorithm"))),
          modelCombination_(config) {
    modelCombination_.build(searchAlgorithm_->modelCombinationNeeded(), searchAlgorithm_->acousticModelNeeded());
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

void SearchAlgorithm::addFeature(py::array_t<double> feature) {
    if (feature.ndim() != 1) {
        error() << "Received feature vector of invalid dim " << feature.ndim() << "; should be 1";
    }

    // Read-only view of the array (without bounds checking)
    auto buffer = feature.unchecked<1>();

    // Shape of the array: [F]
    auto F = buffer.shape(1);

    std::vector<f32> currentFeature;
    currentFeature.reserve(F);
    for (ssize_t f = 0ul; f < F; ++f) {
        currentFeature.push_back(buffer(f));
    }
    addFeatureInternal(currentFeature);
}

void SearchAlgorithm::addFeatures(py::array_t<double> features) {
    if (features.ndim() != 2) {
        error() << "Received feature tensor of invalid dim " << features.ndim() << "; should be 2";
    }

    // Read-only view of the array (without bounds checking)
    auto buffer = features.unchecked<2>();

    // Shape of the array: [T, F]
    auto T = buffer.shape(0);
    auto F = buffer.shape(1);

    // Iterate over time axis and slice off each individual feature vector
    for (ssize_t t = 0ul; t < T; ++t) {
        std::vector<f32> currentFeature;
        currentFeature.reserve(F);
        for (ssize_t f = 0ul; f < F; ++f) {
            currentFeature.push_back(buffer(t, f));
        }
        addFeatureInternal(currentFeature);
    }
}

void SearchAlgorithm::addFeatureInternal(const std::vector<float>& feature) {
    // Abuse feature index as timestamp
    searchAlgorithm_->addFeature(Flow::dataPtr(new Nn::FeatureVector(feature, currentFeatureIdx_, currentFeatureIdx_ + 1)));
    ++currentFeatureIdx_;
}

std::string SearchAlgorithm::getCurrentBestTranscription() {
    decodeMore();

    auto lattice = buildLattice(searchAlgorithm_->getCurrentBestWordLattice());
    auto best    = Ftl::best<Flf::Lattice>(lattice);

    std::stringstream ss;

    for (Flf::ConstStateRef stateRef = best->getState(best->initialStateId()); stateRef->hasArcs();
         stateRef                    = best->getState(stateRef->begin()->target())) {
        const Flf::Arc& arc(*stateRef->begin());

        ss << best->getInputAlphabet()->symbol(arc.input()) << " ";
    }

    return ss.str();
}

bool SearchAlgorithm::decodeMore() {
    return searchAlgorithm_->decodeMore();
}

std::string SearchAlgorithm::recognizeSegment(py::array_t<double> features) {
    enterSegment();
    addFeatures(features);
    finishSegment();
    decodeMore();
    return getCurrentBestTranscription();
}

Flf::ConstLatticeRef SearchAlgorithm::buildLattice(Core::Ref<const Search::LatticeAdaptor> la) {
    auto semiring = Flf::Semiring::create(Fsa::SemiringTypeTropical, 2);
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
    handler->setLexicon(modelCombination_.lexicon());
    if (la->empty()) {
        return Flf::ConstLatticeRef();
    }
    ::Lattice::ConstWordLatticeRef             lattice    = la->wordLattice(handler);
    Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
    Fsa::ConstAutomatonRef                     amFsa      = lattice->part(::Lattice::WordLattice::acousticFsa);
    Fsa::ConstAutomatonRef                     lmFsa      = lattice->part(::Lattice::WordLattice::lmFsa);
    require_(Fsa::isAcyclic(amFsa) && Fsa::isAcyclic(lmFsa));

    Flf::StaticBoundariesRef b = Flf::StaticBoundariesRef(new Flf::StaticBoundaries);
    Flf::StaticLatticeRef    s = Flf::StaticLatticeRef(new Flf::StaticLattice);
    s->setType(Fsa::TypeAcceptor);
    s->setProperties(Fsa::PropertyAcyclic | Flf::PropertyCrossWord, Fsa::PropertyAll);
    s->setInputAlphabet(modelCombination_.lexicon()->lemmaPronunciationAlphabet());
    s->setSemiring(semiring);
    s->setDescription("recog");
    s->setBoundaries(Flf::ConstBoundariesRef(b));
    s->setInitialStateId(0);

    Flf::Time timeOffset = (*boundaries)[amFsa->initialStateId()].time();

    Fsa::Stack<Fsa::StateId>   S;
    Core::Vector<Fsa::StateId> sidMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
    sidMap[amFsa->initialStateId()] = 0;
    S.push_back(amFsa->initialStateId());
    Fsa::StateId nextSid   = 2;
    Flf::Time    finalTime = 0;
    while (!S.isEmpty()) {
        Fsa::StateId sid = S.pop();
        verify(sid < sidMap.size());
        const ::Lattice::WordBoundary& boundary((*boundaries)[sid]);
        Fsa::ConstStateRef             amSr = amFsa->getState(sid);
        Fsa::ConstStateRef             lmSr = lmFsa->getState(sid);
        Flf::State*                    sp   = new Flf::State(sidMap[sid]);
        s->setState(sp);
        b->set(sp->id(), Flf::Boundary(boundary.time() - timeOffset,
                                       Flf::Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
        if (amSr->isFinal()) {
            auto scores = semiring->create();
            scores->set(0, amSr->weight());
            scores->set(1, static_cast<Flf::Score>(lmSr->weight()) / semiring->scale(1));
            sp->newArc(1, scores, sentenceEndLabel);
            finalTime = std::max(finalTime, boundary.time() - timeOffset);
        }
        for (Fsa::State::const_iterator am_a = amSr->begin(), lm_a = lmSr->begin(); (am_a != amSr->end()) && (lm_a != lmSr->end()); ++am_a, ++lm_a) {
            sidMap.grow(am_a->target(), Fsa::InvalidStateId);
            if (sidMap[am_a->target()] == Fsa::InvalidStateId) {
                sidMap[am_a->target()] = nextSid++;
                S.push(am_a->target());
            }
            Fsa::ConstStateRef targetAmSr = amFsa->getState(am_a->target());
            Fsa::ConstStateRef targetLmSr = amFsa->getState(lm_a->target());
            if (targetAmSr->isFinal() && targetLmSr->isFinal()) {
                if (am_a->input() == Fsa::Epsilon) {
                    auto scores = semiring->create();
                    scores->set(0, am_a->weight());
                    scores->set(1, static_cast<Flf::Score>(lm_a->weight()) / semiring->scale(1));
                    scores->add(0, Flf::Score(targetAmSr->weight()));
                    scores->add(1, Flf::Score(targetLmSr->weight()) / semiring->scale(1));
                    sp->newArc(1, scores, sentenceEndLabel);
                }
                else {
                    auto scores = semiring->create();
                    scores->set(0, am_a->weight());
                    scores->set(1, static_cast<Flf::Score>(lm_a->weight()) / semiring->scale(1));
                    sp->newArc(sidMap[am_a->target()], scores, am_a->input());
                }
            }
            else {
                auto scores = semiring->create();
                scores->set(0, am_a->weight());
                scores->set(1, static_cast<Flf::Score>(lm_a->weight()) / semiring->scale(1));
                sp->newArc(sidMap[am_a->target()], scores, am_a->input());
            }
        }
    }
    Flf::State* sp = new Flf::State(1);
    sp->setFinal(semiring->clone(semiring->one()));
    s->setState(sp);
    b->set(sp->id(), Flf::Boundary(finalTime));
    return s;
}
