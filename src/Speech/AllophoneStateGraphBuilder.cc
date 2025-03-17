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
#define FAST_MODEL

#include <Bliss/Fsa.hh>
#include <Bliss/Orthography.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Output.hh>
#include <Fsa/Project.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Sort.hh>
#include <Fsa/Storage.hh>

#include "Alignment.hh"
#include "AllophoneStateGraphBuilder.hh"
#include "ModelCombination.hh"

using namespace Speech;

const Core::ParameterInt paramMinDuration("label-min-duration", "minimum occurance of speech label", 1);

AllophoneStateGraphBuilder::AllophoneStateGraphBuilder(const Core::Configuration&         c,
                                                       Core::Ref<const Bliss::Lexicon>    lexicon,
                                                       Core::Ref<const Am::AcousticModel> acousticModel,
                                                       bool                               flatModelAcceptor)
        : Precursor(c),
          lexicon_(lexicon),
          orthographicParser_(0),
          acousticModel_(acousticModel),
          modelChannel_(config, "model-automaton"),
          flatModelAcceptor_(flatModelAcceptor),
          minDuration_(paramMinDuration(c)) {}

AllophoneStateGraphBuilder::~AllophoneStateGraphBuilder() {
    delete orthographicParser_;
}

void AllophoneStateGraphBuilder::addSilenceOrNoise(const Bliss::Pronunciation* pron) {
    silencesAndNoises_.push_back(pron);
}

void AllophoneStateGraphBuilder::addSilenceOrNoise(const Bliss::Lemma* lemma) {
    Bliss::Lemma::PronunciationIterator p, p_end;
    for (Core::tie(p, p_end) = lemma->pronunciations(); p != p_end; ++p) {
        addSilenceOrNoise(p->pronunciation());
    }
}

void AllophoneStateGraphBuilder::setSilencesAndNoises(const std::vector<std::string>& silencesAndNoises) {
    verify(silencesAndNoises_.empty());
    std::vector<std::string>::const_iterator noiseIt = silencesAndNoises.begin();
    for (; noiseIt != silencesAndNoises.end(); ++noiseIt) {
        std::string noise(*noiseIt);
        Core::normalizeWhitespace(noise);
        const Bliss::Lemma* lemma = lexicon_->lemma(noise);
        if (lemma) {
            if (lemma->nPronunciations() != 0) {
                addSilenceOrNoise(lemma);
            }
            else {
                warning("did not find a pronunciation");
            }
        }
        else {
            warning("did not find lemma");
        }
    }
}

Bliss::OrthographicParser& AllophoneStateGraphBuilder::orthographicParser() {
    if (!orthographicParser_)
        orthographicParser_ = new Bliss::OrthographicParser(select("orthographic-parser"), lexicon_);
    return *orthographicParser_;
}

Core::Ref<Fsa::StaticAutomaton> AllophoneStateGraphBuilder::lemmaPronunciationToLemmaTransducer() {
    if (!lemmaPronunciationToLemmaTransducer_) {
        auto lemmaPronunciationToLemmaTransducer = lexicon_->createLemmaPronunciationToLemmaTransducer();
        // sort transducer by output symbols to accelerate composition operations
        lemmaPronunciationToLemmaTransducer_ =
                Fsa::staticCompactCopy(Fsa::sort(lemmaPronunciationToLemmaTransducer, Fsa::SortTypeByOutput));

        Fsa::info(lemmaPronunciationToLemmaTransducer_, log("lemma-pronuncation-to-lemma transducer"));
    }
    return lemmaPronunciationToLemmaTransducer_;
}

Core::Ref<Fsa::StaticAutomaton> AllophoneStateGraphBuilder::phonemeToLemmaPronunciationTransducer() {
    if (!phonemeToLemmaPronunciationTransducer_) {
        auto phonemeToLemmaPronunciationTransducer = lexicon_->createPhonemeToLemmaPronunciationTransducer(false);
        // sort transducer by output symbols to accelerate composition operations
        phonemeToLemmaPronunciationTransducer_ =
                Fsa::staticCompactCopy(Fsa::sort(phonemeToLemmaPronunciationTransducer, Fsa::SortTypeByOutput));
        Fsa::info(phonemeToLemmaPronunciationTransducer_, log("phoneme-to-lemma-pronuncation transducer"));
    }
    return phonemeToLemmaPronunciationTransducer_;
}

Core::Ref<Fsa::StaticAutomaton> AllophoneStateGraphBuilder::allophoneStateToPhonemeTransducer() {
    if (!allophoneStateToPhonemeTransducer_) {
        Core::Ref<const Bliss::PhonemeAlphabet> phonemeAlphabet(
                dynamic_cast<const Bliss::PhonemeAlphabet*>(
                        phonemeToLemmaPronunciationTransducer()->getInputAlphabet().get()));

        Core::Ref<Am::TransducerBuilder> tb = acousticModel_->createTransducerBuilder();

        tb->setDisambiguators(phonemeAlphabet->nDisambiguators());
        tb->selectAllophonesFromLexicon();

        // for efficiency reasons, precompute flat allophoneStateToPhoneme transducer
        // without loop and skip transitions and apply transition model (i.e. loops
        // and skips) afterwards on the final transducer
        tb->selectFlatModel();

        tb->selectAllophoneStatesAsInput();
        auto allophoneStateToPhonemeTransducer = tb->createPhonemeLoopTransducer();
        allophoneStateToPhonemeTransducer_ =
                Fsa::staticCompactCopy(Fsa::sort(allophoneStateToPhonemeTransducer, Fsa::SortTypeByOutput));

        // To accelerate the application of context dependency, it would be nice, if allophoneStateToPhonemeTransducer
        // was deterministic wrt. its output symbols:
        // allophoneStateToPhonemeTransducer_ =
        //      Fsa::ConstAutomatonRef(Fsa::staticCompactCopy(Fsa::invert(Fsa::determinize(Fsa::invert(allophoneStateToPhonemeTransducer_)))));
        // Unfortunately, this is currently not possible due to ambiguities at word boundaries
    }
    return allophoneStateToPhonemeTransducer_;
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::singlePronunciationAllophoneStateToPhonemeTransducer() {
    if (!singlePronunciationAllophoneStateToPhonemeTransducer_) {
        Core::Ref<const Bliss::PhonemeAlphabet> phonemeAlphabet(
                dynamic_cast<const Bliss::PhonemeAlphabet*>(
                        phonemeToLemmaPronunciationTransducer()->getInputAlphabet().get()));

        Core::Ref<Am::TransducerBuilder> tb = acousticModel_->createTransducerBuilder();
        tb->selectAllophonesFromLexicon();
        tb->selectCoarticulatedSinglePronunciation();
        if (flatModelAcceptor_)
            tb->selectFlatModel();
        else
            tb->selectTransitionModel();
        tb->selectAllophoneStatesAsInput();
        singlePronunciationAllophoneStateToPhonemeTransducer_ =
                tb->createPhonemeLoopTransducer();
    }
    return singlePronunciationAllophoneStateToPhonemeTransducer_;
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const std::string& orth) {
    return finalizeTransducer(buildTransducer(orth));
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const std::string& orth, const std::string& leftContextOrth, const std::string& rightContextOrth) {
    return finalizeTransducer(buildTransducer(orth, leftContextOrth, rightContextOrth));
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const InputLevel& level) {
    if (level == lemma) {
        return build(Fsa::projectOutput(lemmaPronunciationToLemmaTransducer()));
    }
    else if (level == phone) {
        Core::Ref<Am::TransducerBuilder> tb = acousticModel_->createTransducerBuilder();
        if (flatModelAcceptor_)
            tb->selectFlatModel();
        else
            tb->selectTransitionModel();
        tb->selectAllophoneStatesAsInput();
        return finalizeTransducer(tb->createPhonemeLoopTransducer());
    }
    else {
        criticalError("unknown input level");
        return AllophoneStateGraphRef();
    }
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::buildTransducer(const std::string& orth) {
    return buildTransducer(orthographicParser().createLemmaAcceptor(orth));
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::buildTransducer(std::string const& orth, std::string const& leftContextOrth, std::string const& rightContextOrth) {
    Core::Vector<Fsa::ConstAutomatonRef> lemma_acceptors;
    if (not leftContextOrth.empty()) {
        lemma_acceptors.push_back(Fsa::allSuffixes(orthographicParser().createLemmaAcceptor(leftContextOrth)));
    }
    lemma_acceptors.push_back(orthographicParser().createLemmaAcceptor(orth));
    if (not rightContextOrth.empty()) {
        lemma_acceptors.push_back(Fsa::allPrefixes(orthographicParser().createLemmaAcceptor(rightContextOrth)));
    }

    if (lemma_acceptors.size() == 1) {
        return buildTransducer(lemma_acceptors[0]);
    }
    else {
        return buildTransducer(Fsa::determinize(Fsa::removeEpsilons(Fsa::concat(lemma_acceptors))));
    }
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(Fsa::ConstAutomatonRef lemmaAcceptor) {
    return finalizeTransducer(buildTransducer(lemmaAcceptor));
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::buildFlatTransducer(
        Fsa::ConstAutomatonRef lemmaAcceptor) {
    if (modelChannel_.isOpen()) {
        Fsa::info(lemmaAcceptor, modelChannel_);
        Fsa::drawDot(lemmaAcceptor, "/tmp/lemma-acceptor.dot");
        Fsa::write(lemmaAcceptor, "bin:/tmp/lemma-acceptor.binfsa.gz");
    }

    require(lemmaAcceptor->type() == Fsa::TypeAcceptor);
    require(lemmaAcceptor->getInputAlphabet() == lexicon_->lemmaAlphabet());

    Fsa::AutomatonCounts counts;

    // remove silence and phrases by choosing the shortest path for a flat model acceptor
    // we trim in order to check for not empty but incomplete graphs without final states
    lemmaPronunciationToLemmaTransducer()->setSemiring(lemmaAcceptor->semiring());
    Fsa::ConstAutomatonRef lemmaPronunciationAcceptor =
            Fsa::projectOutput(
                    Fsa::trim(
                            Fsa::composeMatching(
                                    flatModelAcceptor_ ? Fsa::best(Fsa::extend(lemmaAcceptor, Fsa::Weight(1.0))) : Fsa::ConstAutomatonRef(lemmaAcceptor),
                                    Fsa::invert(lemmaPronunciationToLemmaTransducer()))));

    if (modelChannel_.isOpen()) {
        Fsa::info(lemmaPronunciationAcceptor, modelChannel_);
        Fsa::drawDot(lemmaPronunciationAcceptor, "/tmp/lemma-pronunciation.dot");
        Fsa::write(lemmaPronunciationAcceptor, "bin:/tmp/lemma-pronunciation.binfsa.gz");
    }

    if (lemmaPronunciationAcceptor->initialStateId() == Fsa::InvalidStateId)
        criticalError("lemma-pronuncation graph is empty. Probably the current sentence contains a word that has no pronunciation.");

    phonemeToLemmaPronunciationTransducer()->setSemiring(lemmaAcceptor->semiring());
    Fsa::ConstAutomatonRef phon = Fsa::trim(
            Fsa::composeMatching(phonemeToLemmaPronunciationTransducer(), lemmaPronunciationAcceptor));
    if (modelChannel_.isOpen()) {
        Fsa::info(phon, modelChannel_);
        Fsa::drawDot(phon, "/tmp/phon.dot");
        Fsa::write(phon, "bin:/tmp/phon.binfsa.gz");
    }

    if (phon->initialStateId() == Fsa::InvalidStateId)
        criticalError("phoneme graph is empty.  Probably the current sentence contains a word that has no pronunciation.");

    // remove pronunciation variants
    if (flatModelAcceptor_)
        phon = Fsa::best(phon);

    allophoneStateToPhonemeTransducer()->setSemiring(lemmaAcceptor->semiring());
    Fsa::ConstAutomatonRef model = Fsa::trim(
            Fsa::composeMatching(allophoneStateToPhonemeTransducer(), phon));
    if (modelChannel_.isOpen()) {
        Fsa::info(model, modelChannel_);
        Fsa::drawDot(model, "/tmp/allophon.dot");
        Fsa::write(model, "bin:/tmp/allophon.binfsa.gz");
    }
    return model;
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::finishTransducer(Fsa::ConstAutomatonRef model) {
    if (modelChannel_.isOpen()) {
        Fsa::info(model, log());
        Fsa::drawDot(model, "/tmp/states.dot");
        Fsa::write(model, "bin:/tmp/states.binfsa.gz", Fsa::storeStates);
        auto modelNoEps = Fsa::removeEpsilons(Fsa::removeDisambiguationSymbols(Fsa::projectInput(model)));
        Fsa::drawDot(modelNoEps, "/tmp/states-no-eps.dot");
        Fsa::write(modelNoEps, "bin:/tmp/states-no-eps.binfsa.gz", Fsa::storeStates);
    }
    if (model->initialStateId() == Fsa::InvalidStateId)
        criticalError("allophone-state graph is empty.");
    return model;
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::addLoopTransition(Fsa::ConstAutomatonRef model) {
    if (!flatModelAcceptor_) {
        model                               = Fsa::cache(model);
        Core::Ref<Am::TransducerBuilder> tb = acousticModel_->createTransducerBuilder();
        tb->selectAllophoneStatesAsInput();
        tb->selectTransitionModel();
        tb->setDisambiguators(1); // word end disambiguators
        model = tb->applyTransitionModel(model);
        if (modelChannel_.isOpen()) {
            Fsa::info(model, modelChannel_);
            Fsa::drawDot(model, "/tmp/allophon-transiton.dot");
            Fsa::write(model, "bin:/tmp/allophon-transiton.binfsa.gz");
        }
    }
    return model;
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::finalizeTransducer(Fsa::ConstAutomatonRef allophoneStateToLemmaPronuncationTransducer) {
    AllophoneStateGraphRef modelAcceptor = Fsa::removeEpsilons(
            Fsa::removeDisambiguationSymbols(
                    Fsa::projectInput(allophoneStateToLemmaPronuncationTransducer)));

    if (modelChannel_.isOpen()) {
        Fsa::info(modelAcceptor, modelChannel_);
        Fsa::drawDot(modelAcceptor, "/tmp/model.dot");
        Fsa::write(modelAcceptor, "bin:/tmp/model.binfsa.gz", Fsa::storeStates);
    }
    return modelAcceptor;
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const Bliss::Coarticulated<Bliss::Pronunciation>& pronunciation) {
    Core::Ref<Am::TransducerBuilder> tb = acousticModel_->createTransducerBuilder();
    if (flatModelAcceptor_)
        tb->selectFlatModel();
    else
        tb->selectTransitionModel();
    tb->selectAllophoneStatesAsInput();
    tb->setSilencesAndNoises(silencesAndNoises_.empty() ? 0 : &silencesAndNoises_);
    return finalizeTransducer(tb->createPronunciationTransducer(pronunciation));
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const Alignment& alignment) {
    return build(alignment, singlePronunciationAllophoneStateToPhonemeTransducer());
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const Alignment& alignment, const Bliss::Coarticulated<Bliss::Pronunciation>& pronunciation) {
    return build(alignment, build(pronunciation));
}

AllophoneStateGraphRef AllophoneStateGraphBuilder::build(const Alignment& alignment, AllophoneStateGraphRef allophoneStateGraph) {
    Fsa::ConstAutomatonRef model =
            Fsa::trim(Fsa::composeMatching(createAlignmentGraph(alignment),
                                           allophoneStateGraph));
    if (model->initialStateId() == Fsa::InvalidStateId)
        warning("Allophone state graph generated from alignment has no final state.");
    return finalizeTransducer(model);
}

Fsa::ConstAutomatonRef AllophoneStateGraphBuilder::createAlignmentGraph(const Alignment& alignment) {
    Core::Ref<Fsa::StaticAutomaton> f(new Fsa::StaticAutomaton());
    f->setSemiring(Fsa::TropicalSemiring);
    f->setInputAlphabet(acousticModel_->allophoneStateAlphabet());
    f->setType(Fsa::TypeAcceptor);
    f->addProperties(Fsa::PropertyStorage | Fsa::PropertySortedByInput |
                     Fsa::PropertySortedByOutput | Fsa::PropertyAcyclic);

    Fsa::State*  sp  = 0;
    Fsa::StateId sid = f->newState(Fsa::StateTagNone, f->semiring()->one())->id();
    f->setInitialStateId(sid);
    for (Alignment::const_iterator al = alignment.begin(); al != alignment.end(); ++al) {
        sp  = f->fastState(sid);
        sid = f->newState(Fsa::StateTagNone, f->semiring()->one())->id();
        sp->newArc(sid, f->semiring()->one(), al->emission);
    }
    f->fastState(sid)->addTags(Fsa::StateTagFinal);
    f->normalize();
    return f;
}


// -------- HMM Topology --------
Fsa::ConstAutomatonRef HMMTopologyGraphBuilder::buildTransducer(Fsa::ConstAutomatonRef lemmaAcceptor) {
    Fsa::ConstAutomatonRef model = buildFlatTransducer(lemmaAcceptor);
    model = addLoopTransition(model);
    if (minDuration_ > 1)
        model = applyMinimumDuration(model);
    return finishTransducer(model);
}

Fsa::ConstAutomatonRef HMMTopologyGraphBuilder::applyMinimumDuration(Fsa::ConstAutomatonRef model) {
    Fsa::LabelId silenceId = acousticModel_->silenceAllophoneStateIndex();
    Core::Ref<Fsa::StaticAutomaton> automaton = Fsa::staticCopy(model);
    Fsa::ConstAlphabetRef inAlphabet = automaton->getInputAlphabet();

    std::deque<Fsa::StateId> stateQueue;
    std::unordered_set<Fsa::StateId> doneStates;
    stateQueue.push_back(automaton->initialStateId());
    Fsa::StateId staticMaxId = automaton->maxStateId();

    while (!stateQueue.empty()) {
        Fsa::StateId s = stateQueue.front();
        stateQueue.pop_front();
        if (doneStates.count(s) > 0)
            continue;
        // newly inserted states should not be traversed
        verify(s <= staticMaxId);

        Fsa::State* state = automaton->fastState(s);
        u32 nArcs = state->nArcs();
        for (u32 idx = 0; idx < nArcs; ++idx) {
            Fsa::Arc* a = const_cast<Fsa::Arc*>(state->getArc(idx));
            Fsa::StateId target = a->target_;
            Fsa::LabelId input = a->input_;
            stateQueue.push_back(target);
            if (target == s || input == silenceId || inAlphabet->isDisambiguator(input) ||
                Score(a->weight_) >= Core::Type<Score>::max)
                continue;
            // repeat forward with zero weight
            for (u32 repeat = 1; repeat < minDuration_; ++repeat) { 
                Fsa::State* ns = automaton->newState();
                ns->newArc(target, Fsa::Weight(0), input, Fsa::Epsilon);
                target = automaton->maxStateId();
            }
            a->target_ = target;
        }
        doneStates.insert(s);
    }

    return automaton;
}

// -------- CTC Topology --------
CTCTopologyGraphBuilder::CTCTopologyGraphBuilder(const Core::Configuration& config,
                                                 Core::Ref<const Bliss::Lexicon> lexicon,
                                                 Core::Ref<const Am::AcousticModel> acousticModel,
                                                 bool flatModelAcceptor) :
        Precursor(config, lexicon, acousticModel, flatModelAcceptor),
        transitionChecked_(false),
        finalStateId_(Core::Type<Fsa::StateId>::max) {
    // Note: not emission index yet
    blankId_ = acousticModel_->blankAllophoneStateIndex();
    verify(blankId_ != Fsa::InvalidLabelId);
    log() << "blank allophone id " << blankId_;
    // silence is allowed but not necessarily used
    silenceId_ = acousticModel_->silenceAllophoneStateIndex(); 
}

void CTCTopologyGraphBuilder::checkTransitionModel() {
    if (transitionChecked_)
        return;

    // label loop, no skip, no weights: realized via transition model
    bool correct = true;
    for (u32 idx = 0, total = acousticModel_->nStateTransitions(); idx < total; ++idx) {
        const Am::StateTransitionModel& st = *(acousticModel_->stateTransition(idx));
        bool correct = st[Am::StateTransitionModel::forward] == 0 &&
                       st[Am::StateTransitionModel::skip] >= Core::Type<Score>::max &&
                       st[Am::StateTransitionModel::exit] == 0;
        if (idx == Am::TransitionModel::entryM1 || idx == Am::TransitionModel::entryM2)
            correct = correct && st[Am::StateTransitionModel::loop] >= Core::Type<Score>::max;
        else
            correct = correct && st[Am::StateTransitionModel::loop] == 0;
        if (!correct)
            criticalError("wrong transitions ! please set forward:0, skip:inf, exit:0 and loop:inf(entry)/0(*)");
    }
    verify(correct);
    transitionChecked_ = true;
}

Fsa::ConstAutomatonRef CTCTopologyGraphBuilder::addLoopTransition(Fsa::ConstAutomatonRef model) {
    checkTransitionModel();
    return Precursor::addLoopTransition(model);
}

Fsa::ConstAutomatonRef CTCTopologyGraphBuilder::buildTransducer(Fsa::ConstAutomatonRef lemmaAcceptor) {
    Fsa::ConstAutomatonRef model = buildFlatTransducer(lemmaAcceptor);
    model = addLoopTransition(model);
    // remove epsilon so that repeated identical label detection could work
    model = Fsa::removeEpsilons(Fsa::removeDisambiguationSymbols(Fsa::projectInput(model)));
    Core::Ref<Fsa::StaticAutomaton> automaton = Fsa::staticCopy(model);

    finalStateId_ = Core::Type<Fsa::StateId>::max;
    std::deque<Fsa::StateId> stateQueue;
    std::unordered_set<Fsa::StateId> doneStates;
    stateQueue.push_back(automaton->initialStateId());
    Fsa::StateId staticMaxId = automaton->maxStateId();

    while (!stateQueue.empty()) {
        Fsa::StateId s = stateQueue.front();
        stateQueue.pop_front();
        if (doneStates.count(s) > 0)
            continue;
        // newly inserted states should not be traversed
        verify(s <= staticMaxId);
        addBlank(automaton, s, stateQueue);
        doneStates.insert(s);
    }

    return finishTransducer(automaton);
}

void CTCTopologyGraphBuilder::addBlank(Core::Ref<Fsa::StaticAutomaton>& automaton,
                                       Fsa::StateId s,
                                       std::deque<Fsa::StateId>& stateQueue) {
    Fsa::ConstAlphabetRef inAlphabet = automaton->getInputAlphabet();
    Fsa::Weight zeroWeight(0);
    Fsa::State* state = automaton->fastState(s);
    u32 nArcs = state->nArcs();
    // find non-blank loop label for later consecutive identical label handling
    Fsa::LabelId loopLabel = Fsa::InvalidLabelId;
    for (u32 idx = 0; idx < nArcs; ++idx) {
        const Fsa::Arc* arcLoop = state->getArc(idx);
        if ((arcLoop->target_ == s) && (arcLoop->input_ != blankId_)) {
            loopLabel = arcLoop->input_;
            break;
        }
    }
    for (u32 idx = 0; idx < nArcs; ++idx) {
        const Fsa::Arc* a = state->getArc(idx);
        Fsa::StateId target = a->target_;
        Fsa::LabelId input = a->input_;
        stateQueue.push_back(target);
        // skip loop and useless arcs for blank
        if (target == s || input == blankId_ || inAlphabet->isDisambiguator(input) ||
            Score(a->weight_) >= Core::Type<Score>::max)
            continue;
        // add blank
        Fsa::State* blankState = automaton->newState();
        Fsa::StateId blankStateId = automaton->maxStateId();
        blankState->newArc(blankStateId, zeroWeight, blankId_, Fsa::Epsilon);
        blankState->newArc(target, a->weight_, input, a->output_);
        // handle consecutive identical label: if label loop and forward represent the same label,
        // we should overwrite the original arc target to make the blank unskippable
        if (loopLabel != Fsa::InvalidLabelId && 
            acousticModel_->emissionIndex(input) == acousticModel_->emissionIndex(loopLabel)) {
            Fsa::Arc* aa = const_cast<Fsa::Arc*>(state->getArc(idx));
            aa->target_ = blankStateId;
            aa->input_ = blankId_;
            aa->weight_ = zeroWeight;
        } else {
            state->newArc(blankStateId, zeroWeight, blankId_, Fsa::Epsilon);  // optional blank
        }


        // apply minimum duration here to avoid traversing the automaton again
        if (minDuration_ > 1 && input != silenceId_) {
            // repeat forward with zero weight
            for (u32 repeat = 1; repeat < minDuration_; ++repeat) {
                Fsa::State* ns = automaton->newState();
                ns->newArc(target, zeroWeight, input, Fsa::Epsilon);
                target = automaton->maxStateId();
            }
            Fsa::Arc* aa = const_cast<Fsa::Arc*>(state->getArc(idx));
            aa->target_ = target;
            blankState->rbegin()->target_ = target;
        }
    }

    // tailing blanks: loop on a single additional final state
    if (state->isFinal()) {
        if (finalStateId_ == Core::Type<Fsa::StateId>::max) {
            Fsa::State* finalState = automaton->newState();
            finalStateId_ = automaton->maxStateId();
            verify(finalStateId_ != Core::Type<Fsa::StateId>::max);
            finalState->newArc(finalStateId_, zeroWeight, blankId_, Fsa::Epsilon);
            automaton->setStateFinal(finalState);
        }
        state->newArc(finalStateId_, zeroWeight, blankId_, Fsa::Epsilon);
    }
}


// -------- RNA Topology --------
void RNATopologyGraphBuilder::addBlank(Core::Ref<Fsa::StaticAutomaton>& automaton,
                                       Fsa::StateId s,
                                       std::deque<Fsa::StateId>& stateQueue) {
    Fsa::State* state = automaton->fastState(s);
    u32 nArcs = state->nArcs();
    for (u32 idx = 0; idx < nArcs; ++idx) {
        const Fsa::Arc* a = state->getArc(idx);
        Fsa::StateId target = a->target_;
        stateQueue.push_back(target);
        verify(target != s); // no loop
    }
    state->newArc(s, Fsa::Weight(0), blankId_, Fsa::Epsilon);
}


