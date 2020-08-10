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
#include <Am/ClassicTransducerBuilder.hh>
#include <Bliss/Lexicon.hh>
#include <Core/CompressedStream.hh>
#include <Core/Debug.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Project.hh>
#include <Fsa/Sort.hh>
#include <OpenFst/Count.hh>
#include <OpenFst/Input.hh>
#include <OpenFst/LabelMap.hh>
#include <OpenFst/Output.hh>
#include <OpenFst/Relabel.hh>
#include <OpenFst/Scale.hh>
#include <OpenFst/Utility.hh>
#include <Search/Wfst/ContextTransducerBuilder.hh>
#include <Search/Wfst/CreateOperations.hh>
#include <Search/Wfst/LabelMapper.hh>
#include <Search/Wfst/LexiconBuilder.hh>
#include <Search/Wfst/NonWordTokens.hh>
#include <Search/Wfst/StateSequence.hh>
#include <Search/Wfst/StateTree.hh>
#include <fst/arc-map.h>
#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/project.h>
#include <fst/relabel.h>

using namespace Search::Wfst;
using namespace Search::Wfst::Builder;

const Core::ParameterBool BuildGrammar::paramAddEmptySyntacticTokens(
        "add-empty-tokens",
        "add empty syntactic token sequences (set to false is G is minimized using log semiring)",
        true);

const Core::ParameterBool BuildGrammar::paramAddSentenceBoundaries(
        "add-sentence-boundaries",
        "add symbols for the sentence boundary to the G transducer",
        false);

const Core::ParameterBool BuildGrammar::paramAddSentenceBegin(
        "add-sentence-begin",
        "add symbols for the sentence begin to the G transducer",
        false);

const Core::ParameterBool BuildGrammar::paramAddSentenceEnd(
        "add-sentence-end",
        "add symbols for the sentence end to the G transducer",
        false);

Operation::AutomatonRef BuildGrammar::process() {
    Fsa::ConstAutomatonRef fsaG = resources_.languageModel()->getFsa();
    Fsa::AutomatonCounts   cnt  = Fsa::count(fsaG);
    log("G synt: %d states %zu arcs", cnt.nStates_, cnt.nArcs_);
    Lm::Token          se            = resources_.languageModel()->unscaled()->sentenceEndToken();
    const Fsa::LabelId sentenceEnd   = se->id();
    Lm::Token          sb            = resources_.languageModel()->unscaled()->sentenceBeginToken();
    const Fsa::LabelId sentenceBegin = sb->id();
    resources_.deleteLanguageModel();
    bool addSentenceBegin = paramAddSentenceBegin(Operation::config);
    bool addSentenceEnd   = paramAddSentenceEnd(Operation::config);
    log("add sentence begin: %d, add sentence end: %d", addSentenceBegin, addSentenceEnd);
    if (paramAddSentenceBoundaries(Operation::config))
        addSentenceBegin = addSentenceEnd = true;
    if (addSentenceBegin || addSentenceEnd) {
        log("adding sentence boundaries");
        Core::Ref<Fsa::StaticAutomaton> sg = Fsa::staticCopy(fsaG);
        fsaG.reset();
        addSentenceBoundaries(sg, sentenceBegin, sentenceEnd,
                              addSentenceBegin, addSentenceEnd);
        fsaG = sg;
    }

    u32          nDisambiguators = countDisambiguators(fsaG->getInputAlphabet());
    AutomatonRef g               = OpenFst::convertFromFsa<Fsa::Automaton, Automaton>(fsaG);
    fsaG.reset();

    if (outputType() != outputSyntacticTokens) {
        g = mapSymbols(g);
    }
    g->setAttribute(Automaton::attrNumDisambiguators, nDisambiguators);
    return g;
}

Operation::AutomatonRef BuildGrammar::mapSymbols(AutomatonRef g) const {
    u32 nDisambiguators = 0;
    require(outputType() != outputSyntacticTokens);
    bool addEps = paramAddEmptySyntacticTokens(Operation::config);
    log("creating lemma to syntactic token transducer ")
            << (addEps ? "with" : "without") << "empty syntactic tokens";
    Core::Ref<const Bliss::Lexicon> lexicon = resources_.lexicon();
    FstLib::StdProjectFst *         result = 0, *intermediate = 0;
    OpenFst::VectorFst *            l2s = 0, *lp2l = 0;

    l2s = OpenFst::convertFromFsa(lexicon->createLemmaToSyntacticTokenTransducer(addEps, nDisambiguators));
    log("projecting to lemmas");
    FstLib::ArcSort(l2s, FstLib::StdOLabelCompare());
    result = new FstLib::StdProjectFst(FstLib::StdComposeFst(*l2s, *g), FstLib::PROJECT_INPUT);
    if (outputType() == outputLemmaPronunciations) {
        lp2l = OpenFst::convertFromFsa(lexicon->createLemmaPronunciationToLemmaTransducer(nDisambiguators));
        log("projecting to lemma pronunciations");
        Mm::Score pronunciationScale = resources_.pronunciationScale();
        if (pronunciationScale != 1.0) {
            log("applying pronunciation scale %f", pronunciationScale);
            OpenFst::scaleWeights(lp2l, pronunciationScale);
        }
        intermediate = result;
        result       = new FstLib::StdProjectFst(FstLib::StdComposeFst(*lp2l, *intermediate), FstLib::PROJECT_INPUT);
    }
    AutomatonRef staticFst = new Automaton(*result);
    delete g;
    delete intermediate;
    delete l2s;
    delete lp2l;
    delete result;
    return staticFst;
}

void BuildGrammar::addSentenceBoundaries(Core::Ref<Fsa::StaticAutomaton> g,
                                         Fsa::LabelId                    sentenceBegin,
                                         Fsa::LabelId                    sentenceEnd,
                                         bool addBegin, bool addEnd) const {
    if (addEnd) {
        Fsa::State* finalState = g->newState();
        log("new final state: %d", finalState->id());
        finalState->setFinal(g->semiring()->one());
        Fsa::StateId final = finalState->id();
        DBG(1) << "final: " << final << std::endl;
        for (Fsa::StateId s = 0; s <= g->maxStateId(); ++s) {
            Fsa::State::Ref state = g->state(s);
            if (!state) {
                warning("invalid state id: ") << s;
                continue;
            }
            verify(state);
            if (!state->isFinal() || s == final)
                continue;
            DBG(1) << "final state: " << state->id() << " " << (float)state->weight() << std::endl;
            Fsa::Weight weight = state->weight();
            state->newArc(final, weight, sentenceEnd);
            state->unsetFinal();
            state->setWeight(g->semiring()->zero());
            DBG(1) << state->id() << " -> " << final << std::endl;
        }
    }
    if (addBegin) {
        Fsa::State* initial = g->newState();
        log("new initial state: %d", initial->id());
        initial->newArc(g->initialStateId(), g->semiring()->one(), sentenceBegin);
        g->setInitialStateId(initial->id());
    }
    g->unsetProperties(Fsa::PropertySortedByInput);
}

const char* BuildLexicon::attrInitialPhoneOffset  = "initialPhoneOffset";
const char* BuildLexicon::attrWordLabelOffset     = "wordLabelOffset";
const char* BuildLexicon::attrDisambiguatorOffset = "disambiguatorOffset";

const Core::ParameterBool BuildLexicon::paramCloseLexicon(
        "close", "build closure", true);

const Core::ParameterBool BuildLexicon::paramCloseWithSilence(
        "close-with-silence", "add silence/noise arcs for closure", true);

Operation::AutomatonRef BuildLexicon::process() {
    Core::Ref<const Bliss::Lexicon> lexicon = resources_.lexicon();
    LexiconBuilder                  builder(Operation::select("lexicon-builder"), *lexicon);
    log("using %d disambiguators", nDisambiguators_);
    builder.setGrammarDisambiguators(nDisambiguators_);
    const bool          closeL           = paramCloseLexicon(Operation::config);
    const bool          closeWithSilence = paramCloseWithSilence(Operation::config);
    bool                buildClosed      = closeL && !closeWithSilence;
    OpenFst::VectorFst* l                = builder.build(buildClosed);
    if (outputType() != outputLemmaPronunciations) {
        if (builder.addWordDisambiguators())
            error("cannot use output type other than lemma pronunciations");
        mapOutputSymbols(l);
    }
    if (closeL && closeWithSilence) {
        log("building closure");
        if (closeWithSilence)
            log("using silence/noise arcs for closure");
        builder.close(l, closeWithSilence);
    }
    AutomatonRef result = new Automaton();
    FstLib::Cast(*l, result);
    delete l;
    result->setAttribute(attrInitialPhoneOffset, builder.initialPhoneOffset());
    result->setAttribute(attrWordLabelOffset, builder.wordLabelOffset());
    result->setAttribute(attrDisambiguatorOffset, builder.disambiguatorOffset());
    result->setAttribute(Automaton::attrNumDisambiguators, builder.nPhoneDisambiguators());
    return result;
}

void BuildLexicon::mapOutputSymbols(OpenFst::VectorFst* l) const {
    Core::Ref<const Bliss::Lexicon> lexicon = resources_.lexicon();
    log("mapping output symbols");
    if ((outputType() == outputLemmas) || (outputType() == outputSyntacticTokens)) {
        OpenFst::VectorFst* lp2l               = OpenFst::convertFromFsa(lexicon->createLemmaPronunciationToLemmaTransducer(nDisambiguators_));
        Mm::Score           pronunciationScale = resources_.pronunciationScale();
        if (pronunciationScale != 1.0) {
            log("applying pronunciation scale %f", pronunciationScale);
            OpenFst::scaleWeights(lp2l, pronunciationScale);
        }
        FstLib::ComposeOptions opts(true, FstLib::SEQUENCE_FILTER);
        FstLib::Compose(*l, *lp2l, l, opts);
        delete lp2l;
    }
    if (outputType() == outputSyntacticTokens) {
        OpenFst::VectorFst*    l2s = OpenFst::convertFromFsa(lexicon->createLemmaToSyntacticTokenTransducer(true, nDisambiguators_));
        FstLib::ComposeOptions opts(true, FstLib::SEQUENCE_FILTER);
        FstLib::Compose(*l, *l2s, l);
        delete l2s;
    }
}

Operation::AutomatonRef BuildOldLexicon::process() {
    if (!paramCloseLexicon(Operation::config)) {
        warning("lexicon is always closed using this construction");
    }
    log("using %d disambiguators", nDisambiguators_);
    Core::Ref<const Bliss::Lexicon> lexicon      = resources_.lexicon();
    const bool                      isAcrossWord = resources_.acousticModel()->isAcrossWordModelEnabled();
    Fsa::ConstAutomatonRef          l            = lexicon->createPhonemeToLemmaPronunciationTransducer(nDisambiguators_, true, isAcrossWord);
    Fsa::AlphabetCounts             a            = count(l->getInputAlphabet());
    AutomatonRef                    result       = OpenFst::convertFromFsa<Fsa::Automaton, Automaton>(l);
    l.reset();
    mapOutputSymbols(result);
    result->setAttribute(Automaton::attrNumDisambiguators, a.nDisambiguators_);
    Fsa::LabelId initialPhoneOffset = a.maxLabelId_ + 1;
    result->setAttribute(attrInitialPhoneOffset, initialPhoneOffset);
    result->setAttribute(attrWordLabelOffset, -1);
    u32 disambiguatorOffset = lexicon->phonemeInventory()->phonemeAlphabet()->disambiguator(0);
    result->setAttribute(attrDisambiguatorOffset, disambiguatorOffset);
    log("disambiguators in lexicon: %d", a.nDisambiguators_);
    log("initial phone offset in lexicon: %d", initialPhoneOffset);
    return result;
}

Operation::AutomatonRef CloseLexicon::process() {
    log("building closure");
    Core::Ref<const Bliss::Lexicon> lexicon = resources_.lexicon();
    LexiconBuilder                  builder(Operation::select("lexicon-builder"), *lexicon);
    const int                       initialPhoneOffset = input_->getIntAttribute(BuildLexicon::attrInitialPhoneOffset);
    const int                       wordLabelOffset    = input_->getIntAttribute(BuildLexicon::attrWordLabelOffset);
    builder.setInitialPhoneOffset(initialPhoneOffset);
    builder.setWordLabelOffset(wordLabelOffset);
    const bool closeWithSilence = BuildLexicon::paramCloseWithSilence(config);
    if (closeWithSilence)
        log("using silence/noise arcs for closure");
    builder.close(input_, closeWithSilence);
    return input_;
}

const Core::Choice LemmaMapping::mapChoice(
        "pronunciation-to-lemma", LemmaPronunciationToLemma,
        "lemma-to-syntactic-token", LemmaToSyntacticToken,
        Core::Choice::endMark());

const Core::ParameterChoice LemmaMapping::paramMapType(
        "type", &mapChoice, "type of mapping", LemmaPronunciationToLemma);

const Core::ParameterFloat LemmaMapping::paramScale(
        "scale", "weight scaling factor", 1.0);

Operation::AutomatonRef LemmaMapping::process() {
    Core::Ref<const Bliss::Lexicon> lexicon = resources_.lexicon();
    MapType                         mType   = static_cast<MapType>(paramMapType(config));
    nDisambiguators_                        = std::max(nDisambiguators_, 0);
    log("using %d disambiguators", nDisambiguators_);
    Fsa::ConstAutomatonRef m;
    switch (mType) {
        case LemmaPronunciationToLemma:
            log("lemma pronunciation to lemma mapping");
            m = lexicon->createLemmaPronunciationToLemmaTransducer(nDisambiguators_);
            break;
        case LemmaToSyntacticToken:
            log("lemma to syntactic token mapping");
            m = lexicon->createLemmaToSyntacticTokenTransducer(true, nDisambiguators_);
            break;
        default:
            defect();
            break;
    }
    f32 scale = paramScale(config);
    if (scale != 1.0) {
        log("applying scale %f", scale);
        m = Fsa::multiply(m, Fsa::Weight(scale));
    }
    AutomatonRef result = OpenFst::convertFromFsa<Fsa::Automaton, Automaton>(m);
    m.reset();
    return result;
}

const Core::ParameterFloat AddPronunciationWeight::paramScale(
        "scale", "weight scaling factor", 1.0);

Operation::AutomatonRef AddPronunciationWeight::process() {
    nDisambiguators_ = std::max(nDisambiguators_, 0);
    log("using %d disambiguators", nDisambiguators_);

    AutomatonRef     result = new Automaton();
    OpenFst::StateId state  = result->AddState();
    result->SetFinal(state, OpenFst::Weight::One());
    result->SetStart(state);
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lpa = resources_.lexicon()->lemmaPronunciationAlphabet();
    Bliss::Lexicon::PronunciationIterator              piBegin, piEnd;
    Core::tie(piBegin, piEnd) = resources_.lexicon()->pronunciations();
    for (Bliss::Lexicon::PronunciationIterator pi = piBegin; pi != piEnd; ++pi) {
        Bliss::Pronunciation::LemmaIterator liBegin, liEnd;
        Core::tie(liBegin, liEnd) = (*pi)->lemmas();
        for (Bliss::Pronunciation::LemmaIterator li = liBegin; li != liEnd; ++li) {
            const Bliss::LemmaPronunciation* lemmaPron = li;
            OpenFst::Weight                  weight    = OpenFst::Weight(lemmaPron->pronunciationScore());
            OpenFst::Label                   label     = OpenFst::convertLabelFromFsa(lemmaPron->id());
            result->AddArc(state, OpenFst::Arc(label, label, weight, state));
        }
    }
    for (s32 d = 0; d < nDisambiguators_; ++d) {
        OpenFst::Label label = OpenFst::convertLabelFromFsa(lpa->disambiguator(d));
        result->AddArc(state, OpenFst::Arc(label, label, OpenFst::Weight::One(), state));
    }
    OpenFst::SymbolTable* symbols = OpenFst::convertAlphabet(lpa, "lemma-pronunciations");
    result->SetInputSymbols(symbols);
    result->SetOutputSymbols(symbols);
    FstLib::ArcSort(result, FstLib::StdILabelCompare());
    f32 scale = paramScale(config);
    if (scale != 1.0) {
        log("applying scale %f", scale);
        OpenFst::scaleWeights(result, scale);
    }
    return result;
}

Operation::AutomatonRef RestoreOutputSymbols::process() {
    log("project to input");
    FstLib::SymbolTable* outputSymbols = input_->OutputSymbols()->Copy();
    FstLib::Project(input_, FstLib::PROJECT_INPUT);
    log("restoring output symbols");
    int wordLabelOffset = input_->getIntAttribute(BuildLexicon::attrWordLabelOffset);
    verify(wordLabelOffset != Automaton::InvalidIntAttribute);
    verify(wordLabelOffset > 0);
    int disambiguatorOffset = input_->getIntAttribute(BuildLexicon::attrDisambiguatorOffset);
    verify(disambiguatorOffset != Automaton::InvalidIntAttribute);
    verify(disambiguatorOffset > 0);
    log("word label offset: %d", wordLabelOffset);
    log("disambiguator offset: %d", disambiguatorOffset);
    FstLib::ArcMap(input_,
                   RestoreOutputLabelMapper<OpenFst::Arc>(wordLabelOffset,
                                                          disambiguatorOffset));
    input_->SetOutputSymbols(outputSymbols);
    return input_;
}

bool RemovePhoneDisambiguators::addInput(AutomatonRef f) {
    if (nDisambiguators_ < 0 && !DisambiguatorDependentOperation::addInput(f)) {
        DisambiguatorDependentOperation::error("disambiguator count required");
        return false;
    }
    return SleeveOperation::addInput(f);
}

Operation::AutomatonRef RemovePhoneDisambiguators::process() {
    log("removing phone disambiguators");
    u32 disambiguatorOffset = input_->getIntAttribute(BuildLexicon::attrDisambiguatorOffset);
    log("using disambiguator offset %d", disambiguatorOffset);
    log("using %d disambiguators", nDisambiguators_);
    RemoveDisambiguatorMapper<OpenFst::Arc> mapper(disambiguatorOffset, disambiguatorOffset + nDisambiguators_);
    FstLib::ArcMap(input_, mapper);
    return input_;
}

Operation::AutomatonRef PushOutputLabels::process() {
    log("pushing output labels");
    pushOutputLabels(input_);
    return input_;
}

const Core::ParameterString CheckLabels::paramStateSequences(
        "state-sequences", "state sequences file", "");

Operation::AutomatonRef CheckLabels::process() {
    log("checking labels");
    std::string stateSequences = paramStateSequences(config);
    log("loading state sequences: %s", stateSequences.c_str());
    StateSequenceList ssl;
    if (!ssl.read(stateSequences))
        criticalError("cannot read state sequences");
    for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
        for (OpenFst::ArcIterator aiter(*input_, siter.Value()); !aiter.Done(); aiter.Next()) {
            const OpenFst::Arc& arc = aiter.Value();
            if (arc.olabel != OpenFst::Epsilon && arc.ilabel != OpenFst::Epsilon) {
                const StateSequence& ss = ssl[arc.ilabel - 1];
                if (!ss.isFinal()) {
                    log("invalid arc labels: state=%d output=%d, input=%d, initial=%d, final=%d",
                        siter.Value(), arc.olabel, arc.ilabel, ss.isInitial(), ss.isFinal());
                }
            }
        }
    }
    return input_;
}

const Core::ParameterFloat AddNonWordTokens::paramWeight(
        "weight", "weight used for non-word tokens", 0.0);
const Core::ParameterStringVector AddNonWordTokens::paramNonWordLemmas(
        "non-word-lemmas", "non-word lemma symbols", ",");
const Core::ParameterBool AddNonWordTokens::paramAllStates(
        "all-states", "add loop transitions to all states", true);
const Core::ParameterBool AddNonWordTokens::paramInitialState(
        "initial-state", "add loop transitions to the initial state", false);
const Core::ParameterBool AddNonWordTokens::paramFinalState(
        "final-state", "add loop transitions to the final states", false);
const Core::ParameterBool AddNonWordTokens::paramUnigramState(
        "unigram-state", "add loop transitions to the unigram state", false);
const Core::ParameterBool AddNonWordTokens::paramRenormalize(
        "renormalize", "renormalize weights of modified states", false);

void AddNonWordTokens::addArcs(OpenFst::StateId s, f32 weight, const std::vector<OpenFst::Label>& labels) {
    for (std::vector<OpenFst::Label>::const_iterator l = labels.begin(); l != labels.end(); ++l) {
        input_->AddArc(s, OpenFst::Arc(*l, *l, OpenFst::Weight(weight), s));
    }
    if (renormalize_) {
        renormalizeWeights(s);
    }
}

void AddNonWordTokens::renormalizeWeights(OpenFst::StateId s) {
    typedef FstLib::LogWeight LogWeight;
    LogWeight                 sum = LogWeight::Zero();
    for (OpenFst::ArcIterator aiter(*input_, s); !aiter.Done(); aiter.Next())
        sum = FstLib::Plus(sum, LogWeight(aiter.Value().weight.Value()));
    for (OpenFst::MutableArcIterator aiter(input_, s); !aiter.Done(); aiter.Next()) {
        OpenFst::Arc arc = aiter.Value();
        arc.weight       = OpenFst::Weight(FstLib::Divide(LogWeight(arc.weight.Value()), sum).Value());
        aiter.SetValue(arc);
    }
}

OpenFst::Label AddNonWordTokens::getLabel(const Bliss::Lemma* lemma) const {
    Fsa::LabelId label = Fsa::InvalidLabelId;
    switch (outputType()) {
        case outputLemmas: {
            label = lemma->id();
        } break;
        case outputLemmaPronunciations: {
            verify(lemma->nPronunciations() == 1);
            Bliss::Lemma::LemmaPronunciationRange pron = lemma->pronunciations();
            label                                      = resources_.lexicon()->lemmaPronunciationAlphabet()->index(
                    pron.first);
        } break;
        case outputSyntacticTokens: {
            const Bliss::SyntacticTokenSequence& tokens = lemma->syntacticTokenSequence();
            if (tokens.isEpsilon()) {
                warning("adding empty syntactic token for %s", lemma->symbol().str());
                label = Fsa::Epsilon;
            }
            else {
                verify(tokens.length() == 1);
                label = (*tokens.begin())->id();
            }
        } break;
        default:
            defect();
    }
    return OpenFst::convertLabelFromFsa(label);
}

void AddNonWordTokens::getLabels(const std::vector<std::string>& lemmas,
                                 std::vector<OpenFst::Label>&    labels) const {
    Core::Ref<const Bliss::Lexicon>       lexicon = resources_.lexicon();
    Core::Ref<const Bliss::LemmaAlphabet> li      = lexicon->lemmaAlphabet();
    for (std::vector<std::string>::const_iterator symbol = lemmas.begin();
         symbol != lemmas.end(); ++symbol) {
        const Bliss::Lemma* lemma = lexicon->lemma(*symbol);
        if (!lemma) {
            criticalError("unknown lemma symbol: '%s'", symbol->c_str());
            return;
        }
        log("non word lemma '%s'", lemma->symbol().str());
        labels.push_back(getLabel(lemma));
    }
}

OpenFst::StateId AddNonWordTokens::getFinalState() const {
    bool             moreThanOne = false;
    OpenFst::StateId state       = OpenFst::findFinalState(*input_, &moreThanOne);
    if (moreThanOne)
        error("expected only one final state (sentence end)");
    if (state == OpenFst::InvalidStateId) {
        error("no final state found");
    }
    return state;
}

OpenFst::StateId AddNonWordTokens::getUnigramState() const {
    OpenFst::StateId state     = input_->Start();
    OpenFst::StateId prevState = state;
    while (input_->NumInputEpsilons(state)) {
        for (OpenFst::ArcIterator aiter(*input_, state); !aiter.Done(); aiter.Next()) {
            const OpenFst::Arc& arc = aiter.Value();
            if (arc.ilabel == OpenFst::Epsilon) {
                state = arc.nextstate;
                break;
            }
        }
        if (state == prevState) {
            error("epsilon loop found");
            return OpenFst::InvalidStateId;
        }
        prevState = state;
    }
    return state;
}

Operation::AutomatonRef AddNonWordTokens::process() {
    f32 weight = paramWeight(Operation::config);
    SleeveOperation::log("using weight: %f", weight);
    if (renormalize_)
        SleeveOperation::log("re-normalizing weights");
    std::vector<OpenFst::Label> labels;
    getLabels(paramNonWordLemmas(Operation::config), labels);
    if (paramAllStates(Operation::config)) {
        log("adding loop arcs to all states");
        for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
            addArcs(siter.Value(), weight, labels);
        }
    }
    else {
        std::set<OpenFst::StateId> silenceStates;
        if (paramInitialState(Operation::config)) {
            log("adding loop arcs to initial state: %d", input_->Start());
            addArcs(input_->Start(), weight, labels);
            silenceStates.insert(input_->Start());
        }
        if (paramUnigramState(Operation::config)) {
            OpenFst::StateId state = getUnigramState();
            log("adding loop arcs to unigram state: %d", state);
            addArcs(state, weight, labels);
            silenceStates.insert(state);
        }

        if (paramFinalState(Operation::config)) {
            u32 nFinal = 0;
            for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
                if (OpenFst::isFinalState(*input_, siter.Value()) &&
                    !silenceStates.count(siter.Value())) {
                    addArcs(siter.Value(), weight, labels);
                    ++nFinal;
                }
            }
            log("adding loop arcs to %d final states", nFinal);
        }
    }
    return input_;
}

Operation::AutomatonRef RemoveEmptyPath::process() {
    log("removing empty path");
    bool             moreThanOne = false;
    OpenFst::StateId final       = OpenFst::findFinalState(*input_, &moreThanOne);
    if (moreThanOne)
        error("expected only one final state");
    if (final == OpenFst::InvalidStateId)
        error("no final state found");
    OpenFst::StateId state        = input_->Start();
    OpenFst::StateId unigram      = OpenFst::InvalidStateId;
    OpenFst::StateId firstUnigram = input_->AddState();
    log("new state: %d", firstUnigram);

    // modify arcs of initial state:
    //  - remove epsilon transition to final state
    //  - redirect epsilon arc to new "first unigram state"
    std::vector<OpenFst::Arc> newArcs;
    newArcs.reserve(input_->NumArcs(state));
    OpenFst::Arc backoffArc;
    backoffArc.nextstate = OpenFst::InvalidStateId;
    verify(input_->NumInputEpsilons(state) <= 2);
    for (OpenFst::MutableArcIterator aiter(input_, state); !aiter.Done(); aiter.Next()) {
        const OpenFst::Arc& arc = aiter.Value();
        if (arc.nextstate != final) {
            if (arc.ilabel == OpenFst::Epsilon)
                backoffArc = arc;
            else
                newArcs.push_back(arc);
        }
        else {
            verify(arc.ilabel == OpenFst::Epsilon);
        }
    }
    input_->DeleteArcs(state);
    verify(backoffArc.nextstate != OpenFst::InvalidStateId);
    unigram = backoffArc.nextstate;
    log("unigram state: %d", unigram);
    backoffArc.nextstate = firstUnigram;
    input_->AddArc(state, backoffArc);
    OpenFst::addArcs(input_, state, newArcs);

    // add arcs from unigram state to "first unigram state"
    // except for the epsilon arc to the final state (sentence end)
    verify(input_->NumInputEpsilons(unigram) <= 1);
    for (OpenFst::MutableArcIterator aiter(input_, unigram); !aiter.Done(); aiter.Next()) {
        const OpenFst::Arc& arc = aiter.Value();
        if (arc.nextstate != final) {
            verify(arc.ilabel != OpenFst::Epsilon);
            input_->AddArc(firstUnigram, arc);
        }
    }
    return input_;
}

const Core::ParameterString CreateSubwordGrammar::paramSubwordList(
        "subword-list", "file with one subword token per line", "");
const Core::ParameterString CreateSubwordGrammar::paramTransitionSymbol(
        "transition-symbol", "symbol which activates the subword LM", "[UNKNOWN]");

bool CreateSubwordGrammar::readSubwordList(const std::string& filename) {
    Core::Ref<const Bliss::SyntacticTokenAlphabet> synt =
            resources_.lexicon()->syntacticTokenAlphabet();
    Core::CompressedInputStream cin(filename);
    if (!cin)
        return false;
    while (cin) {
        std::string symbol;
        cin >> symbol;
        if (symbol.empty())
            continue;
        Fsa::LabelId syntacticId = synt->index(symbol);
        if (syntacticId == Fsa::InvalidLabelId) {
            error("unknown symbol: '%s'", symbol.c_str());
            return false;
        }
        if (outputType() == outputSyntacticTokens)
            subwordTokens_.insert(syntacticId);
        else
            addLemma(syntacticId);
    }
    return true;
}

bool CreateSubwordGrammar::addLemma(Fsa::LabelId synt) {
    const Bliss::SyntacticToken* st = resources_.lexicon()->syntacticTokenAlphabet()->syntacticToken(synt);
    verify(st);
    Bliss::SyntacticToken::LemmaIterator lemma, lend;
    Core::tie(lemma, lend) = st->lemmas();
    for (; lemma != lend; ++lemma) {
        Fsa::LabelId lemmaId = (*lemma)->id();
        if (outputType() == outputLemmas)
            subwordTokens_.insert(lemmaId);
        else
            addLemmaPronunciation(lemmaId);
    }
    return true;
}

bool CreateSubwordGrammar::addLemmaPronunciation(Fsa::LabelId lemma) {
    verify(outputType() == outputLemmaPronunciations);
    Core::Ref<const Bliss::LemmaAlphabet> lemmas = resources_.lexicon()->lemmaAlphabet();
    const Bliss::Lemma*                   l      = lemmas->lemma(lemma);
    verify(l);
    Bliss::Lemma::PronunciationIterator pron, pend;
    Core::tie(pron, pend) = l->pronunciations();
    for (; pron != pend; ++pron) {
        subwordTokens_.insert(pron->id());
    }
    return true;
}

Operation::AutomatonRef CreateSubwordGrammar::process() {
    const std::string subwordFile = paramSubwordList(Operation::config);
    log("reading subword list: %s", subwordFile.c_str());
    if (!readSubwordList(subwordFile)) {
        error("cannot read subword list");
        return input_;
    }
    else {
        log("%zd subword tokens", subwordTokens_.size());
    }
    const OpenFst::SymbolTable* symbols = input_->OutputSymbols();
    if (!symbols) {
        error("symbol table is required");
        return input_;
    }
    OpenFst::StateId oldStart = input_->Start();
    OpenFst::StateId newStart = input_->AddState();
    input_->SetStart(newStart);
    input_->SetFinal(newStart, OpenFst::Weight::One());
    Core::Ref<const Fsa::Alphabet> alphabet;
    switch (outputType()) {
        case outputSyntacticTokens:
            alphabet = resources_.lexicon()->syntacticTokenAlphabet();
            break;
        case outputLemmas:
            alphabet = resources_.lexicon()->lemmaAlphabet();
            break;
        case outputLemmaPronunciations:
            alphabet = resources_.lexicon()->lemmaPronunciationAlphabet();
            break;
    }
    const std::string transitionSymbol = paramTransitionSymbol(Operation::config);
    Fsa::LabelId      transitionLabel  = alphabet->index(transitionSymbol);
    if (transitionLabel == Fsa::InvalidLabelId) {
        error("unknown transition symbol '%s'", transitionSymbol.c_str());
        return input_;
    }

    for (Fsa::Alphabet::const_iterator t = alphabet->begin(); t != alphabet->end(); ++t) {
        Fsa::LabelId      id        = t;
        const std::string symbol    = *t;
        const bool        isSubword = subwordTokens_.count(id);
        if (!isSubword && !symbol.empty() && id != transitionLabel) {
            OpenFst::Label label = OpenFst::convertLabelFromFsa(id);
            if (input_->InputSymbols() && input_->InputSymbols()->Find(label).empty())
                warning("symbol not in symbol table: %s %d", symbol.c_str(), label);
            input_->AddArc(newStart, OpenFst::Arc(label, label, OpenFst::Weight::One(), newStart));
        }
    }
    for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
        OpenFst::StateId state = siter.Value();
        if (state == newStart)
            continue;
        if (OpenFst::isFinalState(*input_, state)) {
            input_->AddArc(state, OpenFst::Arc(OpenFst::Epsilon, OpenFst::Epsilon, OpenFst::Weight::One(), newStart));
            input_->SetFinal(state, OpenFst::Weight::Zero());
        }
        for (OpenFst::MutableArcIterator aiter(input_, state); !aiter.Done(); aiter.Next()) {
            OpenFst::Arc arc = aiter.Value();
            arc.ilabel       = OpenFst::Epsilon;
            aiter.SetValue(arc);
        }
    }

    OpenFst::Label l = OpenFst::convertLabelFromFsa(transitionLabel);
    log("using transition label '%s' %d", transitionSymbol.c_str(), l);
    input_->AddArc(newStart, OpenFst::Arc(l, OpenFst::Epsilon, OpenFst::Weight::One(), oldStart));
    return input_;
}

Operation::AutomatonRef ContextBuilder::process() {
    const int initialPhoneOffset =
            input_->getIntAttribute(BuildLexicon::attrInitialPhoneOffset);
    verify(initialPhoneOffset != Automaton::InvalidIntAttribute);
    const int disambiguatorOffset =
            input_->getIntAttribute(BuildLexicon::attrDisambiguatorOffset);
    verify(disambiguatorOffset != Automaton::InvalidIntAttribute);
    const int nDisambiguators =
            input_->getIntAttribute(Automaton::attrNumDisambiguators);
    verify(nDisambiguators != Automaton::InvalidIntAttribute);

    log("using inital phone offset %d", initialPhoneOffset);
    log("using disambiguator offset %d", disambiguatorOffset);
    log("using %d disambiguators", nDisambiguators);

    ContextTransducerBuilder tb(select("context-builder"),
                                resources_.acousticModel(), resources_.lexicon());
    tb.setDisambiguators(nDisambiguators, disambiguatorOffset);
    tb.setInitialPhoneOffset(initialPhoneOffset);
    tb.setWordDisambiguators(disambiguatorOffset);
    if (input_->InputSymbols())
        tb.setPhoneSymbols(input_->InputSymbols());
    AutomatonRef result = new Automaton();
    FstLib::Cast(*tb.build(), result);
    FstLib::ArcSort(result, FstLib::StdOLabelCompare());
    input_->copyAttribute(result, Automaton::attrNumDisambiguators);
    result->setAttribute(BuildLexicon::attrWordLabelOffset, tb.getWordLabelOffset());
    log("word label offset: %d", tb.getWordLabelOffset());
    result->setAttribute(BuildLexicon::attrDisambiguatorOffset, tb.getDisambiguatorOffset());
    return result;
}

Operation::AutomatonRef HmmBuilder::process() {
    log("using %d disambiguators", nDisambiguators_);
    Core::Ref<Am::TransducerBuilder> tb = resources_.acousticModel()->createTransducerBuilder();
    tb->setDisambiguators(nDisambiguators_);
    tb->selectAllophonesFromLexicon();
    tb->selectAllophonesAsInput();
    tb->selectFlatModel();
    Fsa::ConstAutomatonRef h = tb->createEmissionLoopTransducer(true);
    Fsa::sort(h, Fsa::SortTypeByOutput);
    AutomatonRef result = OpenFst::convertFromFsa<Fsa::Automaton, Automaton>(h);
    result->setAttribute(Automaton::attrNumDisambiguators, nDisambiguators_);
    return result;
}

const Core::ParameterString CreateStateSequences::paramFilename(
        "filename",
        "file name of the state sequences of the search network", "");

bool CreateStateSequences::precondition() const {
    if (filename_.empty()) {
        warning("no file name for state sequences");
        return false;
    }
    return SleeveOperation::precondition();
}

Operation::AutomatonRef CreateStateSequences::process() {
    log("creating state sequencs");
    StateSequenceBuilder builder(select("state-sequences"), resources_.acousticModel(), resources_.lexicon());
    u32                  nDisambiguators = input_->getIntAttribute(Automaton::attrNumDisambiguators);
    verify(nDisambiguators != Automaton::InvalidIntAttribute);
    log("using %d disambiguators", nDisambiguators);
    builder.setNumDisambiguators(nDisambiguators);
    builder.build();
    StateSequenceList* states = builder.createStateSequenceList();
    log("number of states sequences: %d", static_cast<int>(states->size()));
    log("writing state sequences to %s", filename_.c_str());
    states->write(filename_);
    builder.relabelTransducer(input_);
    delete states;
    return input_;
}

const Core::ParameterBool NonWordDependentOperation::paramAddNonWords(
        "add-non-words", "consider non-word state sequences", false);

u32 NonWordDependentOperation::numSpecialSymbols() const {
    if (!addNonWords_) {
        return 0;
    }
    else {
        NonWordTokens nonWordTokens(select("non-word-tokens"), *resources_.lexicon());
        nonWordTokens.init();
        u32 nNonWordModels = nonWordTokens.phones().size();
        log("assuming last %d state sequences are non-word models", nNonWordModels);
        return nNonWordModels;
    }
}

const Core::ParameterString Factorize::paramStateSequences(
        "state-sequences", "state sequences file", "");
const Core::ParameterString Factorize::paramNewStateSequences(
        "new-state-sequences", "new state sequences file", "");

Operation::AutomatonRef Factorize::process() {
    std::string stateSequencesFile = paramStateSequences(config);
    log("loading state sequences: %s", stateSequencesFile.c_str());
    StateSequenceList ssl;
    if (!ssl.read(stateSequencesFile))
        criticalError("cannot read state sequences");
    log("# state sequences: %d", u32(ssl.size()));
    u32                  nSpecialTokens     = numSpecialSymbols();
    u32                  specialTokenOffset = ssl.size() - nSpecialTokens;
    TiedStateSequenceMap newLabels;

    OpenFst::InDegree<OpenFst::Arc> inDegree(*input_);
    std::stack<OpenFst::StateId>    queue;
    queue.push(input_->Start());
    std::vector<bool> visited(input_->NumStates(), false);
    visited[queue.top()] = true;
    while (!queue.empty()) {
        OpenFst::StateId s = queue.top();
        queue.pop();
        for (OpenFst::MutableArcIterator ai(input_, s); !ai.Done(); ai.Next()) {
            OpenFst::Arc  arc = ai.Value();
            StateSequence seq;
            s32           seqIndex = -1;
            if (arc.ilabel != OpenFst::Epsilon) {
                seqIndex = OpenFst::convertLabelToFsa(arc.ilabel);
                seq      = ssl[seqIndex];
            }
            OpenFst::StateId ns        = arc.nextstate;
            OpenFst::Label   output    = arc.olabel;
            OpenFst::Label   newOutput = output;
            OpenFst::Weight  weight    = arc.weight;
            while (inDegree[ns] == 1 && !seq.isFinal() && seqIndex < specialTokenOffset) {
                if (OpenFst::isFinalState(*input_, ns))
                    break;
                if (input_->NumArcs(ns) != 1)
                    break;
                const OpenFst::Arc nextArc = OpenFst::ArcIterator(*input_, ns).Value();
                if (nextArc.olabel != OpenFst::Epsilon) {
                    if (newOutput != OpenFst::Epsilon)
                        break;
                    else
                        newOutput = nextArc.olabel;
                }

                if (nextArc.ilabel != OpenFst::Epsilon) {
                    u32 nextSeqIndex = OpenFst::convertLabelToFsa(nextArc.ilabel);
                    if (nextSeqIndex >= specialTokenOffset)
                        break;
                    StateSequence nextSeq = ssl[nextSeqIndex];
                    if (nextSeq.isFinal())
                        seq.setFinal();
                    if (nextSeq.isInitial()) {
                        seq.setInitial();
                        verify_eq(seq.nStates(), 0);
                    }
                    for (u32 si = 0; si < nextSeq.nStates(); ++si)
                        seq.appendState(nextSeq.state(si).emission_, nextSeq.state(si).transition_);
                }
                output = newOutput;
                weight = FstLib::Times(weight, nextArc.weight);
                ns     = nextArc.nextstate;
            }
            arc.nextstate = ns;
            if (seq.nStates()) {
                if (seqIndex < specialTokenOffset)
                    arc.ilabel = OpenFst::convertLabelFromFsa(newLabels.index(seq));
                else
                    arc.ilabel = -1 - (seqIndex - specialTokenOffset);
            }
            arc.olabel = output;
            ai.SetValue(arc);
            if (!visited[ns]) {
                visited[ns] = true;
                queue.push(ns);
            }
        }
    }
    FstLib::Connect(input_);

    StateSequenceList newList;
    newLabels.createStateSequenceList(newList);
    OpenFst::LabelMapping labelMapping;
    if (addNonWords_) {
        // add special sequences to the end of the list
        s32 l = -1;
        for (u32 i = specialTokenOffset; i < ssl.size(); ++i, --l) {
            newList.push_back(ssl[i]);
            labelMapping.push_back(std::make_pair(l, newList.size()));
        }
    }
    if (!labelMapping.empty())
        FstLib::Relabel(input_, labelMapping, OpenFst::LabelMapping());
    stateSequencesFile = paramNewStateSequences(config);
    log("writing %d state sequences: %s", u32(newList.size()), stateSequencesFile.c_str());
    newList.write(stateSequencesFile);
    return input_;
}

const Core::ParameterString ExpandStates::paramStateSequences(
        "state-sequences", "state sequences file", "");
const Core::ParameterString ExpandStates::paramNewStateSequences(
        "new-state-sequences", "new state sequences file", "");

Operation::AutomatonRef ExpandStates::process() {
    std::string stateSequences = paramStateSequences(config);
    log("loading state sequences: %s", stateSequences.c_str());
    StateSequenceList ssl;
    if (!ssl.read(stateSequences))
        criticalError("cannot read state sequences");
    u32                        nSpecialTokens     = numSpecialSymbols();
    u32                        specialTokenOffset = ssl.size() - nSpecialTokens;
    StateSequenceList          newSsl;
    TiedStateSequenceMap       sequences;
    std::vector<StateSequence> specialSequences;
    OpenFst::StateId           nStates = input_->NumStates();
    for (OpenFst::StateId state = 0; state < nStates; ++state) {
        for (OpenFst::MutableArcIterator aiter(input_, state); !aiter.Done(); aiter.Next()) {
            const OpenFst::Arc& arc = aiter.Value();
            if (arc.ilabel == OpenFst::Epsilon || StateSequenceBuilder::isDisambiguator(arc.ilabel))
                continue;
            s32                  seqIndex       = OpenFst::convertLabelToFsa(arc.ilabel);
            const StateSequence& ss             = ssl[seqIndex];
            bool                 isRegularLabel = (seqIndex < specialTokenOffset);
            OpenFst::Arc         newArc;
            if (ss.nStates() > 1) {
                expandArc(arc, ss, isRegularLabel, &sequences, &specialSequences, &newArc);
            }
            else {
                newArc = arc;
                if (isRegularLabel) {
                    newArc.ilabel = OpenFst::convertLabelFromFsa(sequences.index(ss));
                }
                else {
                    newArc.ilabel = -1 - specialSequences.size();
                    specialSequences.push_back(ss);
                }
            }
            aiter.SetValue(newArc);
        }
    }
    stateSequences = paramNewStateSequences(config);
    StateSequenceList newList;
    sequences.createStateSequenceList(newList);
    OpenFst::LabelMapping labelMapping;
    if (addNonWords_) {
        // add special sequences to the end of the list
        s32 l = -1;
        for (u32 i = 0; i < specialSequences.size(); ++i, --l) {
            newList.push_back(specialSequences[i]);
            labelMapping.push_back(std::make_pair(l, newList.size()));
        }
    }
    if (!labelMapping.empty())
        FstLib::Relabel(input_, labelMapping, OpenFst::LabelMapping());

    log("writing %zd state sequences: %s", newList.size(), stateSequences.c_str());
    newList.write(stateSequences);
    return input_;
}

void ExpandStates::expandArc(const OpenFst::Arc& arc, const StateSequence& ss,
                             bool                        isRegularLabel,
                             TiedStateSequenceMap*       sequences,
                             std::vector<StateSequence>* specialSequences,
                             OpenFst::Arc*               firstArc) {
    OpenFst::StateId prevState = OpenFst::InvalidStateId;
    for (u32 s = 0; s < ss.nStates(); ++s) {
        OpenFst::Arc newArc;
        newArc.ilabel = OpenFst::InvalidLabelId;
        newArc.olabel = OpenFst::Epsilon;
        newArc.weight = OpenFst::Weight::One();
        StateSequence newss;
        newss.appendState(ss.state(s).emission_, ss.state(s).transition_);
        if (s == 0) {
            if (ss.isInitial())
                newss.addFlag(Am::Allophone::isInitialPhone);
            newArc = arc;
        }
        if (s == (ss.nStates() - 1)) {
            if (ss.isFinal())
                newss.addFlag(Am::Allophone::isFinalPhone);
            newArc.nextstate = arc.nextstate;
        }
        else {
            newArc.nextstate = input_->AddState();
        }
        if (isRegularLabel || s < (ss.nStates() - 1)) {
            newArc.ilabel = OpenFst::convertLabelFromFsa(sequences->index(newss));
        }
        else {
            newArc.ilabel = -1 - specialSequences->size();
            specialSequences->push_back(newss);
        }

        if (s == 0) {
            *firstArc = newArc;
        }
        else {
            input_->AddArc(prevState, newArc);
        }
        prevState = newArc.nextstate;
    }
}

const Core::ParameterString ConvertStateSequences::paramInput(
        "hmm-list", "hmm list file", "");
const Core::ParameterString ConvertStateSequences::paramOutput(
        "state-sequences", "state sequences filename", "");
const Core::ParameterString ConvertStateSequences::paramHmmSymbols(
        "hmm-symbols", "hmm symbol table", "");
const Core::ParameterString ConvertStateSequences::paramStateSymbols(
        "state-symbols", "hmm state symbol table", "");

bool ConvertStateSequences::precondition() const {
    return !(paramInput(config).empty() || paramOutput(config).empty(),
             paramHmmSymbols(config).empty() || paramStateSymbols(config).empty());
}

Operation::AutomatonRef ConvertStateSequences::process() {
    HmmListConverter      converter(config);
    OpenFst::SymbolTable* hmmSymbols = OpenFst::SymbolTable::ReadText(paramHmmSymbols(config));
    verify(hmmSymbols);
    log("read %d hmm symbols", static_cast<u32>(hmmSymbols->NumSymbols()));
    OpenFst::SymbolTable* stateSymbols = OpenFst::SymbolTable::ReadText(paramStateSymbols(config));
    verify(stateSymbols);
    log("read %d hmm state symbols", static_cast<u32>(stateSymbols->NumSymbols()));
    converter.setHmmSymbols(hmmSymbols);
    converter.setHmmStateSymbols(stateSymbols);
    const std::string hmmList = paramInput(config);
    log("converting hmm list %s", hmmList.c_str());
    StateSequenceList* states = converter.creatStateSequenceList(hmmList);
    verify(states);
    const std::string output = paramOutput(config);
    log("writing state sequences to %s", output.c_str());
    states->write(output);
    Core::Channel dumpChannel(config, "dump");
    if (dumpChannel.isOpen()) {
        states->dump(resources_.acousticModel(), resources_.lexicon(), dumpChannel);
    }
    delete states;
    return AutomatonRef(0);
}

const Core::ParameterString BuildStateTree::paramStateSequencesFile(
        "state-sequences", "state sequences filename", "");

Operation::AutomatonRef BuildStateTree::process() {
    StateTreeConverter tree(config, resources_.lexicon(), resources_.acousticModel());
    AutomatonRef       result = new Automaton();
    tree.createFst(result);
    if (!tree.writeStateSequences(paramStateSequencesFile(config)))
        error("failed to write state sequences to %s", paramStateSequencesFile(config).c_str());
    return result;
}
