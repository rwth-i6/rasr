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
#include <Core/Debug.hh>
#include <OpenFst/Count.hh>
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/StateSequence.hh>
#include <Search/Wfst/StateTree.hh>
#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/connect.h>
#include <fst/determinize.h>
#include <fst/encode.h>
#include <fst/rmepsilon.h>

using namespace Search::Wfst;

const Core::Choice StateTreeConverter::choiceCompression(
        "none", CompressionNone,
        "factorized", CompressionFactorized,
        "hmm-label", CompressionHmmLabel,
        Core::Choice::endMark());

const Core::ParameterChoice StateTreeConverter::paramCompression(
        "compression", &choiceCompression, "compression method", CompressionNone);

const Core::ParameterBool StateTreeConverter::paramEpsilonArcs(
        "epsilon-arcs", "add input epsilon arcs for word ends arcs", false);

const Core::ParameterBool StateTreeConverter::paramMergeNonTreeArcs(
        "merge-non-tree-arcs", "add epsilon arcs before states with in-degree > 1", false);

const Core::ParameterBool StateTreeConverter::paramAddDisambiguators(
        "add-disambiguators", "add disambiguator symbols", false);

const Core::ParameterBool StateTreeConverter::paramPushWordLabels(
        "push-word-labels", "combine equal word labels before fan-out", false);

StateTreeConverter::StateTreeConverter(
        const Core::Configuration&         c,
        Bliss::LexiconRef                  lexicon,
        Core::Ref<const Am::AcousticModel> am)
        : Core::Component(c),
          lexicon_(lexicon),
          am_(am),
          factorize_(false),
          hmmLabels_(false),
          wordEndEpsArcs_(paramEpsilonArcs(config)),
          mergeNonTreeArcs_(paramMergeNonTreeArcs(config)),
          addDisambiguators_(paramAddDisambiguators(config)),
          pushWordLabels_(paramPushWordLabels(config)),
          numDisambiguators_(0),
          labels_(0),
          stateSequences_(0),
          stateTree_(0) {
    switch (static_cast<CompressionType>(paramCompression(config))) {
        case CompressionFactorized:
            log("building factorized state tree");
            factorize_ = true;
            labels_    = new TiedStateSequenceMap();
            break;
        case CompressionHmmLabel: {
            log("using standard hmm labels");
            hmmLabels_ = true;
            StateSequenceBuilder seqBuilder(select("states-sequences"), am_, lexicon_);
            seqBuilder.build();
            stateSequences_ = seqBuilder.createStateSequenceList();
            break;
        }
        default:
            log("not using state tree compression");
            labels_ = new TiedStateSequenceMap();
            break;
    }
    if (wordEndEpsArcs_)
        log("using word end epsilon arcs");
    if (mergeNonTreeArcs_)
        log("merging non-tree arcs");
    if (addDisambiguators_)
        log("adding disambiguator symbols");
    if (pushWordLabels_)
        log("pushing word end labels");
}

StateTreeConverter::~StateTreeConverter() {
    delete labels_;
    delete stateSequences_;
    delete stateTree_;
}

OpenFst::StateId StateTreeConverter::getState(OpenFst::VectorFst* fst, StateTree::StateId s, bool final, bool* isNew) {
    StateMap::const_iterator iter = stateMap_.find(s);
    if (iter == stateMap_.end()) {
        OpenFst::StateId fs = stateMap_[s] = fst->AddState();
        if (final)
            fst->SetFinal(fs, OpenFst::Weight::One());
        *isNew = true;
        return fs;
    }
    else {
        *isNew = false;
        return iter->second;
    }
}

OpenFst::Label StateTreeConverter::getLabel(StateTree::StateId s, bool initial, bool final) {
    Am::AcousticModel::EmissionIndex        e = stateTree_->stateDesc(s).acousticModel;
    Am::AcousticModel::StateTransitionIndex t = stateTree_->stateDesc(s).transitionModelIndex;
    verify(e != StateTree::invalidAcousticModel);
    if (!hmmLabels_) {
        require(labels_);
        StateSequence seq;
        seq.appendState(e, t);
        u8 flags = 0;
        if (final)
            flags |= Am::Allophone::isFinalPhone;
        if (initial)
            flags |= Am::Allophone::isInitialPhone;
        seq.setFlags(flags);
        return OpenFst::convertLabelFromFsa(labels_->index(seq));
    }
    else {
        return encodeHmmState(e, t, initial, final);
    }
}

OpenFst::Label StateTreeConverter::getOutputLabel(const StateTree::Exit& exit) const {
    return OpenFst::convertLabelFromFsa(exit.pronunciation->id());
}

void StateTreeConverter::createFst(OpenFst::VectorFst* fst) {
    stateTree_ = new StateTree(select("state-tree"), lexicon_, am_);

    std::deque<StateTree::StateId> queue;
    std::vector<bool>              visited(stateTree_->nStates(), false);
    StateTree::StateId             s          = stateTree_->root();
    bool                           isNewState = false;
    OpenFst::StateId               initial    = getState(fst, s, true, &isNewState);
    getState(fst, stateTree_->ciRoot(), true, &isNewState);
    visited[s] = true;
    fst->SetStart(initial);
    queue.push_front(s);
    while (!queue.empty()) {
        s = queue.front();
        queue.pop_front();
        OpenFst::StateId fstState = getState(fst, s, false, &isNewState);
        DBG(1) << VAR(s) << " " << VAR(fstState) << ENDDBG;
        StateTree::SuccessorIterator nextState, nextStateEnd;
        for (Core::tie(nextState, nextStateEnd) = stateTree_->successors(s); nextState != nextStateEnd; ++nextState) {
            const bool       isInitial    = (stateTree_->stateDepth(s) == 0);
            OpenFst::Label   l            = getLabel(nextState, isInitial, false);
            OpenFst::StateId fstNextState = getState(fst, nextState, false, &isNewState);
            if (!isNewState)
                nonTreeStates_.insert(fstNextState);
            DBG(1) << VAR(nextState) << " " << VAR(fstNextState) << " " << VAR(l) << " " << VAR(isInitial) << VAR(stateTree_->stateDepth(s)) << ENDDBG;
            DBG(1) << "flags: initial=" << labels_->get(OpenFst::convertLabelToFsa(l)).isInitial()
                   << " final=" << labels_->get(OpenFst::convertLabelToFsa(l)).isFinal() << ENDDBG;
            fst->AddArc(fstState, OpenFst::Arc(l, OpenFst::Epsilon, OpenFst::Weight::One(), fstNextState));
            if (!visited[nextState]) {
                visited[nextState] = true;
                queue.push_front(nextState);
            }
            if (!wordEndEpsArcs_)
                addWordEndArcs(fst, nextState, isInitial, fstState, &visited, &queue);
        }
        if (wordEndEpsArcs_)
            addWordEndEpsilonArcs(fst, s, fstState, &visited, &queue);
    }
    if (mergeNonTreeArcs_)
        mergeArcs(fst);

    delete stateTree_;
    stateTree_ = 0;
    if (addDisambiguators_) {
        log("disambiguators: %d", numDisambiguators_);
    }
    if (!wordEndEpsArcs_) {
        FstLib::Connect(fst);
        nonTreeStates_.clear();
    }

    statistics(fst, "before-compression");
    if (factorize_) {
        factorize(fst);
    }
    else if (hmmLabels_) {
        convertToHmmLabels(fst);
    }
    if (!hmmLabels_) {
        verify(!stateSequences_);
        stateSequences_ = new StateSequenceList();
        labels_->createStateSequenceList(*stateSequences_);
    }
    statistics(fst, "after-compression");
    OpenFst::SymbolTable* osymbols = OpenFst::convertAlphabet(lexicon_->lemmaPronunciationAlphabet(), "output");
    fst->SetOutputSymbols(osymbols);
}

void StateTreeConverter::wordEndLabels(StateTree::StateId s, WordEndMap* labels) const {
    StateTree::const_ExitIterator we, weEnd;
    Core::tie(we, weEnd) = stateTree_->wordEnds(s);
    for (; we != weEnd; ++we) {
        WordEndMap::const_iterator weId = labels->find(we->pronunciation);
        if (weId == labels->end()) {
            u32 wordEndId = labels->size();
            labels->insert(WordEndMap::value_type(we->pronunciation, wordEndId));
        }
    }
}

void StateTreeConverter::addWordEndArcs(OpenFst::VectorFst* fst, StateTree::SuccessorIterator nextState, bool isInitial, OpenFst::StateId fstState,
                                        std::vector<bool>* visited, std::deque<StateTree::StateId>* queue) {
    WordEndMap distinctWordEnds;
    wordEndLabels(nextState, &distinctWordEnds);

    std::vector<OpenFst::StateId> wordEndStates;
    if (pushWordLabels_ && distinctWordEnds.size() > 1) {
        for (u32 w = 0; w < distinctWordEnds.size(); ++w)
            wordEndStates.push_back(fst->AddState());
    }

    StateTree::const_ExitIterator weBegin, weEnd;
    Core::tie(weBegin, weEnd) = stateTree_->wordEnds(nextState);
    u32 numWordEnds           = std::distance(weBegin, weEnd);

    for (StateTree::const_ExitIterator we = weBegin; we != weEnd; ++we) {
        bool                       isNewState = false;
        u32                        wordEndId  = 0;
        WordEndMap::const_iterator weId       = distinctWordEnds.find(we->pronunciation);
        wordEndId                             = distinctWordEnds.find(we->pronunciation)->second;

        OpenFst::StateId rootState = getState(fst, we->transitEntry, false, &isNewState);
        if (!isNewState)
            nonTreeStates_.insert(rootState);
        verify(stateTree_->stateDesc(we->transitEntry).acousticModel == StateTree::invalidAcousticModel);
        OpenFst::Label input  = getLabel(nextState, isInitial, true);
        OpenFst::Label output = getOutputLabel(*we);
        DBG(1) << VAR(we->transitEntry) << " " << VAR(rootState) << " " << VAR(input) << " "
               << lexicon_->lemmaPronunciationAlphabet()->symbol(we->pronunciation->id()) << ENDDBG;
        DBG(1) << "flags: initial=" << labels_->get(OpenFst::convertLabelToFsa(input)).isInitial()
               << " final=" << labels_->get(OpenFst::convertLabelToFsa(input)).isFinal() << ENDDBG;
        OpenFst::StateId nextState = rootState;
        OpenFst::StateId prevState = fstState;
        if (!wordEndStates.empty()) {
            OpenFst::Label l = OpenFst::Epsilon;
            if (addDisambiguators_)
                l = OpenFst::convertLabelFromFsa(AllophoneToAlloponeStateSequenceMap::getDisambiguator(wordEndId));
            fst->AddArc(prevState, OpenFst::Arc(l, output, OpenFst::Weight::One(), wordEndStates[wordEndId]));
            output    = OpenFst::Epsilon;
            prevState = wordEndStates[wordEndId];
        }
        else if (addDisambiguators_ && numWordEnds > 1) {
            nextState        = fst->AddState();
            OpenFst::Label l = OpenFst::convertLabelFromFsa(AllophoneToAlloponeStateSequenceMap::getDisambiguator(wordEndId));
            fst->AddArc(nextState, OpenFst::Arc(l, OpenFst::Epsilon, OpenFst::Weight::One(), rootState));
        }
        fst->AddArc(prevState, OpenFst::Arc(input, output, OpenFst::Weight::One(), nextState));
        if (!visited->at(we->transitEntry)) {
            visited->at(we->transitEntry) = true;
            queue->push_back(we->transitEntry);
        }
    }
    if (addDisambiguators_) {
        numDisambiguators_ = std::max(u32(distinctWordEnds.size()), numDisambiguators_);
    }
}

void StateTreeConverter::addWordEndEpsilonArcs(OpenFst::VectorFst* fst, StateTree::StateId state, OpenFst::StateId fstState,
                                               std::vector<bool>* visited, std::deque<StateTree::StateId>* queue) {
    WordEndMap distinctWordEnds;
    wordEndLabels(state, &distinctWordEnds);
    if (pushWordLabels_ && distinctWordEnds.size() > 1) {
        OpenFst::StateId wordLabelState = fst->AddState();
        for (WordEndMap::const_iterator w = distinctWordEnds.begin(); w != distinctWordEnds.end(); ++w) {
            OpenFst::Label input  = OpenFst::Epsilon;
            OpenFst::Label output = OpenFst::convertLabelFromFsa(w->first->id());
            if (addDisambiguators_)
                input = OpenFst::convertLabelFromFsa(AllophoneToAlloponeStateSequenceMap::getDisambiguator(w->second));
            fst->AddArc(fstState, OpenFst::Arc(input, output, OpenFst::Weight::One(), wordLabelState));
        }
        fstState = wordLabelState;
    }

    StateTree::const_ExitIterator we, we_end;
    for (Core::tie(we, we_end) = stateTree_->wordEnds(state); we != we_end; ++we) {
        bool             isNewState = false;
        u32              wordEndId  = distinctWordEnds.find(we->pronunciation)->second;
        OpenFst::StateId rootState  = getState(fst, we->transitEntry, false, &isNewState);
        if (!isNewState)
            nonTreeStates_.insert(rootState);
        verify(stateTree_->stateDesc(we->transitEntry).acousticModel == StateTree::invalidAcousticModel);
        OpenFst::Label   output    = pushWordLabels_ ? OpenFst::Epsilon : getOutputLabel(*we);
        OpenFst::Label   input     = OpenFst::Epsilon;
        OpenFst::StateId prevState = fstState;
        if (addDisambiguators_ && !pushWordLabels_)
            input = OpenFst::convertLabelFromFsa(AllophoneToAlloponeStateSequenceMap::getDisambiguator(wordEndId));
        fst->AddArc(prevState, OpenFst::Arc(input, output, OpenFst::Weight::One(), rootState));

        if (!visited->at(we->transitEntry)) {
            visited->at(we->transitEntry) = true;
            queue->push_back(we->transitEntry);
        }
    }
    if (addDisambiguators_) {
        numDisambiguators_ = std::max(u32(distinctWordEnds.size()), numDisambiguators_);
    }
}

void StateTreeConverter::mergeArcs(OpenFst::VectorFst* fst) const {
    typedef std::unordered_map<OpenFst::StateId, OpenFst::StateId> StateMap;
    StateMap                                                       mergeStates;
    OpenFst::StateId                                               nStates = fst->NumStates();
    for (OpenFst::StateId s = 0; s < nStates; ++s) {
        for (OpenFst::MutableArcIterator aiter(fst, s); !aiter.Done(); aiter.Next()) {
            OpenFst::Arc arc = aiter.Value();
            if (nonTreeStates_.count(arc.nextstate)) {
                OpenFst::StateId         ns = OpenFst::InvalidStateId;
                StateMap::const_iterator i  = mergeStates.find(arc.nextstate);
                if (i == mergeStates.end()) {
                    ns = fst->AddState();
                    mergeStates.insert(StateMap::value_type(arc.nextstate, ns));
                    fst->AddArc(ns, OpenFst::Arc(arc.ilabel, OpenFst::Epsilon, OpenFst::Weight::One(), arc.nextstate));
                }
                else {
                    ns = i->second;
                }
                arc.ilabel    = OpenFst::Epsilon;
                arc.nextstate = ns;
                aiter.SetValue(arc);
            }
        }
    }
}

OpenFst::Label StateTreeConverter::encodeHmmState(Am::AcousticModel::EmissionIndex        emission,
                                                  Am::AcousticModel::StateTransitionIndex transition,
                                                  bool isInitial, bool isFinal) const {
    const u8       tm    = transition;
    const u16      em    = emission;
    OpenFst::Label label = em << 10;
    label |= static_cast<OpenFst::Label>(tm) << 2;
    label |= static_cast<OpenFst::Label>(isInitial) << 1;
    label |= static_cast<OpenFst::Label>(isFinal);
    return label + 1;
}

void StateTreeConverter::convertToHmmLabels(OpenFst::VectorFst* fst) {
    OpenFst::VectorFst* s2e = createStateSequenceToEmissionTransducer();
    FstLib::ArcSort(fst, FstLib::ILabelCompare<OpenFst::Arc>());
    OpenFst::VectorFst result;
    FstLib::Compose(*s2e, *fst, &result);
    log("composed");
    FstLib::RmEpsilon(&result, true);
    log("epsilon removed");
    delete s2e;
    *fst = result;
}

OpenFst::VectorFst* StateTreeConverter::createStateSequenceToEmissionTransducer() const {
    require(stateSequences_);
    OpenFst::VectorFst *result = new OpenFst::VectorFst(), *detResult = new OpenFst::VectorFst();
    OpenFst::StateId    initial = result->AddState();
    result->SetStart(initial);
    result->SetFinal(initial, OpenFst::Weight::One());
    for (u32 seqId = 0; seqId < stateSequences_->size(); ++seqId) {
        const StateSequence& seq = (*stateSequences_)[seqId];
        OpenFst::StateId     s   = initial;
        for (u32 state = 0; state < seq.nStates(); ++state) {
            const bool       isLastState = (state == seq.nStates() - 1);
            OpenFst::StateId ns;
            OpenFst::Label   input     = OpenFst::Epsilon;
            const bool       isInitial = (state == 0 && seq.isInitial());
            const bool       isFinal   = (isLastState && seq.isFinal());
            OpenFst::Label   output    = encodeHmmState(seq.state(state).emission_, seq.state(state).transition_,
                                                        isInitial, isFinal);
            if (isLastState) {
                ns    = initial;
                input = OpenFst::convertLabelFromFsa(seqId);
            }
            else {
                ns = result->AddState();
            }
            result->AddArc(s, OpenFst::Arc(input, output, OpenFst::Weight::One(), ns));
            s = ns;
        }
    }
    Fsa::AutomatonCounts cnt = OpenFst::count(*result);
    log("before det: %d states, %d arcs", cnt.nStates_, (int)cnt.nArcs_);
    FstLib::EncodeMapper<OpenFst::Arc> encoder(FstLib::kEncodeLabels, FstLib::ENCODE);
    FstLib::Encode(result, &encoder);
    FstLib::Determinize(*result, detResult);
    delete result;
    FstLib::Decode(detResult, encoder);
    FstLib::ArcSort(detResult, FstLib::OLabelCompare<OpenFst::Arc>());
    cnt = OpenFst::count(*detResult);
    log("after det: %d states, %d arcs", cnt.nStates_, (int)cnt.nArcs_);
    return detResult;
}

void StateTreeConverter::factorize(OpenFst::VectorFst* fst) {
    log("factorizing state tree transducer");
    OpenFst::InDegree<OpenFst::Arc> inDegree(*fst);
    TiedStateSequenceMap*           newLabels = new TiedStateSequenceMap();
    std::stack<OpenFst::StateId>    queue;
    queue.push(fst->Start());
    std::vector<bool> visited(fst->NumStates(), false);
    visited[queue.top()] = true;
    while (!queue.empty()) {
        OpenFst::StateId s = queue.top();
        queue.pop();
        DBG(1) << VAR(s) << ENDDBG;
        for (OpenFst::MutableArcIterator ai(fst, s); !ai.Done(); ai.Next()) {
            OpenFst::Arc  arc = ai.Value();
            StateSequence seq = labels_->get(OpenFst::convertLabelToFsa(arc.ilabel));
            if (arc.olabel)
                verify(seq.isFinal());
            OpenFst::StateId ns = arc.nextstate;
            DBG(1) << VAR(arc.nextstate) << " " << VAR(seq.isFinal()) << " " << VAR(seq.isInitial()) << ENDDBG;
            OpenFst::Label output = arc.olabel;
            while (inDegree[ns] == 1) {
                if (OpenFst::isFinalState(*fst, ns))
                    break;
                if (fst->NumArcs(ns) != 1)
                    break;
                const OpenFst::Arc nextArc = OpenFst::ArcIterator(*fst, ns).Value();
                StateSequence      nextSeq = labels_->get(OpenFst::convertLabelToFsa(nextArc.ilabel));
                DBG(1) << VAR(inDegree[ns]) << " " << VAR(fst->NumArcs(ns)) << " " << VAR(nextArc.nextstate) << " " << VAR(nextSeq.isFinal()) << " " << VAR(nextSeq.isInitial()) << ENDDBG;
                seq.addFlag(nextSeq.flags());
                verify(nextSeq.nStates() == 1);
                seq.appendState(nextSeq.state(0).emission_, nextSeq.state(0).transition_);
                if (nextArc.olabel != OpenFst::Epsilon) {
                    verify(output == OpenFst::Epsilon);
                    output = nextArc.olabel;
                }
                ns = nextArc.nextstate;
                if (nextSeq.isFinal()) {
                    verify(output);
                    break;
                }
            }
            arc.nextstate = ns;
            DBG(1) << VAR(ns) << " " << VAR(seq.nStates()) << " " << VAR(output) << " " << VAR(seq.isFinal()) << " " << VAR(seq.isInitial()) << ENDDBG;
            arc.ilabel = OpenFst::convertLabelFromFsa(newLabels->index(seq));
            arc.olabel = output;
            if (arc.olabel)
                verify(seq.isFinal());
            ai.SetValue(arc);
            if (!visited[ns]) {
                visited[ns] = true;
                queue.push(ns);
            }
        }
    }
    FstLib::Connect(fst);
    delete labels_;
    labels_ = newLabels;
}

bool StateTreeConverter::writeStateSequences(const std::string& filename) const {
    require(stateSequences_);
    Core::Channel dumpChannel(config, "dump");
    if (dumpChannel.isOpen())
        stateSequences_->dump(am_, lexicon_, dumpChannel);
    return stateSequences_->write(filename);
}

void StateTreeConverter::statistics(const OpenFst::VectorFst* fst, const std::string& description) const {
    require(stateSequences_ || labels_);
    u32 maxSeqLength = 0, sumSeqLength = 0;
    u32 nLabels = (stateSequences_ ? stateSequences_->size() : labels_->size());
    for (u32 s = 0; s < nLabels; ++s) {
        u32 nStates = (stateSequences_ ? (*stateSequences_)[s].nStates() : labels_->get(s).nStates());
        sumSeqLength += nStates;
        maxSeqLength = std::max(nStates, maxSeqLength);
    }
    log() << Core::XmlOpen(description)
          << Core::XmlFull("states", fst->NumStates())
          << Core::XmlFull("labels", nLabels)
          << Core::XmlFull("max hmm length", maxSeqLength)
          << Core::XmlFull("avg. hmm length", (static_cast<float>(sumSeqLength) / nLabels))
          << Core::XmlClose(description);
}
