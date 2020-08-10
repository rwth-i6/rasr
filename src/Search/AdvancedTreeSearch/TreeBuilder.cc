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
#include "TreeBuilder.hh"
#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Configuration.hh>
#include <Search/StateTree.hh>
#include <algorithm>
#include "PersistentStateTree.hh"

// TODO: Verify that pushed word-ends have the same transition penalty as the corresponding unpushed word-ends

const Core::ParameterBool paramAddCiTransitions(
        "add-ci-transitions",
        "whether context-independent acoustic transitions should be inserted between words. Useful for non-fluid speech, specifically when the training data consistent of fluid speech",
        false);  // if this is false, then an additional special-root is used, which is followed only by non-words. it is labeled #|[sil] (where [sil] is the first special-phone)

const Core::ParameterBool paramUseRootForCiExits(
        "use-root-for-ci-exits",
        "whether the root-node should be used as target for exits behind context-independent phones",
        true);

const Core::ParameterInt paramMinPhones(
        "min-phones",
        "minimum number of phones which are expanded without pushing the word ends",
        1);

const Core::ParameterInt paramMinimizeIterations(
        "minimization-iterations",
        "usually only the first 2 iterations show an effect",
        2);

const Core::ParameterBool paramForceExactWordEnds(
        "force-exact-word-ends",
        "",
        false);

const Core::ParameterBool paramKeepRoots(
        "keep-roots",
        "keep roots as they were after initial building (i.e. don't minimize them). might become useful to insert new words on-the-fly in the future, or to have correct boundary-information right after decoding.",
        false);

const Core::ParameterBool paramAllowCrossWordSkips(
        "allow-cross-word-skips",
        "add additional word labels to allow skips over word boundaries; equal skip penalties for all states are recommended",
        false);

const Core::ParameterBool paramRepeatSilence(
        "repeat-silence",
        "repeat silence. this makes cross-word skipping consistent in forward/backward case, given that all forward/skip penalties are the same",
        false);

using namespace Search;

typedef std::set<Bliss::Phoneme::Id> PhonemeList;

TreeBuilder::TreeBuilder(Core::Configuration          config,
                         const Bliss::Lexicon&        lexicon,
                         const Am::AcousticModel&     acousticModel,
                         Search::PersistentStateTree& network,
                         bool                         initialize,
                         bool                         arcBased)
        : lexicon_(lexicon),
          acousticModel_(acousticModel),
          network_(network),
          config_(config),
          minPhones_(paramMinPhones(config)),
          forceExactWordEnds_(paramForceExactWordEnds(config)),
          keepRoots_(paramKeepRoots(config)),
          allowCrossWordSkips_(paramAllowCrossWordSkips(config)),
          repeatSilence_(paramRepeatSilence(config)),
          reverse_(isBackwardRecognition(config)),
          arcBased_(arcBased) {
    if (allowCrossWordSkips_) {
        Score skipPenalty    = acousticModel_.stateTransition(0)->operator[](Am::StateTransitionModel::skip);
        Score forwardPenalty = acousticModel_.stateTransition(0)->operator[](Am::StateTransitionModel::forward);
        for (u32 t = 0; t < acousticModel_.nStateTransitions(); ++t) {
            Score modelPenalty        = acousticModel_.stateTransition(t)->operator[](Am::StateTransitionModel::skip);
            Score modelForwardPenalty = acousticModel_.stateTransition(t)->operator[](Am::StateTransitionModel::forward);
            if (modelPenalty != skipPenalty)
                Core::Application::us()->warning() << "Inconsistency for forward/backward decoding: Transition model " << t
                                                   << ": skip penalty differs from previous value: " << modelPenalty
                                                   << " (previous value " << skipPenalty << ")";
            if (modelForwardPenalty != forwardPenalty)
                Core::Application::us()->warning() << "Inconsistency for forward/backward decoding: Transition model " << t
                                                   << ": forward penalty differs from previous value: " << modelForwardPenalty
                                                   << " (previous value " << forwardPenalty << ")";
        }
    }

    if (reverse_)
        log() << "building backward network";
    else
        log() << "building forward network";

    if (initialize) {
        verify(!network_.rootState);
        network_.masterTree = network_.structure.allocateTree();

        // Non-coarticulated root state
        network_.ciRootState = network_.rootState = createRoot(Bliss::Phoneme::term, Bliss::Phoneme::term, 0);
    }
}

TreeBuilder::HMMSequence TreeBuilder::arcSequence(u32 acousticModelIndex) const {
    verify(acousticModelIndex < arcSequences_.size());
    return arcSequences_[acousticModelIndex];
}

std::string TreeBuilder::arcDesc(u32 acousticModelIndex) const {
    verify(acousticModelIndex < arcSequences_.size());
    ArcDesc            desc = arcDescs_[acousticModelIndex];
    std::ostringstream os;
    if (desc.central == Core::Type<Bliss::Phoneme::Id>::max) {
        os << "*";
    }
    else {
        if (isContextDependent(desc.central)) {
            if (desc.left == Core::Type<Bliss::Phoneme::Id>::max)
                os << "*";
            else if (desc.left == Bliss::Phoneme::term || !isContextDependent(desc.left))
                os << "#";
            else
                os << acousticModel_.phonemeInventory()->phoneme(desc.left)->symbol();
            os << "/";
        }
        os << acousticModel_.phonemeInventory()->phoneme(desc.central)->symbol();
        if (isContextDependent(desc.central)) {
            os << "/";
            if (desc.right == Core::Type<Bliss::Phoneme::Id>::max)
                os << "*";
            else if (desc.right == Bliss::Phoneme::term || !isContextDependent(desc.right))
                os << "#";
            else
                os << acousticModel_.phonemeInventory()->phoneme(desc.right)->symbol();
        }
    }
    return os.str();
}

void TreeBuilder::hmmFromAllophone(TreeBuilder::HMMSequence& ret,
                                   Bliss::Phoneme::Id        left,
                                   Bliss::Phoneme::Id        central,
                                   Bliss::Phoneme::Id        right,
                                   u32                       boundary,
                                   bool                      allowNonStandard) {
    verify(ret.length == 0);
    verify(central != Bliss::Phoneme::term);
    verify(acousticModel_.phonemeInventory()->isValidPhonemeId(central));
    Bliss::ContextPhonology::SemiContext history, future;

    if (reverse_) {
        std::swap(left, right);
        if (boundary == Am::Allophone::isFinalPhone)
            boundary = Am::Allophone::isInitialPhone;
        else if (boundary == Am::Allophone::isInitialPhone)
            boundary = Am::Allophone::isFinalPhone;
    }

    if (isContextDependent(central)) {
        if (acousticModel_.phonemeInventory()->isValidPhonemeId(left) && isContextDependent(left))
            history.append(1, left);
        if (acousticModel_.phonemeInventory()->isValidPhonemeId(right) && isContextDependent(right))
            future.append(1, right);
    }

    const Am::Allophone* allophone = acousticModel_.allophoneAlphabet()->allophone(Am::Allophone(Bliss::ContextPhonology::PhonemeInContext(central, history, future), boundary));

    const Am::ClassicHmmTopology* hmmTopology = acousticModel_.hmmTopology(central);

    for (u32 phoneState = 0; phoneState < hmmTopology->nPhoneStates(); ++phoneState) {
        Am::AllophoneState   alloState = acousticModel_.allophoneStateAlphabet()->allophoneState(allophone, phoneState);
        StateTree::StateDesc desc;
        desc.acousticModel = acousticModel_.emissionIndex(alloState);  // Decision tree look-up for CART id.

        for (u32 subState = 0; subState < hmmTopology->nSubStates(); ++subState) {
            desc.transitionModelIndex = acousticModel_.stateTransitionIndex(alloState, subState);
            verify(desc.transitionModelIndex < Core::Type<StateTree::StateDesc::TransitionModelIndex>::max);

            verify(ret.length < HMMSequence::MaxLength);  // So far hard-wired to MaxLength = 12.
            ret.hmm[ret.length] = desc;
            ++ret.length;
        }
    }

    if (arcBased_) {
        HMMSequence newRet;
        newRet.length                      = 1;
        newRet.hmm[0].transitionModelIndex = boundary;

        ArcSequenceHash::const_iterator it = arcSequencesHash_.find(ret);
        if (it != arcSequencesHash_.end()) {
            newRet.hmm[0].acousticModel = it->second;
            if (arcDescs_[it->second].central != central)
                arcDescs_[it->second].central = Core::Type<Bliss::Phoneme::Id>::max;
            if (arcDescs_[it->second].left != left)
                arcDescs_[it->second].left = Core::Type<Bliss::Phoneme::Id>::max;
            if (arcDescs_[it->second].right != right)
                arcDescs_[it->second].right = Core::Type<Bliss::Phoneme::Id>::max;
        }
        else {
            newRet.hmm[0].acousticModel = arcSequences_.size();
            verify(newRet.hmm[0].acousticModel == arcSequences_.size());
            arcSequencesHash_[ret] = arcSequences_.size();
            arcSequences_.push_back(ret);
            ArcDesc desc;
            desc.left    = left;
            desc.central = central;
            desc.right   = right;
            arcDescs_.push_back(desc);
        }
        ret = newRet;
    }
    if (reverse_)
        ret.reverse();

    if (repeatSilence_ && ret.length == 1 && central == acousticModel_.silence()) {
        ret.hmm[1] = ret.hmm[0];
        ret.length = 2;
    }
}

void TreeBuilder::build() {
    std::pair<Bliss::Lexicon::PronunciationIterator, Bliss::Lexicon::PronunciationIterator> prons = lexicon_.pronunciations();

    u32 coarticulatedInitial = 0, uncoarticulatedInitial = 0, coarticulatedFinal = 0, uncoarticulatedFinal = 0;

    // Collect initial/final phonemes
    for (Bliss::Lexicon::PronunciationIterator pronIt = prons.first; pronIt != prons.second; ++pronIt) {
        const Bliss::Pronunciation& pron(**pronIt);
        if (pron.length()) {
            Bliss::Phoneme::Id initial = pron[0], fin = pron[pron.length() - 1];
            if (reverse_)
                std::swap(initial, fin);

            if (!initialPhonemes_.count(initial)) {
                initialPhonemes_.insert(initial);
                if (isContextDependent(initial))
                    coarticulatedInitial += 1;
                else
                    uncoarticulatedInitial += 1;
            }

            if (!finalPhonemes_.count(fin)) {
                if (isContextDependent(fin))
                    coarticulatedFinal += 1;
                else
                    uncoarticulatedFinal += 1;

                finalPhonemes_.insert(fin);
            }
        }
        else {
            log() << "Ignoring 0-length pronunciation in state-network: '" << pron.format(acousticModel_.phonemeInventory()) << "'";
        }
    }

    if ((uncoarticulatedFinal == 0 || uncoarticulatedInitial == 0) && !paramAddCiTransitions(config_))
        Core::Application::us()->error() << "There are no context-independent initial or final phonemes in the lexicon, word-end detection will not work properly. Consider adding context-independent phonemes, or setting add-ci-transitions=true";

    log() << "coarticulated initial phones: " << coarticulatedInitial
          << " uncoarticulated: " << uncoarticulatedInitial
          << ", coarticulated final phones: " << uncoarticulatedFinal
          << " uncoarticulated: " << uncoarticulatedFinal;

    bool useRootForCiExits = paramUseRootForCiExits(config_) && !paramAddCiTransitions(config_);

    // Build the network-like non-coarticulated portion starting at the context-independent root
    log() << "building";
    for (Bliss::Lexicon::PronunciationIterator pronIt = prons.first; pronIt != prons.second; ++pronIt) {
        const Bliss::Pronunciation& pron(**pronIt);
        u32                         pronLength = pron.length();
        if (pronLength == 0)
            continue;

        std::pair<StateId, StateId> currentState(0, network_.rootState);

        std::vector<Bliss::Phoneme::Id> phones;
        for (u32 phoneIndex = 0; phoneIndex < pronLength; ++phoneIndex)
            phones.push_back(pron[phoneIndex]);

        if (reverse_)
            std::reverse(phones.begin(), phones.end());

        for (u32 phoneIndex = 0; phoneIndex < pronLength - 1; ++phoneIndex)
            currentState = extendPhone(currentState.second, phoneIndex, phones);

        std::pair<Bliss::Pronunciation::LemmaIterator, Bliss::Pronunciation::LemmaIterator> lemmaProns = pron.lemmas();

        if (pronLength - 1 < minPhones_ || !isContextDependent(phones[pronLength - 1])) {
            // Statically expand the fan-out.
            for (std::set<Bliss::Phoneme::Id>::iterator initialIt = initialPhonemes_.begin(); initialIt != initialPhonemes_.end(); ++initialIt) {
                std::pair<StateId, StateId> tail = extendPhone(currentState.second, pronLength - 1, phones, Bliss::Phoneme::term, *initialIt);
                for (Bliss::Pronunciation::LemmaIterator lemmaPron = lemmaProns.first; lemmaPron != lemmaProns.second; ++lemmaPron) {
                    u32 exit;
                    if (!isContextDependent(phones[pronLength - 1]) && useRootForCiExits)
                        exit = addExit(tail.first, tail.second, Bliss::Phoneme::term, Bliss::Phoneme::term, 0, lemmaPron->id());  // Use the non-coarticulated root node
                    else
                        exit = addExit(tail.first, tail.second, phones[pronLength - 1], *initialIt, 0, lemmaPron->id());
                    if (pronLength == 1)
                        initialFinalPhoneSuffix_[RootKey(phones[0], *initialIt, 1)].insert(ID_FROM_LABEL(exit));
                }
            }
        }
        else {
            // Minimize the remaining phoneme, insert corresponding word-ends.
            for (Bliss::Pronunciation::LemmaIterator lemmaPron = lemmaProns.first; lemmaPron != lemmaProns.second; ++lemmaPron) {
                if (pronLength == 1) {
                    addExit(currentState.first, currentState.second, Bliss::Phoneme::term, phones[0], -1, lemmaPron->id());

                    for (std::set<Bliss::Phoneme::Id>::const_iterator finalIt = finalPhonemes_.begin(); finalIt != finalPhonemes_.end(); ++finalIt) {
                        Search::PersistentStateTree::Exit exit;
                        exit.transitState  = createRoot(*finalIt, phones[0], -1);
                        exit.pronunciation = lemmaPron->id();
                        addSuccessor(createRoot(*finalIt, phones[0], 0), ID_FROM_LABEL(createExit(exit)));
                    }
                }
                else {
                    u32 exit = addExit(currentState.first, currentState.second, phones[pronLength - 2], phones[pronLength - 1], -1, lemmaPron->id());
                    if (pronLength == 2)
                        initialPhoneSuffix_[RootKey(phones[0], phones[1], 1)].insert(ID_FROM_LABEL(exit));
                }
            }
        }
    }

    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size() << " roots: " << roots_.size();

    buildFanInOutStructure();

    skipRootTransitions();

    u32 it = paramMinimizeIterations(config_);
    for (u32 i = 0; i < it; ++i)
        minimize();

    if (allowCrossWordSkips_)
        addCrossWordSkips();

    log() << "building ready";
}

void TreeBuilder::addCrossWordSkips() {
    log() << "adding cross-word skips";
    u32 oldNodes = network_.structure.stateCount();

    for (StateId node = 1; node < oldNodes; ++node) {
        {
            bool hasWordEnd   = false;
            bool hasSuccessor = false;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
                if (!target.isLabel())
                    hasSuccessor = true;
                if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2)
                    hasWordEnd = true;
            }
            verify(hasSuccessor || hasWordEnd);
        }

        std::set<PersistentStateTree::Exit> skipRoots;

        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (target.isLabel())
                continue;
            for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(*target); target2; ++target2)
                if (target2.isLabel())
                    skipRoots.insert(network_.exits[target2.label()]);
        }

        if (skipRoots.size()) {
            for (std::set<PersistentStateTree::Exit>::iterator it = skipRoots.begin(); it != skipRoots.end(); ++it) {
                PersistentStateTree::Exit e(*it);
                verify(e.pronunciation != Bliss::LemmaPronunciation::invalidId);
                if (network_.structure.state(e.transitState).stateDesc.transitionModelIndex == Am::TransitionModel::entryM2)
                    continue;
                e.transitState = createSkipRoot(e.transitState);
                network_.structure.addOutputToNode(node, createExit(e));
            }
        }

        {
            bool hasWordEnd   = false;
            bool hasSuccessor = false;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
                if (!target.isLabel())
                    hasSuccessor = true;
                if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2)
                    hasWordEnd = true;
            }
            verify(hasSuccessor || hasWordEnd);
        }
    }

    for (StateId node = 1; node < oldNodes; ++node) {
        bool hasWordEnd   = false;
        bool hasSuccessor = false;
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (!target.isLabel())
                hasSuccessor = true;
            if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2)
                hasWordEnd = true;
        }
        verify(hasSuccessor || hasWordEnd);
    }

    log() << "added " << network_.structure.stateCount() - oldNodes << " skip-roots";

    network_.cleanup();
}

void TreeBuilder::skipRootTransitions() {
    for (StateId node = 1; node < network_.structure.stateCount(); ++node) {
        if (network_.structure.state(node).stateDesc.acousticModel == Search::StateTree::invalidAcousticModel)
            continue;

        HMMStateNetwork::ChangePlan change = network_.structure.change(node);
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (target.isLabel())
                continue;

            if (network_.structure.state(*target).stateDesc.acousticModel == Search::StateTree::invalidAcousticModel) {
                change.removeSuccessor(*target);
                for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(*target); target2; ++target2)
                    change.addSuccessor(*target2);
            }
        }
        change.apply();
    }
}

std::vector<TreeBuilder::StateId> TreeBuilder::minimize(bool forceDeterminization, bool onlyMinimizeBackwards, bool allowLost) {
    log() << "minimizing";

    if (forceExactWordEnds_)
        log() << "forcing exact word-ends";

    for (std::set<StateId>::iterator it = network_.unpushedCoarticulatedRootStates.begin(); it != network_.unpushedCoarticulatedRootStates.end(); ++it)
        verify(network_.coarticulatedRootStates.count(*it));

    std::set<StateId>   usedRoots;
    std::deque<StateId> active;

    std::vector<u32> fanIn(network_.structure.stateCount(), 0);

    for (StateId node = 1; node < network_.structure.stateCount(); ++node) {
        active.push_back(node);
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (target.isLabel()) {
                usedRoots.insert(network_.exits[target.label()].transitState);
                fanIn[network_.exits[target.label()].transitState] += 1;
            }
            else {
                fanIn[*target] += 1;
            }
        }
    }

    log() << "keeping " << usedRoots.size() << " out of " << network_.coarticulatedRootStates.size() << " roots";
    std::set<StateId> oldCoarticulatedRoots = network_.coarticulatedRootStates;
    for (std::set<StateId>::iterator it = oldCoarticulatedRoots.begin(); it != oldCoarticulatedRoots.end(); ++it) {
        if (usedRoots.count(*it) == 0) {
            network_.coarticulatedRootStates.erase(*it);
            network_.rootTransitDescriptions.erase(*it);
            network_.unpushedCoarticulatedRootStates.erase(*it);
            network_.structure.clearOutputEdges(*it);
        }
    }

    std::vector<StateId> determinizeMap(network_.structure.stateCount(), 0);

    u32 determinizeClashes = 0;

    if (onlyMinimizeBackwards) {
        log() << "skipping determinization";
        for (StateId node = 1; node < network_.structure.stateCount(); ++node)
            determinizeMap[node] = node;
    }
    else {
        // Determinize states: Join successor states with the same state-desc
        while (!active.empty()) {
            StateId state = active.front();
            active.pop_front();
            HMMStateNetwork::ChangePlan                                                                change = network_.structure.change(state);
            typedef std::unordered_multimap<StateTree::StateDesc, StateId, StateTree::StateDesc::Hash> SuccessorHash;
            SuccessorHash                                                                              successors;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(state); target; ++target)
                if (!target.isLabel() && (forceDeterminization || fanIn[*target] == 1))
                    successors.insert(std::make_pair(network_.structure.state(*target).stateDesc, *target));

            while (!successors.empty()) {
                std::pair<SuccessorHash::iterator, SuccessorHash::iterator> items = successors.equal_range(successors.begin()->first);

                SuccessorHash::iterator it = items.first;
                if (++it != items.second) {
                    StateId newNode = network_.structure.allocateTreeNode(network_.masterTree);
                    if (newNode >= determinizeMap.size())
                        determinizeMap.resize(newNode + 1, 0);
                    network_.structure.state(newNode).stateDesc = items.first->first;
                    if (network_.uncoarticulatedWordEndStates.count(items.first->second))
                        network_.uncoarticulatedWordEndStates.insert(newNode);
                    HMMStateNetwork::ChangePlan newChange = network_.structure.change(newNode);
                    // There are multiple successors with the same state-desc, join them
                    for (it = items.first; it != items.second; ++it) {
                        verify(it->second < determinizeMap.size());
                        if (forceExactWordEnds_ && network_.uncoarticulatedWordEndStates.count(it->second))
                            network_.uncoarticulatedWordEndStates.insert(newNode);
                        if (determinizeMap[it->second])
                            ++determinizeClashes;
                        determinizeMap[it->second] = newNode;
                        for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(it->second); target2; ++target2)
                            newChange.addSuccessor(*target2);
                        change.removeSuccessor(it->second);
                    }
                    newChange.apply();
                    change.addSuccessor(newNode);
                    active.push_back(newNode);
                }

                successors.erase(items.first, items.second);
            }
            change.apply();
        }
        log() << "clashes during determinization: " << determinizeClashes;
    }

    // Minimize: Join states with the same successors/exits
    predecessors_.clear();

    std::vector<StateId> minimizeMap(network_.structure.stateCount(), 0);

    minimizeState(network_.rootState, minimizeMap);
    for (std::set<StateId>::iterator it = network_.coarticulatedRootStates.begin(); it != network_.coarticulatedRootStates.end(); ++it)
        minimizeState(*it, minimizeMap);
    for (std::set<StateId>::iterator it = skipRootSet_.begin(); it != skipRootSet_.end(); ++it)
        minimizeState(*it, minimizeMap);

    verify(minimizeMap[network_.rootState] == network_.rootState);

    if (!keepRoots_) {
        std::vector<u32> minimizeExitsMap(network_.exits.size(), Core::Type<u32>::max);
        {
            std::vector<PersistentStateTree::Exit> oldExits;
            oldExits.swap(network_.exits);
            exitHash_.clear();
            for (u32 exitIndex = 0; exitIndex < oldExits.size(); ++exitIndex) {
                PersistentStateTree::Exit exit = oldExits[exitIndex];
                exit.transitState              = minimizeMap[exit.transitState];
                verify(exit.transitState);
                minimizeExitsMap[exitIndex] = createExit(exit);
            }
        }

        log() << "clashes during determinization: " << determinizeClashes;

        log() << "joining exits, coarticulated roots before: " << network_.coarticulatedRootStates.size();
        u32 oldNodeCount = network_.structure.stateCount();  // New nodes may be added during this procedure
        for (StateId state = 1; state < oldNodeCount; ++state) {
            if (minimizeMap[state] == state)
                minimizeExits(state, minimizeExitsMap);
            else
                network_.structure.clearOutputEdges(state);
        }
    }

    log() << "coarticulated roots after joining: " << network_.coarticulatedRootStates.size();

    network_.ciRootState = network_.rootState = minimizeMap[network_.rootState];

    mapSet(network_.coarticulatedRootStates, minimizeMap, true);
    mapSet(network_.unpushedCoarticulatedRootStates, minimizeMap, true);
    mapSet(skipRootSet_, minimizeMap, true);
    mapSet(network_.uncoarticulatedWordEndStates, minimizeMap, forceExactWordEnds_);

    {
        PersistentStateTree::RootTransitDescriptions oldTransitDescs;
        oldTransitDescs.swap(network_.rootTransitDescriptions);

        for (std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::iterator it = oldTransitDescs.begin(); it != oldTransitDescs.end(); ++it) {
            StateId orig = it->first;
            if (orig == network_.rootState || orig >= minimizeMap.size()) {
                if (orig == network_.rootState || network_.coarticulatedRootStates.count(orig))
                    network_.rootTransitDescriptions.insert(*it);
            }
            else {
                StateId mapped = minimizeMap[it->first];
                verify(mapped);
                verify(network_.coarticulatedRootStates.count(mapped));

                if (mapped == network_.rootState) {
                    network_.coarticulatedRootStates.erase(network_.rootState);
                    network_.unpushedCoarticulatedRootStates.erase(network_.rootState);
                    continue;
                }
                network_.rootTransitDescriptions.insert(std::make_pair(mapped, it->second));
            }
        }
    }

    log() << "cleaning";
    u32 lost = 0, kept = 0;
    for (StateId state = 1; state < determinizeMap.size(); ++state) {
        if (determinizeMap[state])
            determinizeMap[state] = minimizeMap[determinizeMap[state]];
        else
            determinizeMap[state] = minimizeMap[state];
    }
    minimizeMap = determinizeMap;

    HMMStateNetwork::CleanupResult cleanupResult = network_.cleanup();
    for (std::vector<StateId>::iterator it = minimizeMap.begin(); it != minimizeMap.end(); ++it) {
        if (*it) {
            if (cleanupResult.nodeMap.count(*it)) {
                *it = cleanupResult.nodeMap[*it];
                kept += 1;
                verify(*it);
            }
            else {
                lost += 1;
                *it = 0;
            }
        }
    }
    log() << "transformed states: " << kept << " lost: " << lost;
    //   verify( allowLost || !lost );

    printStats("after minimization");
    return minimizeMap;
}

void TreeBuilder::mapSet(std::set<StateId>& set, const std::vector<StateId>& minimizeMap, bool force) {
    std::set<StateId> oldSet;
    oldSet.swap(set);
    for (std::set<StateId>::iterator it = oldSet.begin(); it != oldSet.end(); ++it) {
        if (*it >= minimizeMap.size())
            set.insert(*it);
        else if (!minimizeMap[*it]) {
            verify(!force);
        }
        else {
            set.insert(minimizeMap[*it]);
        }
    }
}

void TreeBuilder::minimizeState(StateId state, std::vector<StateId>& minimizeMap) {
    verify(state < minimizeMap.size());
    if (minimizeMap[state])
        return;

    minimizeMap[state] = Core::Type<u32>::max;

    verify(state && state < network_.structure.stateCount());
    std::set<StateId> successors;
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(state); target; ++target) {
        if (target.isLabel()) {
            successors.insert(*target);
            continue;
        }
        verify(*target < minimizeMap.size());
        minimizeState(*target, minimizeMap);
        verify(minimizeMap[*target] != 0);
        if (minimizeMap[*target] == Core::Type<u32>::max) {
            //       std::cout << "detected recursion while minimization on " << *target << std::endl;
            successors.insert(*target);
        }
        else {
            successors.insert(minimizeMap[*target]);
        }
    }

    network_.structure.clearOutputEdges(state);

    StatePredecessor                                                                pred(successors, network_.structure.state(state).stateDesc, forceExactWordEnds_ && network_.uncoarticulatedWordEndStates.count(state));
    std::unordered_map<StatePredecessor, StateId, StatePredecessor::Hash>::iterator it = predecessors_.find(pred);
    if (it != predecessors_.end()) {
        minimizeMap[state] = it->second;
    }
    else {
        minimizeMap[state] = state;
        predecessors_.insert(std::make_pair(pred, state));
        for (std::set<StateId>::iterator succIt = successors.begin(); succIt != successors.end(); ++succIt)
            network_.structure.addTargetToNode(state, *succIt);
    }
}

void TreeBuilder::minimizeExits(StateId state, const std::vector<u32>& minimizeExitsMap) {
    typedef std::multimap<Bliss::LemmaPronunciation::Id, u32> ExitMap;
    ExitMap                                                   successorExits;

    {
        std::set<StateId> successorStates;
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(state); target; ++target) {
            if (target.isLabel()) {
                successorExits.insert(std::make_pair(network_.exits[minimizeExitsMap[target.label()]].pronunciation, minimizeExitsMap[target.label()]));
                continue;
            }
            successorStates.insert(*target);
        }

        if (successorExits.empty())
            return;

        network_.structure.clearOutputEdges(state);
        for (std::set<StateId>::iterator it = successorStates.begin(); it != successorStates.end(); ++it)
            network_.structure.addTargetToNode(state, *it);
    }

    // Join multiple exits for the same pronunciation to one
    while (successorExits.size()) {
        std::pair<ExitMap::iterator, ExitMap::iterator> range = successorExits.equal_range(successorExits.begin()->first);
        ExitMap::iterator                               i     = range.first;
        if (++i == range.second) {
            network_.structure.addOutputToNode(state, range.first->second);
        }
        else {
            // Join
            std::set<StateId>            newRootSuccessors;
            std::set<Bliss::Phoneme::Id> left, right;
            for (i = range.first; i != range.second; ++i) {
                for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(network_.exits[i->second].transitState); target; ++target)
                    newRootSuccessors.insert(*target);
                left.insert(network_.rootTransitDescriptions[network_.exits[i->second].transitState].first);
                right.insert(network_.rootTransitDescriptions[network_.exits[i->second].transitState].second);
            }

            PersistentStateTree::Exit exit;
            exit.pronunciation = range.first->first;
            u32 newNodeLimit   = network_.structure.stateCount();
            exit.transitState  = extendFanIn(newRootSuccessors, rootDesc());
            network_.structure.addOutputToNode(state, createExit(exit));
            if (exit.transitState >= newNodeLimit) {
                network_.coarticulatedRootStates.insert(exit.transitState);
                network_.rootTransitDescriptions.insert(std::make_pair(exit.transitState, std::make_pair(left.size() == 1 ? *left.begin() : Bliss::Phoneme::term, right.size() == 1 ? *right.begin() : Bliss::Phoneme::term)));
                for (i = range.first; i != range.second; ++i) {
                    verify(i->second < network_.exits.size());
                    if (network_.unpushedCoarticulatedRootStates.count(network_.exits[i->second].transitState))
                        network_.unpushedCoarticulatedRootStates.insert(exit.transitState);
                    if (network_.uncoarticulatedWordEndStates.count(network_.exits[i->second].transitState))
                        network_.uncoarticulatedWordEndStates.insert(exit.transitState);
                }
            }
        }
        successorExits.erase(range.first, range.second);
    }
}

void TreeBuilder::buildFanInOutStructure() {
    bool ciTransitions = paramAddCiTransitions(config_);

    // Create temporary coarticulated roots
    for (std::set<Bliss::Phoneme::Id>::const_iterator finalIt = finalPhonemes_.begin(); finalIt != finalPhonemes_.end(); ++finalIt)
        for (std::set<Bliss::Phoneme::Id>::const_iterator initialIt = initialPhonemes_.begin(); initialIt != initialPhonemes_.end(); ++initialIt)
            createRoot(*finalIt, *initialIt, 0);

    log() << "building fan-in";
    // Build the fan-in structure (eg. the HMM structure representing the initial word phonemes, behind roots, up to the joints)
    for (RootHash::const_iterator rootIt = roots_.begin(); rootIt != roots_.end(); ++rootIt) {
        if (rootIt->first.depth != 0 || rootIt->second == network_.rootState)
            continue;
        Bliss::Phoneme::Id initial = rootIt->first.right;
        verify(initialPhonemes_.count(initial));
        verify(initial != Bliss::Phoneme::term);
        u32 paths = 0;

        for (CoarticulationJointHash::const_iterator initialSuffixIt = initialPhoneSuffix_.begin(); initialSuffixIt != initialPhoneSuffix_.end(); ++initialSuffixIt) {
            if (initialSuffixIt->first.left != initial)
                continue;
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, rootIt->first.left, initial, initialSuffixIt->first.right, Am::Allophone::isInitialPhone);
            verify(hmm.length > 0);
            StateId currentNode = extendFanIn(initialSuffixIt->second, hmm[hmm.length - 1]);
            for (s32 s = hmm.length - 2; s >= 0; --s)
                currentNode = extendFanIn(currentNode, hmm[s]);

            addSuccessor(rootIt->second, currentNode);
        }

        for (CoarticulationJointHash::const_iterator initialSuffixIt = initialFinalPhoneSuffix_.begin(); initialSuffixIt != initialFinalPhoneSuffix_.end(); ++initialSuffixIt) {
            if (initialSuffixIt->first.left != initial)
                continue;
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, rootIt->first.left, initial, initialSuffixIt->first.right, Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
            verify(hmm.length > 0);
            StateId currentNode = extendFanIn(initialSuffixIt->second, hmm[hmm.length - 1]);
            for (s32 s = hmm.length - 2; s >= 0; --s)
                currentNode = extendFanIn(currentNode, hmm[s]);

            addSuccessor(rootIt->second, currentNode);
        }
    }

    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size() << " roots: " << roots_.size();

    log() << "building fan-out";

    // Build the fan-out structure (eg. the HMM structure representing the final word phonemes, behind special roots)
    // At the left side delimited by the roots of depth -1, and at the right side by the roots of depth 0
    for (RootHash::const_iterator leftRootIt = roots_.begin(); leftRootIt != roots_.end(); ++leftRootIt) {
        if (leftRootIt->first.depth != -1)
            continue;
        Bliss::Phoneme::Id fin = leftRootIt->first.right;
        verify(finalPhonemes_.count(fin));

        u32 paths = 0;
        for (RootHash::const_iterator rightRootIt = roots_.begin(); rightRootIt != roots_.end(); ++rightRootIt) {
            if (rightRootIt->first.depth != 0 || (rightRootIt->first.left != fin && (!ciTransitions || rightRootIt->first.left != Bliss::Phoneme::term)))
                continue;
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, leftRootIt->first.left, fin, rightRootIt->first.right, Am::Allophone::isFinalPhone, false);
            verify(hmm.length > 0);

            // The last state in the pushed fan-in is equivalent with the corresponding root state

            StateId lastNode    = extendFanIn(network_.structure.targetSet(rightRootIt->second), hmm[hmm.length - 1]);
            StateId currentNode = lastNode;
            for (s32 s = hmm.length - 2; s >= 0; --s)
                currentNode = extendFanIn(currentNode, hmm[s]);

            if (rightRootIt->first.right == Bliss::Phoneme::term || !isContextDependent(rightRootIt->first.right))
                network_.uncoarticulatedWordEndStates.insert(lastNode);

            addSuccessor(leftRootIt->second, currentNode);
        }
        verify(paths);
    }

    printStats("after fan-in/out structure");
}

void TreeBuilder::printStats(std::string occasion) {
    log() << "stats " << occasion << ":";
    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size();
    log() << "coarticulated roots: " << network_.coarticulatedRootStates.size() << " unpushed: " << network_.unpushedCoarticulatedRootStates.size();
    u32 roots = 0;
    for (std::set<StateId>::iterator it = network_.uncoarticulatedWordEndStates.begin(); it != network_.uncoarticulatedWordEndStates.end(); ++it)
        if (network_.coarticulatedRootStates.count(*it))
            ++roots;
    log() << "number of uncoarticulated pushed word-end nodes: " << network_.uncoarticulatedWordEndStates.size() << " out of those are roots: " << roots;
}

TreeBuilder::StateId TreeBuilder::createSkipRoot(StateId baseRoot) {
    SkipRootsHash::const_iterator it = skipRoots_.find(baseRoot);
    if (it != skipRoots_.end())
        return it->second;
    StateTree::StateDesc desc = rootDesc();
    desc.transitionModelIndex = Am::TransitionModel::entryM2;
    StateId ret               = createState(desc);

    skipRoots_.insert(std::make_pair(baseRoot, ret));
    network_.structure.addTargetToNode(ret, baseRoot);
    skipRootSet_.insert(ret);
    network_.coarticulatedRootStates.insert(ret);
    verify(network_.rootTransitDescriptions.count(baseRoot));
    network_.rootTransitDescriptions.insert(std::make_pair(ret, network_.rootTransitDescriptions[baseRoot]));
    return ret;
}

TreeBuilder::StateId TreeBuilder::createRoot(Bliss::Phoneme::Id left, Bliss::Phoneme::Id right, int depth) {
    RootKey                  key(left, right, depth);
    RootHash::const_iterator it = roots_.find(key);
    if (it != roots_.end())
        return it->second;
    StateId ret = createState(rootDesc());
    if (depth == 0 && (left != Bliss::Phoneme::term || right != Bliss::Phoneme::term))
        network_.unpushedCoarticulatedRootStates.insert(ret);

    if (right == Bliss::Phoneme::term || !acousticModel_.phonemeInventory()->phoneme(right)->isContextDependent())
        network_.uncoarticulatedWordEndStates.insert(ret);

    if (left != Bliss::Phoneme::term || right != Bliss::Phoneme::term)
        network_.coarticulatedRootStates.insert(ret);
    roots_.insert(std::make_pair(key, ret));

    network_.rootTransitDescriptions.insert(std::make_pair(ret, std::make_pair(left, right)));

    return ret;
}

TreeBuilder::StateId TreeBuilder::createState(StateTree::StateDesc desc) {
    StateId ret                             = network_.structure.allocateTreeNode(network_.masterTree);
    network_.structure.state(ret).stateDesc = desc;
    return ret;
}

TreeBuilder::StateId TreeBuilder::extendFanIn(StateId successorOrExit, StateTree::StateDesc desc) {
    std::set<StateId> successors;
    successors.insert(successorOrExit);
    return extendFanIn(successors, desc);
}

TreeBuilder::StateId TreeBuilder::extendFanIn(const std::set<StateId>& successorsOrExits, Search::StateTree::StateDesc desc) {
    StatePredecessor           pred(successorsOrExits, desc);
    PredecessorsHash::iterator it = predecessors_.find(pred);
    if (it != predecessors_.end())
        return it->second;
    StateId ret = createState(desc);
    for (std::set<StateId>::const_iterator it = successorsOrExits.begin(); it != successorsOrExits.end(); ++it)
        network_.structure.addTargetToNode(ret, *it);
    predecessors_.insert(std::make_pair(pred, ret));
    return ret;
}

bool TreeBuilder::addSuccessor(StateId predecessor, StateId successor) {
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target)
        if (*target == successor)
            return false;

    network_.structure.addTargetToNode(predecessor, successor);
    return true;
}

std::pair<TreeBuilder::StateId, TreeBuilder::StateId> TreeBuilder::extendPhone(StateId                                currentState,
                                                                               u32                                    phoneIndex,
                                                                               const std::vector<Bliss::Phoneme::Id>& phones,
                                                                               Bliss::Phoneme::Id                     left,
                                                                               Bliss::Phoneme::Id                     right) {
    u8 boundary = 0;
    if (phoneIndex != 0)
        left = phones[phoneIndex - 1];
    else
        boundary |= Am::Allophone::isInitialPhone;

    if (phoneIndex != phones.size() - 1)
        right = phones[phoneIndex + 1];
    else
        boundary |= Am::Allophone::isFinalPhone;

    HMMSequence hmm;
    hmmFromAllophone(hmm, left, phones[phoneIndex], right, boundary, phoneIndex == 0);

    verify(hmm.length >= 1);

    u32     hmmState      = 0;
    StateId previousState = 0;

    if (phoneIndex == 1 && hmmState == 0)
        currentState = extendBodyState(currentState, left, phones[phoneIndex], hmm[hmmState++]);

    for (; hmmState < hmm.length; ++hmmState) {
        previousState = currentState;
        currentState  = extendState(currentState, hmm[hmmState]);
    }

    return std::make_pair(previousState, currentState);
}

TreeBuilder::StateId TreeBuilder::extendState(StateId predecessor, StateTree::StateDesc desc, TreeBuilder::RootKey uniqueKey) {
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target)
        if (!target.isLabel() && network_.structure.state(*target).stateDesc == desc) {
            if (uniqueKey.isValid()) {
                Core::HashMap<StateId, RootKey>::const_iterator it = stateUniqueKeys_.find(*target);
                verify(it != stateUniqueKeys_.end());
                if (!(it->second == uniqueKey))
                    continue;
            }
            return *target;
        }

    // No matching successor found, extend
    StateId ret = createState(desc);
    if (uniqueKey.isValid())
        stateUniqueKeys_.insert(std::make_pair(ret, uniqueKey));
    network_.structure.addTargetToNode(predecessor, ret);
    return ret;
}

u32 TreeBuilder::createExit(PersistentStateTree::Exit exit) {
    ExitHash::iterator exitHashIt = exitHash_.find(exit);
    if (exitHashIt != exitHash_.end()) {
        return exitHashIt->second;
    }
    else {
        // Exit does not exist yet, add it
        network_.exits.push_back(exit);
        u32 exitIndex = network_.exits.size() - 1;
        exitHash_.insert(std::make_pair(exit, exitIndex));
        return exitIndex;
    }
}

u32 TreeBuilder::addExit(StateId                       prePredecessor,
                         StateId                       predecessor,
                         Bliss::Phoneme::Id            leftPhoneme,
                         Bliss::Phoneme::Id            rightPhoneme,
                         int                           depth,
                         Bliss::LemmaPronunciation::Id pron) {
    PersistentStateTree::Exit exit;
    exit.transitState  = createRoot(leftPhoneme, rightPhoneme, depth);
    exit.pronunciation = pron;

    u32 exitIndex = createExit(exit);

    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target)
        if (target.isLabel() && target.label() == exitIndex)
            return exitIndex;

    network_.structure.addOutputToNode(predecessor, ID_FROM_LABEL(exitIndex));
    return exitIndex;
}

TreeBuilder::StateId TreeBuilder::extendBodyState(StateId                      state,
                                                  Bliss::Phoneme::Id           first,
                                                  Bliss::Phoneme::Id           second,
                                                  Search::StateTree::StateDesc desc) {
    RootKey key(first, second, 1);
    StateId ret = extendState(state, desc, key);
    initialPhoneSuffix_[key].insert(ret);
    return ret;
}

bool TreeBuilder::isContextDependent(Bliss::Phoneme::Id phone) const {
    return acousticModel_.phonemeInventory()->phoneme(phone)->isContextDependent();
}

StateTree::StateDesc TreeBuilder::rootDesc() const {
    StateTree::StateDesc desc;
    desc.acousticModel        = Search::StateTree::invalidAcousticModel;
    desc.transitionModelIndex = Am::TransitionModel::entryM1;
    return desc;
}

std::string TreeBuilder::describe(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id> desc) {
    std::ostringstream os;

    if (desc.first == Bliss::Phoneme::term)
        os << "#";
    else
        os << lexicon_.phonemeInventory()->phoneme(desc.first)->symbol();

    os << "<->";

    if (desc.second == Bliss::Phoneme::term)
        os << "#";
    else
        os << lexicon_.phonemeInventory()->phoneme(desc.second)->symbol();

    return os.str();
}

Core::Component::Message TreeBuilder::log() const {
    return Core::Application::us()->log("TreeBuilder: ");
}
