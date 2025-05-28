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

#include <algorithm>

#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Configuration.hh>

#include "Helpers.hh"
#include "PersistentStateTree.hh"
#include "StateTree.hh"
#include "Types.hh"

using namespace Search;

// -------------------- AbstractTreeBuilder --------------------

AbstractTreeBuilder::AbstractTreeBuilder(Core::Configuration          config,
                                         const Bliss::Lexicon&        lexicon,
                                         const Am::AcousticModel&     acousticModel,
                                         Search::PersistentStateTree& network)
        : Core::Component(config),
          lexicon_(lexicon),
          acousticModel_(acousticModel),
          network_(network) {
}

StateId AbstractTreeBuilder::createState(StateTree::StateDesc desc) {
    StateId ret                             = network_.structure.allocateTreeNode();
    network_.structure.state(ret).stateDesc = desc;
    return ret;
}

u32 AbstractTreeBuilder::createExit(PersistentStateTree::Exit exit) {
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

// -------------------- MinimizedTreeBuilder --------------------

// TODO: Verify that pushed word-ends have the same transition penalty as the corresponding unpushed word-ends

const Core::ParameterInt MinimizedTreeBuilder::paramMinPhones(
        "min-phones",
        "minimum number of phones which are expanded without pushing the word ends",
        1);

const Core::ParameterBool MinimizedTreeBuilder::paramAddCiTransitions(
        "add-ci-transitions",
        "whether context-independent acoustic transitions should be inserted between words. Useful for non-fluid speech, specifically when the training data consistent of fluid speech",
        false);  // if this is false, then an additional special-root is used, which is followed only by non-words. it is labeled #|[sil] (where [sil] is the first special-phone)

const Core::ParameterBool MinimizedTreeBuilder::paramUseRootForCiExits(
        "use-root-for-ci-exits",
        "whether the root-node should be used as target for exits behind context-independent phones",
        true);

const Core::ParameterBool MinimizedTreeBuilder::paramForceExactWordEnds(
        "force-exact-word-ends",
        "",
        false);

const Core::ParameterBool MinimizedTreeBuilder::paramKeepRoots(
        "keep-roots",
        "keep roots as they were after initial building (i.e. don't minimize them). might become useful to insert new words on-the-fly in the future, or to have correct boundary-information right after decoding.",
        false);

const Core::ParameterBool MinimizedTreeBuilder::paramAllowCrossWordSkips(
        "allow-cross-word-skips",
        "add additional word labels to allow skips over word boundaries; equal skip penalties for all states are recommended",
        false);

const Core::ParameterBool MinimizedTreeBuilder::paramRepeatSilence(
        "repeat-silence",
        "repeat silence. this makes cross-word skipping consistent in forward/backward case, given that all forward/skip penalties are the same",
        false);

const Core::ParameterInt MinimizedTreeBuilder::paramMinimizeIterations(
        "minimization-iterations",
        "usually only the first 2 iterations show an effect",
        2);

MinimizedTreeBuilder::MinimizedTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize)
        : AbstractTreeBuilder(config, lexicon, acousticModel, network),
          minPhones_(paramMinPhones(config)),
          addCiTransitions_(paramAddCiTransitions(config)),
          useRootForCiExits_(paramUseRootForCiExits(config)),
          forceExactWordEnds_(paramForceExactWordEnds(config)),
          keepRoots_(paramKeepRoots(config)),
          allowCrossWordSkips_(paramAllowCrossWordSkips(config)),
          repeatSilence_(paramRepeatSilence(config)),
          minimizeIterations_(paramMinimizeIterations(config)),
          reverse_(isBackwardRecognition(config)) {
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

    if (reverse_) {
        log() << "building backward network";
    }
    else {
        log() << "building forward network";
    }

    if (initialize) {
        verify(!network_.rootState);

        // Non-coarticulated root state
        network_.ciRootState = network_.rootState = createRoot(Bliss::Phoneme::term, Bliss::Phoneme::term, 0);
    }
}

std::unique_ptr<AbstractTreeBuilder> MinimizedTreeBuilder::newInstance(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize) {
    return std::unique_ptr<AbstractTreeBuilder>(new MinimizedTreeBuilder(config, lexicon, acousticModel, network, initialize));
}

void MinimizedTreeBuilder::build() {
    buildBody();

    buildFanInOutStructure();

    skipRootTransitions();

    for (u32 i = 0; i < minimizeIterations_; ++i) {
        minimize();
    }

    if (allowCrossWordSkips_) {
        addCrossWordSkips();
    }

    log() << "building ready";
}

void MinimizedTreeBuilder::printStats(std::string occasion) {
    log() << "stats " << occasion << ":";
    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size();
    log() << "coarticulated roots: " << network_.coarticulatedRootStates.size() << " unpushed: " << network_.unpushedCoarticulatedRootStates.size();
    u32 roots = 0;
    for (std::set<StateId>::iterator it = network_.uncoarticulatedWordEndStates.begin(); it != network_.uncoarticulatedWordEndStates.end(); ++it) {
        if (network_.coarticulatedRootStates.count(*it)) {
            ++roots;
        }
    }
    log() << "number of uncoarticulated pushed word-end nodes: " << network_.uncoarticulatedWordEndStates.size() << " out of those are roots: " << roots;
}

std::string MinimizedTreeBuilder::describe(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id> desc) {
    std::ostringstream os;

    if (desc.first == Bliss::Phoneme::term) {
        os << "#";
    }
    else {
        os << lexicon_.phonemeInventory()->phoneme(desc.first)->symbol();
    }

    os << "<->";

    if (desc.second == Bliss::Phoneme::term) {
        os << "#";
    }
    else {
        os << lexicon_.phonemeInventory()->phoneme(desc.second)->symbol();
    }

    return os.str();
}

bool MinimizedTreeBuilder::isContextDependent(Bliss::Phoneme::Id phone) const {
    return acousticModel_.phonemeInventory()->phoneme(phone)->isContextDependent();
}

void MinimizedTreeBuilder::buildBody() {
    std::pair<Bliss::Lexicon::PronunciationIterator, Bliss::Lexicon::PronunciationIterator> prons = lexicon_.pronunciations();

    u32 coarticulatedInitial = 0, uncoarticulatedInitial = 0, coarticulatedFinal = 0, uncoarticulatedFinal = 0;

    // Collect initial/final phonemes
    for (Bliss::Lexicon::PronunciationIterator pronIt = prons.first; pronIt != prons.second; ++pronIt) {
        const Bliss::Pronunciation& pron(**pronIt);
        if (pron.length()) {
            Bliss::Phoneme::Id initial = pron[0], fin = pron[pron.length() - 1];
            if (reverse_) {
                std::swap(initial, fin);
            }

            if (!initialPhonemes_.count(initial)) {
                initialPhonemes_.insert(initial);
                if (isContextDependent(initial)) {
                    coarticulatedInitial += 1;
                }
                else {
                    uncoarticulatedInitial += 1;
                }
            }

            if (!finalPhonemes_.count(fin)) {
                if (isContextDependent(fin)) {
                    coarticulatedFinal += 1;
                }
                else {
                    uncoarticulatedFinal += 1;
                }
                finalPhonemes_.insert(fin);
            }
        }
        else {
            log() << "Ignoring 0-length pronunciation in state-network: '" << pron.format(acousticModel_.phonemeInventory()) << "'";
        }
    }

    if ((uncoarticulatedFinal == 0 || uncoarticulatedInitial == 0) && !addCiTransitions_) {
        Core::Application::us()->error() << "There are no context-independent initial or final phonemes in the lexicon, word-end detection will not work properly. Consider adding context-independent phonemes, or setting add-ci-transitions=true";
    }

    log() << "coarticulated initial phones: " << coarticulatedInitial
          << " uncoarticulated: " << uncoarticulatedInitial
          << ", coarticulated final phones: " << uncoarticulatedFinal
          << " uncoarticulated: " << uncoarticulatedFinal;

    bool useRootForCiExits = useRootForCiExits_ && !addCiTransitions_;

    // Build the network-like non-coarticulated portion starting at the context-independent root
    log() << "building";
    for (Bliss::Lexicon::PronunciationIterator pronIt = prons.first; pronIt != prons.second; ++pronIt) {
        const Bliss::Pronunciation& pron(**pronIt);
        u32                         pronLength = pron.length();
        if (pronLength == 0) {
            continue;
        }

        std::pair<StateId, StateId> currentState(0, network_.rootState);

        std::vector<Bliss::Phoneme::Id> phones;
        for (u32 phoneIndex = 0; phoneIndex < pronLength; ++phoneIndex) {
            phones.push_back(pron[phoneIndex]);
        }

        if (reverse_) {
            std::reverse(phones.begin(), phones.end());
        }

        for (u32 phoneIndex = 0; phoneIndex < pronLength - 1; ++phoneIndex) {
            currentState = extendPhone(currentState.second, phoneIndex, phones);
        }

        std::pair<Bliss::Pronunciation::LemmaIterator, Bliss::Pronunciation::LemmaIterator> lemmaProns = pron.lemmas();

        if (pronLength - 1 < minPhones_ || !isContextDependent(phones[pronLength - 1])) {
            // Statically expand the fan-out.
            for (std::set<Bliss::Phoneme::Id>::iterator initialIt = initialPhonemes_.begin(); initialIt != initialPhonemes_.end(); ++initialIt) {
                std::pair<StateId, StateId> tail = extendPhone(currentState.second, pronLength - 1, phones, Bliss::Phoneme::term, *initialIt);
                for (Bliss::Pronunciation::LemmaIterator lemmaPron = lemmaProns.first; lemmaPron != lemmaProns.second; ++lemmaPron) {
                    u32 exit;
                    if (!isContextDependent(phones[pronLength - 1]) && useRootForCiExits) {
                        exit = addExit(tail.second, Bliss::Phoneme::term, Bliss::Phoneme::term, 0, lemmaPron->id());  // Use the non-coarticulated root node
                    }
                    else {
                        exit = addExit(tail.second, phones[pronLength - 1], *initialIt, 0, lemmaPron->id());
                    }
                    if (pronLength == 1) {
                        initialFinalPhoneSuffix_[RootKey(phones[0], *initialIt, 1)].insert(ID_FROM_LABEL(exit));
                    }
                }
            }
        }
        else {
            // Minimize the remaining phoneme, insert corresponding word-ends.
            for (Bliss::Pronunciation::LemmaIterator lemmaPron = lemmaProns.first; lemmaPron != lemmaProns.second; ++lemmaPron) {
                if (pronLength == 1) {
                    addExit(currentState.second, Bliss::Phoneme::term, phones[0], -1, lemmaPron->id());

                    for (std::set<Bliss::Phoneme::Id>::const_iterator finalIt = finalPhonemes_.begin(); finalIt != finalPhonemes_.end(); ++finalIt) {
                        Search::PersistentStateTree::Exit exit;
                        exit.transitState  = createRoot(*finalIt, phones[0], -1);
                        exit.pronunciation = lemmaPron->id();
                        addSuccessor(createRoot(*finalIt, phones[0], 0), ID_FROM_LABEL(createExit(exit)));
                    }
                }
                else {
                    u32 exit = addExit(currentState.second, phones[pronLength - 2], phones[pronLength - 1], -1, lemmaPron->id());
                    if (pronLength == 2) {
                        initialPhoneSuffix_[RootKey(phones[0], phones[1], 1)].insert(ID_FROM_LABEL(exit));
                    }
                }
            }
        }
    }

    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size() << " roots: " << roots_.size();
}

void MinimizedTreeBuilder::buildFanInOutStructure() {
    // Create temporary coarticulated roots
    for (std::set<Bliss::Phoneme::Id>::const_iterator finalIt = finalPhonemes_.begin(); finalIt != finalPhonemes_.end(); ++finalIt) {
        for (std::set<Bliss::Phoneme::Id>::const_iterator initialIt = initialPhonemes_.begin(); initialIt != initialPhonemes_.end(); ++initialIt) {
            createRoot(*finalIt, *initialIt, 0);
        }
    }

    log() << "building fan-in";
    // Build the fan-in structure (e.g. the HMM structure representing the initial word phonemes, behind roots, up to the joints)
    for (RootHash::const_iterator rootIt = roots_.begin(); rootIt != roots_.end(); ++rootIt) {
        if (rootIt->first.depth != 0 || rootIt->second == network_.rootState) {
            continue;
        }
        Bliss::Phoneme::Id initial = rootIt->first.right;
        verify(initialPhonemes_.count(initial));
        verify(initial != Bliss::Phoneme::term);
        u32 paths = 0;

        for (CoarticulationJointHash::const_iterator initialSuffixIt = initialPhoneSuffix_.begin(); initialSuffixIt != initialPhoneSuffix_.end(); ++initialSuffixIt) {
            if (initialSuffixIt->first.left != initial) {
                continue;
            }
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, rootIt->first.left, initial, initialSuffixIt->first.right, Am::Allophone::isInitialPhone);
            verify(hmm.length > 0);
            StateId currentNode = extendFanIn(initialSuffixIt->second, hmm[hmm.length - 1]);
            for (s32 s = hmm.length - 2; s >= 0; --s) {
                currentNode = extendFanIn(currentNode, hmm[s]);
            }

            addSuccessor(rootIt->second, currentNode);
        }

        for (CoarticulationJointHash::const_iterator initialSuffixIt = initialFinalPhoneSuffix_.begin(); initialSuffixIt != initialFinalPhoneSuffix_.end(); ++initialSuffixIt) {
            if (initialSuffixIt->first.left != initial) {
                continue;
            }
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, rootIt->first.left, initial, initialSuffixIt->first.right, Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
            verify(hmm.length > 0);
            StateId currentNode = extendFanIn(initialSuffixIt->second, hmm[hmm.length - 1]);
            for (s32 s = hmm.length - 2; s >= 0; --s) {
                currentNode = extendFanIn(currentNode, hmm[s]);
            }

            addSuccessor(rootIt->second, currentNode);
        }
    }

    log() << "states: " << network_.structure.stateCount() << " exits: " << network_.exits.size() << " roots: " << roots_.size();
    log() << "building fan-out";

    // Build the fan-out structure (e.g. the HMM structure representing the final word phonemes, behind special roots)
    // On the left side delimited by the roots of depth -1, and on the right side by the roots of depth 0
    for (RootHash::const_iterator leftRootIt = roots_.begin(); leftRootIt != roots_.end(); ++leftRootIt) {
        if (leftRootIt->first.depth != -1) {
            continue;
        }
        Bliss::Phoneme::Id fin = leftRootIt->first.right;
        verify(finalPhonemes_.count(fin));

        u32 paths = 0;
        for (RootHash::const_iterator rightRootIt = roots_.begin(); rightRootIt != roots_.end(); ++rightRootIt) {
            if (rightRootIt->first.depth != 0 || (rightRootIt->first.left != fin && (!addCiTransitions_ || rightRootIt->first.left != Bliss::Phoneme::term))) {
                continue;
            }
            ++paths;
            HMMSequence hmm;
            hmmFromAllophone(hmm, leftRootIt->first.left, fin, rightRootIt->first.right, Am::Allophone::isFinalPhone);
            verify(hmm.length > 0);

            // The last state in the pushed fan-in is equivalent with the corresponding root state

            StateId lastNode = extendFanIn(network_.structure.targetSet(rightRootIt->second), hmm[hmm.length - 1]);

            StateId currentNode = lastNode;
            for (s32 s = hmm.length - 2; s >= 0; --s) {
                currentNode = extendFanIn(currentNode, hmm[s]);
            }

            if (rightRootIt->first.right == Bliss::Phoneme::term || !isContextDependent(rightRootIt->first.right)) {
                network_.uncoarticulatedWordEndStates.insert(lastNode);
            }

            addSuccessor(leftRootIt->second, currentNode);
        }
        verify(paths);
    }

    printStats("after fan-in/out structure");
}

void MinimizedTreeBuilder::addCrossWordSkips() {
    log() << "adding cross-word skips";
    u32 oldNodes = network_.structure.stateCount();

    for (StateId node = 1; node < oldNodes; ++node) {
        {
            bool hasWordEnd   = false;
            bool hasSuccessor = false;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
                if (!target.isLabel()) {
                    hasSuccessor = true;
                }
                if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2) {
                    hasWordEnd = true;
                }
            }
            verify(hasSuccessor || hasWordEnd);
        }

        std::set<PersistentStateTree::Exit> skipRoots;

        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (target.isLabel()) {
                continue;
            }
            for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(*target); target2; ++target2) {
                if (target2.isLabel()) {
                    skipRoots.insert(network_.exits[target2.label()]);
                }
            }
        }

        if (skipRoots.size()) {
            for (std::set<PersistentStateTree::Exit>::iterator it = skipRoots.begin(); it != skipRoots.end(); ++it) {
                PersistentStateTree::Exit e(*it);
                verify(e.pronunciation != Bliss::LemmaPronunciation::invalidId);
                if (network_.structure.state(e.transitState).stateDesc.transitionModelIndex == Am::TransitionModel::entryM2) {
                    continue;
                }
                e.transitState = createSkipRoot(e.transitState);
                network_.structure.addOutputToNode(node, createExit(e));
            }
        }

        {
            bool hasWordEnd   = false;
            bool hasSuccessor = false;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
                if (!target.isLabel()) {
                    hasSuccessor = true;
                }
                if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2) {
                    hasWordEnd = true;
                }
            }
            verify(hasSuccessor || hasWordEnd);
        }
    }

    for (StateId node = 1; node < oldNodes; ++node) {
        bool hasWordEnd   = false;
        bool hasSuccessor = false;
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (!target.isLabel()) {
                hasSuccessor = true;
            }
            if (target.isLabel() && network_.structure.state(network_.exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2) {
                hasWordEnd = true;
            }
        }
        verify(hasSuccessor || hasWordEnd);
    }

    log() << "added " << network_.structure.stateCount() - oldNodes << " skip-roots";

    network_.cleanup();
}

void MinimizedTreeBuilder::skipRootTransitions(StateId start) {
    for (StateId node = start; node < network_.structure.stateCount(); ++node) {
        if (network_.structure.state(node).stateDesc.acousticModel == Search::StateTree::invalidAcousticModel) {
            continue;
        }

        HMMStateNetwork::ChangePlan change = network_.structure.change(node);
        for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(node); target; ++target) {
            if (target.isLabel()) {
                continue;
            }

            if (network_.structure.state(*target).stateDesc.acousticModel == Search::StateTree::invalidAcousticModel) {
                change.removeSuccessor(*target);
                for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(*target); target2; ++target2) {
                    change.addSuccessor(*target2);
                }
            }
        }
        change.apply();
    }
}

StateTree::StateDesc MinimizedTreeBuilder::rootDesc() const {
    StateTree::StateDesc desc;
    desc.acousticModel        = Search::StateTree::invalidAcousticModel;
    desc.transitionModelIndex = Am::TransitionModel::entryM1;
    return desc;
}

AbstractTreeBuilder::StateId MinimizedTreeBuilder::createSkipRoot(StateId baseRoot) {
    SkipRootsHash::const_iterator it = skipRoots_.find(baseRoot);
    if (it != skipRoots_.end()) {
        return it->second;
    }
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

AbstractTreeBuilder::StateId MinimizedTreeBuilder::createRoot(Bliss::Phoneme::Id left, Bliss::Phoneme::Id right, int depth) {
    RootKey                  key(left, right, depth);
    RootHash::const_iterator it = roots_.find(key);
    if (it != roots_.end()) {
        return it->second;
    }

    // record newly inserted RootStates to reset network_ later
    StateId ret = createState(rootDesc());

    if (depth == 0 && (left != Bliss::Phoneme::term || right != Bliss::Phoneme::term)) {
        network_.unpushedCoarticulatedRootStates.insert(ret);
    }

    if (right == Bliss::Phoneme::term || !acousticModel_.phonemeInventory()->phoneme(right)->isContextDependent()) {
        network_.uncoarticulatedWordEndStates.insert(ret);
    }

    if (left != Bliss::Phoneme::term || right != Bliss::Phoneme::term) {
        network_.coarticulatedRootStates.insert(ret);
    }

    roots_.insert(std::make_pair(key, ret));

    network_.rootTransitDescriptions.insert(std::make_pair(ret, std::make_pair(left, right)));

    return ret;
}

u32 MinimizedTreeBuilder::addExit(StateId                       predecessor,
                                  Bliss::Phoneme::Id            leftPhoneme,
                                  Bliss::Phoneme::Id            rightPhoneme,
                                  int                           depth,
                                  Bliss::LemmaPronunciation::Id pron) {
    PersistentStateTree::Exit exit;
    exit.transitState  = createRoot(leftPhoneme, rightPhoneme, depth);
    exit.pronunciation = pron;

    u32 exitIndex = createExit(exit);

    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target) {
        if (target.isLabel() && target.label() == exitIndex) {
            return exitIndex;
        }
    }

    network_.structure.addOutputToNode(predecessor, ID_FROM_LABEL(exitIndex));
    return exitIndex;
}

void MinimizedTreeBuilder::hmmFromAllophone(HMMSequence&       ret,
                                            Bliss::Phoneme::Id left,
                                            Bliss::Phoneme::Id central,
                                            Bliss::Phoneme::Id right,
                                            u32                boundary) {
    verify(ret.length == 0);
    verify(central != Bliss::Phoneme::term);
    verify(acousticModel_.phonemeInventory()->isValidPhonemeId(central));
    Bliss::ContextPhonology::SemiContext history, future;

    if (reverse_) {
        std::swap(left, right);
        if (boundary == Am::Allophone::isFinalPhone) {
            boundary = Am::Allophone::isInitialPhone;
        }
        else if (boundary == Am::Allophone::isInitialPhone) {
            boundary = Am::Allophone::isFinalPhone;
        }
    }

    if (isContextDependent(central)) {
        if (acousticModel_.phonemeInventory()->isValidPhonemeId(left) && isContextDependent(left)) {
            history.append(1, left);
        }
        if (acousticModel_.phonemeInventory()->isValidPhonemeId(right) && isContextDependent(right)) {
            future.append(1, right);
        }
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

    if (reverse_) {
        ret.reverse();
    }

    if (repeatSilence_ && ret.length == 1 && central == acousticModel_.silence()) {
        ret.hmm[1] = ret.hmm[0];
        ret.length = 2;
    }
}

bool MinimizedTreeBuilder::addSuccessor(StateId predecessor, StateId successor) {
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target) {
        if (*target == successor) {
            return false;
        }
    }

    network_.structure.addTargetToNode(predecessor, successor);
    return true;
}

std::pair<AbstractTreeBuilder::StateId, AbstractTreeBuilder::StateId> MinimizedTreeBuilder::extendPhone(StateId                                currentState,
                                                                                                        u32                                    phoneIndex,
                                                                                                        const std::vector<Bliss::Phoneme::Id>& phones,
                                                                                                        Bliss::Phoneme::Id                     left,
                                                                                                        Bliss::Phoneme::Id                     right) {
    u8 boundary = 0;
    if (phoneIndex != 0) {
        left = phones[phoneIndex - 1];
    }
    else {
        boundary |= Am::Allophone::isInitialPhone;
    }

    if (phoneIndex != phones.size() - 1) {
        right = phones[phoneIndex + 1];
    }
    else {
        boundary |= Am::Allophone::isFinalPhone;
    }

    HMMSequence hmm;
    hmmFromAllophone(hmm, left, phones[phoneIndex], right, boundary);

    verify(hmm.length >= 1);

    u32     hmmState      = 0;
    StateId previousState = 0;

    if (phoneIndex == 1 && hmmState == 0) {
        currentState = extendBodyState(currentState, left, phones[phoneIndex], hmm[hmmState++]);
    }

    for (; hmmState < hmm.length; ++hmmState) {
        previousState = currentState;
        currentState  = extendState(currentState, hmm[hmmState]);
    }

    return std::make_pair(previousState, currentState);
}

AbstractTreeBuilder::StateId MinimizedTreeBuilder::extendState(StateId predecessor, StateTree::StateDesc desc, MinimizedTreeBuilder::RootKey uniqueKey) {
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target) {
        if (!target.isLabel() && network_.structure.state(*target).stateDesc == desc) {
            if (uniqueKey.isValid()) {
                Core::HashMap<StateId, RootKey>::const_iterator it = stateUniqueKeys_.find(*target);
                verify(it != stateUniqueKeys_.end());
                if (!(it->second == uniqueKey)) {
                    continue;
                }
            }
            return *target;
        }
    }

    // No matching successor found, extend
    StateId ret = createState(desc);
    if (uniqueKey.isValid()) {
        stateUniqueKeys_.insert(std::make_pair(ret, uniqueKey));
    }
    network_.structure.addTargetToNode(predecessor, ret);
    return ret;
}

AbstractTreeBuilder::StateId MinimizedTreeBuilder::extendBodyState(StateId                      state,
                                                                   Bliss::Phoneme::Id           first,
                                                                   Bliss::Phoneme::Id           second,
                                                                   Search::StateTree::StateDesc desc) {
    RootKey key(first, second, 1);
    StateId ret = extendState(state, desc, key);
    initialPhoneSuffix_[key].insert(ret);

    return ret;
}

AbstractTreeBuilder::StateId MinimizedTreeBuilder::extendFanIn(StateId successorOrExit, StateTree::StateDesc desc) {
    std::set<StateId> successors;
    successors.insert(successorOrExit);
    return extendFanIn(successors, desc);
}

AbstractTreeBuilder::StateId MinimizedTreeBuilder::extendFanIn(const std::set<StateId>& successorsOrExits, Search::StateTree::StateDesc desc) {
    StatePredecessor           pred(successorsOrExits, desc);
    PredecessorsHash::iterator it = predecessors_.find(pred);
    if (it != predecessors_.end()) {
        return it->second;
    }
    StateId ret = createState(desc);
    for (std::set<StateId>::const_iterator it = successorsOrExits.begin(); it != successorsOrExits.end(); ++it) {
        network_.structure.addTargetToNode(ret, *it);
    }
    predecessors_.insert(std::make_pair(pred, ret));
    return ret;
}

std::vector<AbstractTreeBuilder::StateId> MinimizedTreeBuilder::minimize(bool forceDeterminization, bool onlyMinimizeBackwards, bool allowLost) {
    log() << "minimizing";

    if (forceExactWordEnds_) {
        log() << "forcing exact word-ends";
    }

    for (std::set<StateId>::iterator it = network_.unpushedCoarticulatedRootStates.begin(); it != network_.unpushedCoarticulatedRootStates.end(); ++it) {
        verify(network_.coarticulatedRootStates.count(*it));
    }

    std::set<StateId>   usedRoots;
    std::deque<StateId> active;

    std::vector<u32> fanIn(network_.structure.stateCount(), 0);

    // Collect all zero-depth roots to skip them during clean-up
    std::set<StateId> usefulRoots;
    for (RootHash::const_iterator rootIt = roots_.begin(); rootIt != roots_.end(); ++rootIt) {
        if (rootIt->first.depth == 0) {
            usefulRoots.insert(rootIt->second);
        }
    }

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

    std::set<StateId> oldCoarticulatedRoots = network_.coarticulatedRootStates;
    for (std::set<StateId>::iterator it = oldCoarticulatedRoots.begin(); it != oldCoarticulatedRoots.end(); ++it) {
        // not clean up 0-depth roots' connection if needed
        if (usedRoots.count(*it) == 0 && usefulRoots.count(*it) == 0) {
            network_.coarticulatedRootStates.erase(*it);
            network_.rootTransitDescriptions.erase(*it);
            network_.unpushedCoarticulatedRootStates.erase(*it);
            network_.structure.clearOutputEdges(*it);
        }
    }
    log() << "keeping " << network_.coarticulatedRootStates.size() << " out of " << oldCoarticulatedRoots.size() << " roots";

    std::vector<StateId> determinizeMap(network_.structure.stateCount(), 0);

    u32 determinizeClashes = 0;

    if (onlyMinimizeBackwards) {
        log() << "skipping determinization";
        for (StateId node = 1; node < network_.structure.stateCount(); ++node) {
            determinizeMap[node] = node;
        }
    }
    else {
        // Determinize states: Join successor states with the same state-desc
        while (!active.empty()) {
            StateId state = active.front();
            active.pop_front();
            HMMStateNetwork::ChangePlan                                                                change = network_.structure.change(state);
            typedef std::unordered_multimap<StateTree::StateDesc, StateId, StateTree::StateDesc::Hash> SuccessorHash;
            SuccessorHash                                                                              successors;
            for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(state); target; ++target) {
                if (!target.isLabel() && (forceDeterminization || fanIn[*target] == 1)) {
                    successors.insert(std::make_pair(network_.structure.state(*target).stateDesc, *target));
                }
            }

            while (!successors.empty()) {
                std::pair<SuccessorHash::iterator, SuccessorHash::iterator> items = successors.equal_range(successors.begin()->first);

                SuccessorHash::iterator it = items.first;
                if (++it != items.second) {
                    StateId newNode = network_.structure.allocateTreeNode();
                    if (newNode >= determinizeMap.size()) {
                        determinizeMap.resize(newNode + 1, 0);
                    }
                    network_.structure.state(newNode).stateDesc = items.first->first;
                    if (network_.uncoarticulatedWordEndStates.count(items.first->second)) {
                        network_.uncoarticulatedWordEndStates.insert(newNode);
                    }
                    HMMStateNetwork::ChangePlan newChange = network_.structure.change(newNode);
                    // There are multiple successors with the same state-desc, join them
                    for (it = items.first; it != items.second; ++it) {
                        verify(it->second < determinizeMap.size());
                        if (forceExactWordEnds_ && network_.uncoarticulatedWordEndStates.count(it->second)) {
                            network_.uncoarticulatedWordEndStates.insert(newNode);
                        }
                        if (determinizeMap[it->second]) {
                            ++determinizeClashes;
                        }
                        determinizeMap[it->second] = newNode;
                        for (HMMStateNetwork::SuccessorIterator target2 = network_.structure.successors(it->second); target2; ++target2) {
                            newChange.addSuccessor(*target2);
                        }
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
    // record original FanIn/Out related predecessor hash
    PredecessorsHash oldPredecessors;
    oldPredecessors.swap(predecessors_);

    std::vector<StateId> minimizeMap(network_.structure.stateCount(), 0);

    minimizeState(network_.rootState, minimizeMap);
    for (std::set<StateId>::iterator it = network_.coarticulatedRootStates.begin(); it != network_.coarticulatedRootStates.end(); ++it) {
        minimizeState(*it, minimizeMap);
    }
    for (std::set<StateId>::iterator it = skipRootSet_.begin(); it != skipRootSet_.end(); ++it) {
        minimizeState(*it, minimizeMap);
    }

    // loop over 0-depth roots to make sure they are mapped and connected with updated successors
    for (std::set<StateId>::iterator it = usefulRoots.begin(); it != usefulRoots.end(); ++it) {
        if (determinizeMap[*it]) {
            minimizeState(determinizeMap[*it], minimizeMap);
        }
        else {
            minimizeState(*it, minimizeMap);
        }
    }

    verify(minimizeMap[network_.rootState] == network_.rootState);

    std::vector<u32> minimizeExitsMap;
    if (!keepRoots_) {
        minimizeExitsMap.resize(network_.exits.size(), Core::Type<u32>::max);
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
        // joint transitRoot is individual state specific, thus not update roots_ for general key
        for (StateId state = 1; state < oldNodeCount; ++state) {
            if (minimizeMap[state] == state) {
                minimizeExits(state, minimizeExitsMap);
            }
            else {
                network_.structure.clearOutputEdges(state);
            }
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
                if (orig == network_.rootState || network_.coarticulatedRootStates.count(orig)) {
                    network_.rootTransitDescriptions.insert(*it);
                }
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
        if (determinizeMap[state]) {
            determinizeMap[state] = minimizeMap[determinizeMap[state]];
        }
        else {
            determinizeMap[state] = minimizeMap[state];
        }
    }
    minimizeMap = determinizeMap;

    // cleanup also changes structure, need to update map accordingly
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

    // update necessary hashes w.r.t. minimizeMap
    predecessors_.swap(oldPredecessors);
    updateHashFromMap(minimizeMap, minimizeExitsMap);

    printStats("after minimization");
    return minimizeMap;
}

void MinimizedTreeBuilder::minimizeState(StateId state, std::vector<StateId>& minimizeMap) {
    verify(state < minimizeMap.size());
    if (minimizeMap[state]) {
        return;
    }

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

void MinimizedTreeBuilder::minimizeExits(StateId state, const std::vector<u32>& minimizeExitsMap) {
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

        if (successorExits.empty()) {
            return;
        }

        network_.structure.clearOutputEdges(state);
        for (std::set<StateId>::iterator it = successorStates.begin(); it != successorStates.end(); ++it) {
            network_.structure.addTargetToNode(state, *it);
        }
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
                for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(network_.exits[i->second].transitState); target; ++target) {
                    newRootSuccessors.insert(*target);
                }
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
                    if (network_.unpushedCoarticulatedRootStates.count(network_.exits[i->second].transitState)) {
                        network_.unpushedCoarticulatedRootStates.insert(exit.transitState);
                    }
                    if (network_.uncoarticulatedWordEndStates.count(network_.exits[i->second].transitState)) {
                        network_.uncoarticulatedWordEndStates.insert(exit.transitState);
                    }
                }
            }
        }
        successorExits.erase(range.first, range.second);
    }
}

void MinimizedTreeBuilder::mapSet(std::set<StateId>& set, const std::vector<StateId>& minimizeMap, bool force) {
    std::set<StateId> oldSet;
    oldSet.swap(set);
    for (std::set<StateId>::iterator it = oldSet.begin(); it != oldSet.end(); ++it) {
        if (*it >= minimizeMap.size()) {
            set.insert(*it);
        }
        else if (!minimizeMap[*it]) {
            verify(!force);
        }
        else {
            set.insert(minimizeMap[*it]);
        }
    }
}

// update hash structures according to minimizeMap (invalid ones are removed)
// should be ok for any number of minimize iterations
void MinimizedTreeBuilder::updateHashFromMap(const std::vector<StateId>& map, const std::vector<u32>& exitMap) {
    Core::HashMap<StateId, RootKey> tmpKeyHash;
    for (Core::HashMap<StateId, RootKey>::iterator iter = stateUniqueKeys_.begin(); iter != stateUniqueKeys_.end(); ++iter) {
        if (iter->first < map.size() && map[iter->first]) {
            tmpKeyHash.insert(std::make_pair(map[iter->first], iter->second));
        }
    }
    stateUniqueKeys_.swap(tmpKeyHash);

    mapCoarticulationJointHash(initialPhoneSuffix_, map, exitMap);
    mapCoarticulationJointHash(initialFinalPhoneSuffix_, map, exitMap);

    RootHash tmpRootHash;
    for (RootHash::const_iterator rootIt = roots_.begin(); rootIt != roots_.end(); ++rootIt) {
        if (rootIt->second < map.size() && map[rootIt->second]) {
            tmpRootHash.insert(std::make_pair(rootIt->first, map[rootIt->second]));
        }
    }
    roots_.swap(tmpRootHash);

    // exits are changed in cleanup
    exitHash_.clear();
    for (u32 idx = 0; idx != network_.exits.size(); ++idx) {
        exitHash_.insert(std::make_pair(network_.exits[idx], idx));
    }

    // PredecessorsHash still the FanIn/Out ones at this point (recorded in minimize())
    PredecessorsHash tmpPredHash;
    for (PredecessorsHash::iterator pIt = predecessors_.begin(); pIt != predecessors_.end(); ++pIt) {
        if (pIt->second >= map.size() || !map[pIt->second]) {
            continue;
        }
        const StatePredecessor& sp = pIt->first;
        std::set<StateId>       tmpSet;
        mapSuccessors(sp.successors, tmpSet, map, exitMap);
        if (!tmpSet.empty()) {
            StatePredecessor spNew(tmpSet, sp.desc, sp.isWordEnd);
            tmpPredHash.insert(std::make_pair(spNew, map[pIt->second]));
        }
    }
    predecessors_.swap(tmpPredHash);
}

inline void MinimizedTreeBuilder::mapCoarticulationJointHash(CoarticulationJointHash& hash, const std::vector<StateId>& map, const std::vector<u32>& exitMap) {
    CoarticulationJointHash tmpHash;
    for (CoarticulationJointHash::iterator iter = hash.begin(); iter != hash.end(); ++iter) {
        std::set<StateId> tmpSet;
        mapSuccessors(iter->second, tmpSet, map, exitMap);
        if (!tmpSet.empty()) {
            tmpHash.insert(std::make_pair(iter->first, tmpSet));
        }
    }
    hash.swap(tmpHash);
}

inline void MinimizedTreeBuilder::mapSuccessors(const std::set<StateId>& successors, std::set<StateId>& tmpSet, const std::vector<StateId>& map, const std::vector<u32>& exitMap) {
    for (std::set<StateId>::const_iterator sIt = successors.cbegin(); sIt != successors.cend(); ++sIt) {
        if (IS_LABEL(*sIt)) {
            u32 eIdx = LABEL_FROM_ID(*sIt);
            if (exitMap.empty() || eIdx >= exitMap.size()) {
                tmpSet.insert(*sIt);
            }
            else {
                tmpSet.insert(ID_FROM_LABEL(exitMap[eIdx]));
            }
        }
        else if (*sIt < map.size() && map[*sIt]) {
            tmpSet.insert(map[*sIt]);
        }
    }
}

// -------------------- CtcAedSharedBaseClassTreeBuilder --------------------

CtcAedSharedBaseClassTreeBuilder::CtcAedSharedBaseClassTreeBuilder(Core::Configuration          config,
                                                                   const Bliss::Lexicon&        lexicon,
                                                                   const Am::AcousticModel&     acousticModel,
                                                                   Search::PersistentStateTree& network)
        : AbstractTreeBuilder(config, lexicon, acousticModel, network) {}

StateId CtcAedSharedBaseClassTreeBuilder::createRoot() {
    return createState(StateTree::StateDesc(Search::StateTree::invalidAcousticModel, Am::TransitionModel::entryM1));
}

StateId CtcAedSharedBaseClassTreeBuilder::extendState(StateId predecessor, StateTree::StateDesc desc) {
    // Check if the successor already exists
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target) {
        if (!target.isLabel() && network_.structure.state(*target).stateDesc == desc) {
            return *target;
        }
    }

    // No matching successor found, extend
    StateId ret = createState(desc);
    network_.structure.addTargetToNode(predecessor, ret);
    return ret;
}

void CtcAedSharedBaseClassTreeBuilder::addTransition(StateId predecessor, StateId successor) {
    auto const& predecessorStateDesc = network_.structure.state(predecessor).stateDesc;
    auto const& successorStateDesc   = network_.structure.state(successor).stateDesc;

    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(predecessor); target; ++target) {
        if (!target.isLabel() && network_.structure.state(*target).stateDesc == successorStateDesc) {
            // The node is already a successor of the predecessor, so the transition already exists
            return;
        }
    }

    // The transition does not exists yet, add it
    network_.structure.addTargetToNode(predecessor, successor);
}

u32 CtcAedSharedBaseClassTreeBuilder::addExit(StateId state, StateId transitState, Bliss::LemmaPronunciation::Id pron) {
    PersistentStateTree::Exit exit;
    exit.transitState  = transitState;
    exit.pronunciation = pron;

    u32 exitIndex = createExit(exit);

    // Check if the exit is already a successor
    // This should only happen if the same lemma is contained multiple times in the lexicon
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(state); target; ++target) {
        if (target.isLabel() && target.label() == exitIndex) {
            return exitIndex;
        }
    }

    // The exit is not part of the successors yet, add it
    network_.structure.addOutputToNode(state, ID_FROM_LABEL(exitIndex));
    return exitIndex;
}

// -------------------- CtcTreeBuilder --------------------

const Core::ParameterBool CtcTreeBuilder::paramLabelLoop(
        "allow-label-loop",
        "allow label loops in the search tree",
        true);

const Core::ParameterBool CtcTreeBuilder::paramBlankLoop(
        "allow-blank-loop",
        "allow loops on the blank nodes in the search tree",
        true);

const Core::ParameterBool CtcTreeBuilder::paramForceBlank(
        "force-blank-between-repeated-labels",
        "require a blank label between two identical labels (only works if label-loops are disabled)",
        true);

CtcTreeBuilder::CtcTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize)
        : CtcAedSharedBaseClassTreeBuilder(config, lexicon, acousticModel, network),
          labelLoop_(paramLabelLoop(config)),
          blankLoop_(paramBlankLoop(config)),
          forceBlank_(paramForceBlank(config)) {
    auto iters = lexicon.phonemeInventory()->phonemes();
    for (auto it = iters.first; it != iters.second; ++it) {
        require(not(*it)->isContextDependent());  // Context dependent labels are not supported
    }

    // Set the StateDesc for blank
    blankAllophoneStateIndex_       = acousticModel_.blankAllophoneStateIndex();
    blankDesc_.acousticModel        = acousticModel_.emissionIndex(blankAllophoneStateIndex_);
    blankDesc_.transitionModelIndex = acousticModel_.stateTransitionIndex(blankAllophoneStateIndex_);
    require_lt(blankDesc_.transitionModelIndex, Core::Type<StateTree::StateDesc::TransitionModelIndex>::max);

    if (initialize) {
        verify(!network_.rootState);
        network_.ciRootState = network_.rootState = createRoot();

        // Create a special root for the word-boundary token if it exists in the lexicon
        if (lexicon.specialLemma("word-boundary") != nullptr) {
            wordBoundaryRoot_ = createRoot();
            network_.otherRootStates.insert(wordBoundaryRoot_);
        }
    }
}

std::unique_ptr<AbstractTreeBuilder> CtcTreeBuilder::newInstance(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize) {
    return std::unique_ptr<AbstractTreeBuilder>(new CtcTreeBuilder(config, lexicon, acousticModel, network));
}

void CtcTreeBuilder::build() {
    auto wordBoundaryLemma = lexicon_.specialLemma("word-boundary");
    if (wordBoundaryLemma != nullptr) {
        addWordBoundaryStates();
    }

    auto blankLemma   = lexicon_.specialLemma("blank");
    auto silenceLemma = lexicon_.specialLemma("silence");
    auto iters        = lexicon_.lemmaPronunciations();

    // Iterate over the lemmata and add them to the tree
    for (auto it = iters.first; it != iters.second; ++it) {
        if ((*it)->lemma() == wordBoundaryLemma) {
            // The wordBoundaryLemma should be a successor of the wordBoundaryRoot_
            // This is handled separately in addWordBoundaryStates()
            continue;
        }

        StateId lastState = extendPronunciation(network_.rootState, (*it)->pronunciation());

        if (wordBoundaryLemma != nullptr && (*it)->lemma() != blankLemma && (*it)->lemma() != silenceLemma) {
            // If existing, the wordBoundaryRoot_ should be the transit state for all word ends except blank and silence
            addExit(lastState, wordBoundaryRoot_, (*it)->id());
        }
        else {
            addExit(lastState, network_.rootState, (*it)->id());
        }
    }
}

StateId CtcTreeBuilder::extendPronunciation(StateId startState, Bliss::Pronunciation const* pron) {
    require(pron != nullptr);

    StateId currentState      = startState;
    StateId prevNonBlankState = invalidTreeNodeIndex;

    for (u32 i = 0u; i < pron->length(); i++) {
        Bliss::Phoneme::Id phoneme = (*pron)[i];

        u32 boundary = 0u;
        if (i == 0) {
            boundary |= Am::Allophone::isInitialPhone;
        }
        if ((i + 1) == pron->length()) {
            boundary |= Am::Allophone::isFinalPhone;
        }

        Bliss::ContextPhonology::SemiContext history, future;
        const Am::Allophone*                 allophone        = acousticModel_.allophoneAlphabet()->allophone(Am::Allophone(Bliss::ContextPhonology::PhonemeInContext(phoneme, history, future), boundary));
        const Am::ClassicHmmTopology*        hmmTopology      = acousticModel_.hmmTopology(phoneme);
        const bool                           allophoneIsBlank = acousticModel_.allophoneStateAlphabet()->index(allophone, 0, false) == blankAllophoneStateIndex_;

        for (u32 phoneState = 0; phoneState < hmmTopology->nPhoneStates(); ++phoneState) {
            Am::AllophoneState   alloState = acousticModel_.allophoneStateAlphabet()->allophoneState(allophone, phoneState);
            StateTree::StateDesc desc;
            desc.acousticModel = acousticModel_.emissionIndex(alloState);  // state-tying look-up

            for (u32 subState = 0; subState < hmmTopology->nSubStates(); ++subState) {
                desc.transitionModelIndex = acousticModel_.stateTransitionIndex(alloState, subState);
                verify(desc.transitionModelIndex < Core::Type<StateTree::StateDesc::TransitionModelIndex>::max);

                // Add new (non-blank) state
                currentState = extendState(currentState, desc);

                if (labelLoop_) {
                    // Add loop for this state
                    addTransition(currentState, currentState);
                }

                bool label_repetition = prevNonBlankState != currentState and prevNonBlankState != invalidTreeNodeIndex and network_.structure.state(prevNonBlankState).stateDesc == network_.structure.state(currentState).stateDesc;
                if (prevNonBlankState != invalidTreeNodeIndex and not(label_repetition and forceBlank_)) {
                    // Add transition from previous non-blank state to this state, allowing to skip the blank state in-between these two
                    // If we want to enforce blank between repeated labels, don't add a transition between two distinct states of equal description
                    addTransition(prevNonBlankState, currentState);
                }
                prevNonBlankState = currentState;

                bool isLastStateInLemma = ((phoneState + 1) == hmmTopology->nPhoneStates()) and ((subState + 1) == hmmTopology->nSubStates()) and (boundary & Am::Allophone::isFinalPhone);
                if (not allophoneIsBlank and not isLastStateInLemma) {
                    // Add blank state after the newly created state
                    currentState = extendState(currentState, blankDesc_);

                    if (blankLoop_) {
                        // Add loop for this blank state
                        addTransition(currentState, currentState);
                    }
                }
            }
        }
    }

    return currentState;
}

void CtcTreeBuilder::addWordBoundaryStates() {
    Bliss::Lemma const* wordBoundaryLemma = lexicon_.specialLemma("word-boundary");
    if (wordBoundaryLemma == nullptr) {
        return;
    }

    // Add the word-boundary to the tree, starting from the wordBoundaryRoot_
    // If the word-boundary has several pronunciation, only the first one is considered
    auto prons = wordBoundaryLemma->pronunciations();

    StateId wordBoundaryEnd = extendPronunciation(wordBoundaryRoot_, (prons.first)->pronunciation());
    require(wordBoundaryEnd != 0);

    Bliss::LemmaPronunciation const* wordBoundaryPronLemma = prons.first;
    require(wordBoundaryPronLemma != nullptr);

    // The "normal" root is the transition state from the word-boundary token, such that a new word can be started afterwards
    addExit(wordBoundaryEnd, network_.rootState, wordBoundaryPronLemma->id());

    std::vector<StateId> wordBoundaryLemmaStartStates;
    for (HMMStateNetwork::SuccessorIterator target = network_.structure.successors(wordBoundaryRoot_); target; ++target) {
        if (!target.isLabel()) {
            wordBoundaryLemmaStartStates.push_back(*target);
        }
    }
    // Add optional blank before the word-boundary lemma
    StateId blankBefore = extendState(wordBoundaryRoot_, blankDesc_);
    for (StateId wbs : wordBoundaryLemmaStartStates) {
        network_.structure.addTargetToNode(blankBefore, wbs);
    }

    if (blankLoop_) {
        // Add loop for this blank state
        addTransition(blankBefore, blankBefore);
    }
}

// -------------------- RnaTreeBuilder --------------------

const Core::ParameterBool RnaTreeBuilder::paramLabelLoop(
        "allow-label-loop",
        "allow label loops in the search tree",
        false);

const Core::ParameterBool RnaTreeBuilder::paramForceBlank(
        "force-blank-between-repeated-labels",
        "require a blank label between two identical labels (only works if label-loops are disabled)",
        false);

RnaTreeBuilder::RnaTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize)
        : CtcTreeBuilder(config, lexicon, acousticModel, network, initialize) {
    this->labelLoop_  = paramLabelLoop(config);
    this->forceBlank_ = paramForceBlank(config);
}

// -------------------- AedTreeBuilder --------------------

AedTreeBuilder::AedTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize)
        : CtcAedSharedBaseClassTreeBuilder(config, lexicon, acousticModel, network) {
    auto iters = lexicon.phonemeInventory()->phonemes();
    for (auto it = iters.first; it != iters.second; ++it) {
        require(not(*it)->isContextDependent());  // Context dependent labels are not supported
    }

    if (initialize) {
        verify(!network_.rootState);
        network_.ciRootState = network_.rootState = createRoot();

        // Create a special root for the word-boundary token if it exists in the lexicon
        if (lexicon.specialLemma("word-boundary") != nullptr) {
            wordBoundaryRoot_ = createRoot();
            network_.otherRootStates.insert(wordBoundaryRoot_);
        }
    }
}

std::unique_ptr<AbstractTreeBuilder> AedTreeBuilder::newInstance(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize) {
    return std::unique_ptr<AbstractTreeBuilder>(new AedTreeBuilder(config, lexicon, acousticModel, network));
}

void AedTreeBuilder::build() {
    auto wordBoundaryLemma = lexicon_.specialLemma("word-boundary");
    if (wordBoundaryLemma != nullptr) {
        addWordBoundaryStates();
    }

    auto sentenceEndLemma = lexicon_.specialLemma("sentence-end");
    if (!sentenceEndLemma) {
        sentenceEndLemma = lexicon_.specialLemma("sentence-boundary");
    }
    require(sentenceEndLemma);
    auto silenceLemma = lexicon_.specialLemma("silence");
    auto iters        = lexicon_.lemmaPronunciations();

    // Iterate over the lemmata and add them to the tree
    for (auto it = iters.first; it != iters.second; ++it) {
        if ((*it)->lemma() == wordBoundaryLemma) {
            // The wordBoundaryLemma should be a successor of the wordBoundaryRoot_
            // This is handled separately in addWordBoundaryStates()
            continue;
        }

        StateId lastState = extendPronunciation(network_.rootState, (*it)->pronunciation());

        if (wordBoundaryLemma != nullptr && (*it)->lemma() != sentenceEndLemma && (*it)->lemma() != silenceLemma) {
            // If existing, the wordBoundaryRoot_ should be the transit state for all word ends except sentence-end and silence
            addExit(lastState, wordBoundaryRoot_, (*it)->id());
        }
        else {
            addExit(lastState, network_.rootState, (*it)->id());
        }
    }
}

StateId AedTreeBuilder::extendPronunciation(StateId startState, Bliss::Pronunciation const* pron) {
    require(pron != nullptr);
    StateId currentState = startState;

    for (u32 i = 0u; i < pron->length(); i++) {
        Bliss::Phoneme::Id phoneme = (*pron)[i];

        u32 boundary = 0u;
        if (i == 0) {
            boundary |= Am::Allophone::isInitialPhone;
        }
        if ((i + 1) == pron->length()) {
            boundary |= Am::Allophone::isFinalPhone;
        }

        Bliss::ContextPhonology::SemiContext history, future;
        const Am::Allophone*                 allophone   = acousticModel_.allophoneAlphabet()->allophone(Am::Allophone(Bliss::ContextPhonology::PhonemeInContext(phoneme, history, future), boundary));
        const Am::ClassicHmmTopology*        hmmTopology = acousticModel_.hmmTopology(phoneme);

        for (u32 phoneState = 0; phoneState < hmmTopology->nPhoneStates(); ++phoneState) {
            Am::AllophoneState   alloState = acousticModel_.allophoneStateAlphabet()->allophoneState(allophone, phoneState);
            StateTree::StateDesc desc;
            desc.acousticModel = acousticModel_.emissionIndex(alloState);  // state-tying look-up

            for (u32 subState = 0; subState < hmmTopology->nSubStates(); ++subState) {
                desc.transitionModelIndex = acousticModel_.stateTransitionIndex(alloState, subState);
                verify(desc.transitionModelIndex < Core::Type<StateTree::StateDesc::TransitionModelIndex>::max);

                // Add new state
                currentState = extendState(currentState, desc);
            }
        }
    }

    return currentState;
}

void AedTreeBuilder::addWordBoundaryStates() {
    Bliss::Lemma const* wordBoundaryLemma = lexicon_.specialLemma("word-boundary");
    if (wordBoundaryLemma == nullptr) {
        return;
    }

    // Add the word-boundary to the tree, starting from the wordBoundaryRoot_
    // If the word-boundary has several pronunciation, only the first one is considered
    auto prons = wordBoundaryLemma->pronunciations();

    StateId wordBoundaryEnd = extendPronunciation(wordBoundaryRoot_, (prons.first)->pronunciation());
    require(wordBoundaryEnd != 0);

    Bliss::LemmaPronunciation const* wordBoundaryPronLemma = prons.first;
    require(wordBoundaryPronLemma != nullptr);

    // The "normal" root is the transition state from the word-boundary token, such that a new word can be started afterwards
    addExit(wordBoundaryEnd, network_.rootState, wordBoundaryPronLemma->id());
}