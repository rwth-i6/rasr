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
 *
 *  author: Wei Zhou
 */

#include "LabelTree.hh"
#include <Am/ClassicAcousticModel.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Search/AdvancedTreeSearch/PersistentStateTree.hh>
#include <Search/AdvancedTreeSearch/TreeBuilder.hh>
#include "MappedArchive.hh"

using namespace Search;

// image version
static u32 formatVersion = 10;
static const Core::ParameterString paramCacheArchive(
  "cache-archive",
  "cache archive in which the label-tree network should be cached",
  "global-cache" );


const Core::Choice LabelTree::labelUnitChoice(
  "hmm",     labelHMM,
  "phoneme",  labelPhoneme,
  "subword",  labelSubword, // maybe redundant as the same can be realized by mis-using phoneme
  "word",     labelWord,
  Core::Choice::endMark());

const Core::ParameterChoice LabelTree::paramLabelUnit(
  "label-unit", &labelUnitChoice,
  "unit of labels (has to match with label scorer)",
  labelPhoneme);
  
// maybe a skip list for more felxibility ?
const Core::ParameterBool LabelTree::paramSkipUnknownLabel(
  "skip-unknown-label",
  "skip unknown labels in tree construction",
  true );

const Core::ParameterBool LabelTree::paramSkipSilence(
  "skip-silence",
  "if lexicon has silence, skip it in tree construction",
  false );

const Core::ParameterBool LabelTree::paramUseTransitionPenalty(
  "use-transition-penalty",
  "whether to use additional transition penalty between labels",
  false );


LabelTree::LabelTree(const Core::Configuration& config, 
                     Core::Ref<const Am::AcousticModel> acousticModel, 
                     Bliss::LexiconRef lexicon,
                     Core::Ref<Nn::LabelScorer> labelScorer) :
    config_(config),
    acousticModel_(acousticModel),
    lexicon_(lexicon),
    labelScorer_(labelScorer),
    archive_(paramCacheArchive(config)),
    labelUnit_((LabelUnitType)paramLabelUnit(config)),
    root_(0),
    numNodes_(0),
    hasPronunciation_(false),
    skipUnknownLabel_(paramSkipUnknownLabel(config)),
    skipSilence_(paramSkipSilence(config)),
    silence_(Core::Type<NodeId>::max),
    useTransitionPenalty_(paramUseTransitionPenalty(config)),
    useExpansionPenalty_(true) {

  if (lexicon_ && labelScorer_) {
    dependencies_.add("lexicon", lexicon_->getDependency());
    dependencies_.add("label-scorer", labelScorer_->getDependency());
  }
  clear(); 
}

void LabelTree::clear() {
  empty_.clear();
  node2successors_.clear();
  exits_.clear();
  endExitIdx_ = Core::Type<u32>::max;
  endNodeId_ = Core::Type<u32>::max;
  hasExit_.clear();
  node2exits_.clear();
  node2labelIndex_.clear(); 
}

void LabelTree::build() {
  switch (labelUnit_) {
    case labelHMM:     buildHMMLabelTree(); break;
    case labelPhoneme: buildPhonemeLabelTree(); break;
    case labelSubword: buildSubwordLabelTree(); break;
    case labelWord:    buildWordLabelTree(); break;
    default:
      Core::Application::us()->criticalError("unknwon label-unit type");
  }

  // finalize
  numNodes_ = node2labelIndex_.size(); // including root
  if (numNodes_ <= 1)
    Core::Application::us()->criticalError() << "no labels in the tree";
  else
    Core::Application::us()->log() << "LabelTree successfully built  " 
                                   << numNodes_ << " label nodes and "
                                   << exits_.size() << " exits";
  makeNodeExitFlag();
}

// Special case: convert from persistent state tree (backward compatibility for HMM models)
//               allophone-based construction and state-tying for classID
// TODO additional handling maybe needed for the following
// - no skip for now !
// - pushed boundary need to be corrected afterwards (so far not, only considered in sentence end)
void LabelTree::buildHMMLabelTree() { 
  hasPronunciation_ = true;
  if (skipSilence_) {
    // this is a bit pointless
    Core::Application::us()->warning() << "no skip-silence for HMM state tree";
    skipSilence_ = false;
  }

  Core::Configuration stConfig(config_, "hmm-state-tree");
  // determinized and minimized HMM state tree
  PersistentStateTree stateTree(stConfig, acousticModel_, lexicon_);
  const Core::ParameterBool paramBuildMinimizedTreeFromScratch("build-minimized-network-from-scratch", "", true);
  int transformation = paramBuildMinimizedTreeFromScratch(stConfig) ? 32 : 0;
  if(!stateTree.read(transformation)) {
    TreeBuilder builder(stConfig, *lexicon_, *acousticModel_, stateTree);
    builder.build();
  }

  u32 nClasses = acousticModel_->nEmissions();
  Nn::ClassLabelWrapper labelWrapper(Core::Configuration(stConfig, "class-labels"), nClasses);

  Core::Application::us()->log() << "converting from persistent HMM state tree to label tree"
                                 << " (discard skip transition)"
                                 << " number of states " << stateTree.structure.stateCount()
                                 << " with number of class labels " << nClasses;
  // stateId to nodeId mapping 
  std::vector<NodeId> state2nodeIdMap(stateTree.structure.stateCount(), Core::Type<NodeId>::max);
  state2nodeIdMap[stateTree.rootState] = root_;
  
  std::stack<StateId> stateStack;
  stateStack.push(stateTree.rootState);

  // only forward transitions need to be considered here
  // cannot use exact same stateId used as nodeId due to output treated as node as well (waste !)
  while (!stateStack.empty()) {
    StateId sId = stateStack.top(); 
    stateStack.pop();
    NodeId nId = state2nodeIdMap[sId];
    verify(nId != Core::Type<NodeId>::max);
 
    const HMMState& state = stateTree.structure.state(sId);
    for(HMMStateNetwork::SuccessorIterator successorIt = stateTree.structure.successors(state); successorIt; ++successorIt) {
      if (successorIt.isLabel()) {
        const PersistentStateTree::Exit& e = stateTree.exits[successorIt.label()];
        const Bliss::LemmaPronunciation* p = lexicon_->lemmaPronunciation(e.pronunciation);
        NodeId transitNode = state2nodeIdMap[e.transitState];
        if (transitNode == Core::Type<NodeId>::max) {
          // transitRoot has no model index 
          transitNode = createNewLabel(Core::Type<Index>::max);
          state2nodeIdMap[e.transitState] = transitNode;
          stateStack.push(e.transitState);
        }
        addExitToNode(nId, p, p->lemma(), transitNode);
      } else {
        StateId sucId = *successorIt;
        Index sucLabelIndex = labelWrapper.getOutputIndexFromClassIndex(stateTree.structure.state(sucId).stateDesc.acousticModel);
        NodeId sucNodeId = state2nodeIdMap[sucId];
        if (sucNodeId == Core::Type<NodeId>::max) {
          sucNodeId = extendLabel(nId, sucLabelIndex);
          state2nodeIdMap[sucId] = sucNodeId;
          stateStack.push(sucId);
        } else {
          verify(node2labelIndex_[sucNodeId] == sucLabelIndex);
          node2successors_[nId].push_back(sucNodeId); // link only 
        }
      }
    } 
  }

  if (!stateTree.uncoarticulatedWordEndStates.empty()) {
    for (std::set<StateId>::const_iterator iter = stateTree.uncoarticulatedWordEndStates.begin(); iter != stateTree.uncoarticulatedWordEndStates.end(); ++iter) {
      verify(state2nodeIdMap[*iter] != Core::Type<NodeId>::max);
      forceEndNodes_.insert(state2nodeIdMap[*iter]);
    }
    forceEndNodes_.insert(root_);
  } 

  // map TDPs to transition penalties
  if (useTransitionPenalty_) {
    Core::Application::us()->log() << "map tdps to transition penalties";
    // transition index mapping
    std::vector<u32> transitionMap(acousticModel_->nStateTransitions());
    std::vector<u32> reverseMap; // only take the 3-states-HMM
    for (u32 tIdx = 0; tIdx < acousticModel_->nStateTransitions(); ++tIdx) {
      u32 tpIdx;
      if (tIdx == Am::TransitionModel::entryM1 || tIdx == Am::TransitionModel::entryM2)
        tpIdx = 0; // root
      else if (tIdx == Am::TransitionModel::phone0 || tIdx == Am::TransitionModel::phone1)
        tpIdx = 1; // default
      else if (tIdx == Am::TransitionModel::silence)
        tpIdx = 2; // special
      else
        tpIdx = 3; // others (more needed ?)
      transitionMap[tIdx] = tpIdx;
      if (tpIdx >= reverseMap.size())
        reverseMap.resize(tpIdx + 1, Core::Type<u32>::max);
      if (reverseMap[tpIdx] == Core::Type<u32>::max)
        reverseMap[tpIdx] = tIdx;
    }
    transitions_.clear();
    // score mapping (scale already included)
    for (u32 tpIdx = 0; tpIdx < reverseMap.size(); ++tpIdx) {
      const Am::StateTransitionModel& tdp = *(acousticModel_->stateTransition(reverseMap[tpIdx]));
      transitions_.emplace_back(tdp[Am::StateTransitionModel::loop], tdp[Am::StateTransitionModel::forward], tdp[Am::StateTransitionModel::exit]);
    }
    // label to transition mapping
    for (StateId sId = 0; sId < stateTree.structure.stateCount(); ++sId) {
      NodeId nId = state2nodeIdMap[sId];
      if (nId == Core::Type<NodeId>::max)
        continue;
      u32 transitionModelIdx = stateTree.structure.state(sId).stateDesc.transitionModelIndex;
      verify(transitionModelIdx < transitionMap.size());
      u32 transitionPenaltyIdx = transitionMap[transitionModelIdx];
      // only record non-default transitions
      if (transitionPenaltyIdx != 1)
        node2transition_[nId] = transitionPenaltyIdx;
    }
  }
}

void LabelTree::buildPhonemeLabelTree() {
  hasPronunciation_ = true;
  // label (string) to index mapping
  const Nn::LabelIndexMap& labelIndexMap = labelScorer_->getLabelIndexMap();
  // special labels to skip in tree construction, explicitly handled in search space
  std::unordered_set<Index> skipIdxs;
  skipIdxs.insert(labelScorer_->getStartLabelIndex());
  skipIdxs.insert(labelScorer_->getEndLabelIndex());
  skipIdxs.insert(labelScorer_->getBlankLabelIndex());

  // map phoneme id (start from 1) to model index
  u32 nPhonemes = lexicon_->phonemeInventory()->nPhonemes();
  std::vector<Index> phonemeId2LabelIndex(nPhonemes+1, Core::Type<Index>::max);
  for (u32 id = 1; id <= nPhonemes; ++id) {
    std::string p = lexicon_->phonemeInventory()->phoneme(id)->symbol();
    if (labelIndexMap.count(p) > 0) {
      phonemeId2LabelIndex[id] = labelIndexMap.at(p);
    } else if (skipUnknownLabel_) {
      phonemeId2LabelIndex[id] = Core::Type<Index>::max;
    } else { // map to unknown label
      Index unkIdx = labelScorer_->getUnknownLabelIndex();
      if (unkIdx == Core::Type<Index>::max)
        Core::Application::us()->criticalError() << "invalid phoneme " << p << " and no unknown label defined";
      else
        Core::Application::us()->warning() << "phoneme " << p << " is mapped to unknown label";
      phonemeId2LabelIndex[id] = unkIdx;
    }
  }

  // optional skip silence
  const Bliss::Lemma* silenceLemma = NULL;
  if (skipSilence_)
    silenceLemma = lexicon_->specialLemma("silence");

  // loop over all pronunciations and construct label tree
  typedef Bliss::Lexicon::PronunciationIterator PronIter;
  typedef Bliss::Pronunciation::LemmaIterator LPIter;
  std::pair<PronIter, PronIter> prons = lexicon_->pronunciations();
  for (PronIter pronIt = prons.first; pronIt != prons.second; ++pronIt) {
    const Bliss::Pronunciation& pron(**pronIt);
    u32 pronLength = pron.length();
    if(pronLength == 0)
      continue;

    bool skip = false; 
    u32 pIdx = 0;
    for (; pIdx < pronLength; ++pIdx) {
      Index labelIndex = phonemeId2LabelIndex[pron[pIdx]];
      skip = (skipUnknownLabel_ && labelIndex == Core::Type<Index>::max) || skipIdxs.count(labelIndex) > 0;
      if (skip)
        break;
    }
    if (skip) {
      Core::Application::us()->warning() << "pronunciation " << pron.format(lexicon_->phonemeInventory())
                                         << " is skipped due to unknown/invalid phoneme label " 
                                         << lexicon_->phonemeInventory()->phoneme(pron[pIdx])->symbol();;
      continue;
    }

    // misleading names, this is acutaully lemma pronunciations
    std::pair<LPIter, LPIter> lps = pron.lemmas();
    if (skipSilence_ && pron.nLemmas() == 1)
      if (lps.first->lemma() == silenceLemma) 
        continue;

    NodeId currentNodeId = root_; 
    for (u32 pIdx = 0; pIdx < pronLength; ++pIdx)
      currentNodeId = extendLabel(currentNodeId, phonemeId2LabelIndex[pron[pIdx]]);

    for (LPIter lpIt = lps.first; lpIt != lps.second; ++lpIt) {
      const Bliss::LemmaPronunciation *lp = lpIt;
      const Bliss::Lemma* l = lp->lemma();
      if (skipSilence_ && l == silenceLemma)
        continue;
      addExitToNode(currentNodeId, lp, l, root_);
    }
  }

  if (useTransitionPenalty_)
    makeTransitionPenalty();
}

// white space separated subwords in one orthography
void LabelTree::buildSubwordLabelTree() { 
  const Nn::LabelIndexMap& labelIndexMap = labelScorer_->getLabelIndexMap();
  std::unordered_set<Index> skipIdxs;
  skipIdxs.insert(labelScorer_->getStartLabelIndex());
  skipIdxs.insert(labelScorer_->getEndLabelIndex());
  skipIdxs.insert(labelScorer_->getBlankLabelIndex());
  
  typedef Bliss::Lexicon::LemmaIterator LemmaIter;
  std::pair<LemmaIter, LemmaIter> lemmas = lexicon_->lemmas();
  for (LemmaIter lmIt = lemmas.first; lmIt != lemmas.second; ++lmIt) {
    const Bliss::Lemma* lemma(*lmIt);
    if (skipSilence_ && lemma == lexicon_->specialLemma("silence"))
      continue; // may skip silence
    if (lemma->nOrthographicForms() == 0)
      continue; // skip empty orth

    // only preferredOrthographicForm since anyway just this is output to result 
    // put different subwords combination of the same word to separate lemmas since they 
    // represent different acoustics now
    std::string orth = lemma->preferredOrthographicForm();
    // verify lemma id can get corresponding lemma for image
    verify((lmIt - lemmas.first) == (u32)lemma->id());

    // separate white space and map to model label index
    std::vector<Index> labelIndex;
    std::istringstream iss(orth); 
    std::string subword;
    bool skip = false; 
    while (iss >> subword) {
      if (labelIndexMap.count(subword) > 0) {
        labelIndex.push_back(labelIndexMap.at(subword));
      } else if (skipUnknownLabel_) {
        skip = true; 
        break;
      } else {
        Index unkIndex = labelScorer_->getUnknownLabelIndex();
        if (unkIndex == Core::Type<Index>::max)
          Core::Application::us()->criticalError() << "invalid subword " << subword
                                                   << " of lemma " << orth << " and no unknown label defined";
        else
          Core::Application::us()->warning() << "subword "<< subword << " of lemma " << orth << " is mapped to unknown label";
        labelIndex.push_back(unkIndex);
      }
      if (skipIdxs.count(labelIndex.back()) > 0) {
        skip = true; 
        break;
      }
    }
    if (skip) {
      Core::Application::us()->warning() << "lemma " << orth << " is skipped due to unknown/invalid subword label " << subword;
      continue;  
    }
    
    NodeId currentNodeId = root_;
    for (u32 idx = 0; idx < labelIndex.size(); ++idx)
      currentNodeId = extendLabel(currentNodeId, labelIndex[idx]);
    addExitToNode(currentNodeId, 0, lemma, root_);
  }

  if (useTransitionPenalty_)
    makeTransitionPenalty();
}

// whole orthography as one label
void LabelTree::buildWordLabelTree() {
  const Nn::LabelIndexMap& labelIndexMap = labelScorer_->getLabelIndexMap();
  std::unordered_set<Index> skipIdxs;
  skipIdxs.insert(labelScorer_->getStartLabelIndex());
  skipIdxs.insert(labelScorer_->getEndLabelIndex());
  skipIdxs.insert(labelScorer_->getBlankLabelIndex());

  typedef Bliss::Lexicon::LemmaIterator LemmaIter;
  std::pair<LemmaIter, LemmaIter> lemmas = lexicon_->lemmas();
  for (LemmaIter lmIt = lemmas.first; lmIt != lemmas.second; ++lmIt) {
    // only preferredOrthographicForm since anyway just this is output to result
    const Bliss::Lemma* lemma(*lmIt);
    if (skipSilence_ && lemma == lexicon_->specialLemma("silence"))
      continue; // may skip silence
    if (lemma->nOrthographicForms() == 0)
      continue; // skip empty orth

    std::string orth = lemma->preferredOrthographicForm();
    verify((lmIt - lemmas.first) == (u32)lemma->id());

    Index labelIndex;    
    if (labelIndexMap.count(orth) > 0) {
      labelIndex = labelIndexMap.at(orth);
    } else if (skipUnknownLabel_) {
      Core::Application::us()->warning() << "lemma " << orth << " is skipped due to unknown label";
      continue;
    } else { // acoustically unknown is unlikely
      labelIndex = labelScorer_->getUnknownLabelIndex();
      if (labelIndex == Core::Type<Index>::max)
        Core::Application::us()->criticalError() << "invalid lemma " << orth << " and no unknown label defined";
      else
        Core::Application::us()->warning() << "lemma " << orth << " is mapped to unknown label";
    }

    if ( skipIdxs.count(labelIndex) > 0) {
      Core::Application::us()->log() << "lemma " << orth << " is skipped in label tree construction";
      continue;
    }

    NodeId currentNodeId = extendLabel(root_, labelIndex);
    addExitToNode(currentNodeId, 0, lemma, root_);
  }

  if (useTransitionPenalty_)
    makeTransitionPenalty();
}

NodeId LabelTree::extendLabel(NodeId source, Index label) {
  // search existing successors
  std::vector<NodeId>& successors = node2successors_[source];
  for (std::vector<NodeId>::iterator iter = successors.begin(); iter != successors.end(); ++iter)
    if (node2labelIndex_[*iter] == label)
      return *iter; 
  // create new label node
  NodeId newNodeId = createNewLabel(label);
  successors.push_back(newNodeId);
  return newNodeId; 
}

NodeId LabelTree::createNewLabel(Index label) {
  // root = 0 and has no model index
  if (node2labelIndex_.empty())
    node2labelIndex_.push_back(Core::Type<Index>::max);

  NodeId newNodeId = node2labelIndex_.size();
  node2labelIndex_.push_back(label);
  return newNodeId;
}

u32 LabelTree::addExitToNode(NodeId nId, const Bliss::LemmaPronunciation* pronunciation, const Bliss::Lemma* lemma, NodeId transitRoot) {
  // search existing exits (no global hashing, only local check)
  std::vector<u32>& exits = node2exits_[nId];
  for (std::vector<u32>::const_iterator iter = exits.begin(); iter != exits.end(); ++iter) {
    verify(*iter < exits_.size());
    const Exit& e = exits_[*iter];
    if (e.pronunciation == pronunciation && e.lemma == lemma)
      return *iter;
  }
  // last (mostly only) label before silence lemma exit
  if (!skipSilence_ && lemma == lexicon_->specialLemma("silence")) {
    // there should be only one exit for silence
    verify(silence_ == Core::Type<NodeId>::max || silence_ == nId);
    silence_ = nId;
  }
  // create new exit
  u32 exitIdx = createNewExit(pronunciation, lemma, transitRoot);
  exits.push_back(exitIdx);
  return exitIdx;
}

u32 LabelTree::createNewExit(const Bliss::LemmaPronunciation* pronunciation, const Bliss::Lemma* lemma, NodeId transitRoot) {
  u32 exitIdx = exits_.size();
  exits_.emplace_back(pronunciation, lemma, transitRoot);
  return exitIdx;
}

// for search simplicity (kind of waste to put it in image, thus always constructed)
void LabelTree::makeNodeExitFlag() {
  hasExit_.resize(numNodes_, false);
  for (Node2Exits::const_iterator iter = node2exits_.begin(); iter != node2exits_.end(); ++iter)
    hasExit_[iter->first] = true;
}

void LabelTree::activateLoop() {
  // transit root HMM state no loop
  std::unordered_set<NodeId> roots;
  if (labelUnit_ == labelHMM)
    for (const Exit& e : exits_)
      roots.insert(e.transitRoot);

  for (NodeId nId = root_+1; nId < numNodes_; ++nId) {
    if (roots.count(nId) > 0)
      continue;
    std::vector<NodeId>& successors = node2successors_[nId];
    successors.push_back(nId);
  }
}

void LabelTree::activateEndLabel(Index endIdx, const Bliss::Lemma* sentEndLemma, bool useNullLemma) {
  verify(hasExit_.size() == numNodes_ && node2labelIndex_.size() == numNodes_);
  // extend end label from root
  endNodeId_ = extendLabel(root_, endIdx);
  endExitIdx_ = Core::Type<u32>::max;
  if (endNodeId_ < numNodes_) {
    // need to adjust existing one
    const std::vector<u32>& exitIds = getExits(endNodeId_);
    for (std::vector<u32>::const_iterator eIt = exitIds.begin(); eIt != exitIds.end(); ++eIt)
      if (exits_[*eIt].lemma == sentEndLemma)
        endExitIdx_ = *eIt;
    hasExit_[endNodeId_] = true;
  } else if (endNodeId_ == numNodes_) {
    // added new label node
    ++numNodes_;
    hasExit_.push_back(true);
  } else 
    Core::Application::us()->criticalError() << "something went wrong in activating end label";

  if (endExitIdx_ == Core::Type<u32>::max) {
    if (useNullLemma)
      endExitIdx_ = addExitToNode(endNodeId_, 0, 0, root_);
    else
      endExitIdx_ = addExitToNode(endNodeId_, 0, sentEndLemma, root_);
    verify(endExitIdx_ == exits_.size() - 1); // should be new
  } else {
    if (useNullLemma)
      exits_[endExitIdx_].lemma = 0;
    else
      exits_[endExitIdx_].lemma = sentEndLemma;
  }
}

// for non-HMM models (otherwise mapped from TDPs)
void LabelTree::makeTransitionPenalty() {
  Core::Application::us()->log() << "make transition penalties";
  Core::ParameterFloat paramLoop("loop", "negative logarithm of probability for loop transition", 3.0),         
                       paramForward("forward", "negative logarithm of probability for forward transition", 0.0),
                       paramExit("exit", "negative logarithm of probability for word end transition", 0.0);
  Core::ParameterFloat paramScale("transition-scale", "scale for transition penalty", 1.0);
  Score scale = paramScale(config_);
  // root:0 ; default:1; special:2
  // special-transition-1,2,3 ... to further speparate special labels ? likely not needed so far
  std::vector<std::string> transitionTypes({"root-transition","default-transition","special-transition"});

  transitions_.clear();
  for (u32 idx = 0; idx < transitionTypes.size(); ++idx) {
    Core::Configuration cfg(config_, transitionTypes[idx]);
    Score loop = paramLoop(cfg), forward = paramForward(cfg), exit = paramExit(cfg);
    transitions_.emplace_back(loop * scale, forward * scale, exit * scale);
    Core::Application::us()->log() << "transition type " << transitionTypes[idx] << " with penalty: "
                                   << "forward=" << forward << " loop=" << loop << " exit=" << exit;
  }

  Core::ParameterStringVector paramSpecialLabels("special-transition-labels", "labels with special transition", ",");
  std::vector<std::string> specialTransitionLabels = paramSpecialLabels(config_);
  const Nn::LabelIndexMap& labelIndexMap = labelScorer_->getLabelIndexMap();
  std::unordered_set<Index> specialIndex;
  for (u32 idx = 0; idx < specialTransitionLabels.size(); ++idx) {
    Nn::LabelIndexMap::const_iterator iter = labelIndexMap.find(specialTransitionLabels[idx]);
    verify(iter != labelIndexMap.end());
    specialIndex.insert(iter->second);
    Core::Application::us()->log() << "special transition label " << specialTransitionLabels[idx] 
                                   << " with model index " << iter->second;
  }

  node2transition_[root_] = 0;
  for (NodeId nId = root_+1; nId < node2labelIndex_.size(); ++nId)
    if (specialIndex.count(node2labelIndex_[nId]) > 0)
      node2transition_[nId] = 2;
}

void LabelTree::setExpansionPenalty(bool allowLoop) {
  useExpansionPenalty_ = useTransitionPenalty_;
  if (useExpansionPenalty_) {
    // root has no loop
    useExpansionPenalty_ = transitions_.front().forward != 0;
    for (u32 tIdx = 1; tIdx < transitions_.size(); ++tIdx) {
      // default and special transitions
      if (transitions_[tIdx].forward != 0 || (allowLoop && transitions_[tIdx].loop != 0)) {
        // either forward or loop has non-zero penalty
        useExpansionPenalty_ = true;
        break;
      }
    }
  }
}

// image I/O 
u32 LabelTree::getChecksum(bool configSetting) const {
  if ( configSetting )
    return dependencies_.getChecksum() + (u32)labelUnit_ + (u32)skipUnknownLabel_ + (u32)useTransitionPenalty_;
  else
    return numNodes_ + (u32)hasPronunciation_ + node2successors_.size() + exits_.size() + node2exits_.size() + node2labelIndex_.size() + forceEndNodes_.size() + transitions_.size() + node2transition_.size();
}

bool LabelTree::write() {
  if(archive_.empty())
    return false;
  Core::MappedArchiveWriter out = Core::Application::us()->getCacheArchiveWriter(archive_, archiveEntry());
  if(!out.good())
    return false;
  Core::Application::us()->log() << "writing label tree into " << archive_;
  
  out << formatVersion << getChecksum(true);
  out << numNodes_ << hasPronunciation_;
  out << silence_;
  out << node2successors_;
  out << node2labelIndex_;
  out << node2exits_;

  // convert pointer to id for Exit
  std::vector<std::pair<u32, NodeId> > tmpExit;
  tmpExit.reserve(exits_.size());
  if (hasPronunciation_) {
    for (const Exit& e : exits_) {
      verify(e.pronunciation->id() >= 0); // should be valid
      tmpExit.emplace_back((u32)e.pronunciation->id(), e.transitRoot);
    }
  } else {
    for (const Exit& e : exits_) {
      verify(e.lemma->id() >= 0); // should be valid
      tmpExit.emplace_back((u32)e.lemma->id(), e.transitRoot);
    }
  }
  out << tmpExit;

  bool forceEndNodes = !forceEndNodes_.empty();
  out << forceEndNodes;
  if (forceEndNodes)
    out << forceEndNodes_;

  // Note: scaled values hard coded into cache 
  // trade efficiency for flexibility (need to rebuild cache for different scaling)
  if (useTransitionPenalty_)
    out << transitions_ << node2transition_;

  out << getChecksum(false);
  return out.good();
}

bool LabelTree::read() {
  if(archive_.empty())
    return false;
  Core::MappedArchiveReader in = Core::Application::us()->getCacheArchiveReader(archive_, archiveEntry());
  if(!in.good())
    return false;
  Core::Application::us()->log() << "reading label tree from " << archive_;

  u32 fmtvs; 
  in >> fmtvs;
  if (fmtvs != formatVersion) {
    Core::Application::us()->warning() << "wrong compressed format, need " << formatVersion << " got " << fmtvs;
    return false;
  }

  u32 checksum = 0; 
  in >> checksum;
  if (checksum != getChecksum(true)) {
    Core::Application::us()->warning() << "wrong checksum for config settings";
    return false;
  }

  in >> numNodes_ >> hasPronunciation_;
  in >> silence_;
  in >> node2successors_;
  in >> node2labelIndex_;
  in >> node2exits_;

  // convert id to pointer for Exit
  std::vector<std::pair<u32, NodeId> > tmpExit; 
  in >> tmpExit;
  exits_.reserve(tmpExit.size());
  if (hasPronunciation_) {
    for (const auto& e : tmpExit) {
      const Bliss::LemmaPronunciation* lp = lexicon_->lemmaPronunciation(e.first);
      exits_.emplace_back(lp, lp->lemma(), e.second);
    }
  } else {
    for (const auto& e : tmpExit) {
      const Bliss::Lemma* l = lexicon_->lemma(e.first);
      exits_.emplace_back((const Bliss::LemmaPronunciation*) 0, l, e.second);
    }
  }

  bool forceEndNodes = false;
  in >> forceEndNodes;
  if (forceEndNodes) {
    Core::Application::us()->log() << "  additional reading forced end nodes";
    in >> forceEndNodes_;
  }

  if (useTransitionPenalty_) {
    Core::Application::us()->log() << "  additional reading labels' transition penalties";
    in >> transitions_ >> node2transition_;
  }

  in >> checksum;
  if (checksum != getChecksum(false)) {
    Core::Application::us()->warning() << "wrong checksum for tree contents";
    return false;
  } else {
    Core::Application::us()->log() << "reading ready";
  }

  makeNodeExitFlag();
  return in.good();
}

