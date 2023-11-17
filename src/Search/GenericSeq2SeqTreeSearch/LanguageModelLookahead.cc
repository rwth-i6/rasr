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

#include "LanguageModelLookahead.hh"
#include "MappedArchive.hh"

using namespace Seq2SeqTreeSearch;

// image version
static u32 formatVersion = 10;
static const Core::ParameterString paramCacheArchive(
  "cache-archive",
  "cache archive in which the label-tree network should be cached",
  "global-cache" );

LanguageModelLookahead::LanguageModelLookahead(const Core::Configuration& config,
                                               Lm::Score wpScale,
                                               Core::Ref<Lm::LanguageModel> lm,
                                               const Search::LabelTree& tree) :
    // Note: lm-lookahead.scale applied here in ScaledLanguageModel
    Precursor(config, wpScale, Core::Ref<const Lm::ScaledLanguageModel>(new Lm::LanguageModelScaling(config, lm)), nullptr) {
  nEntries_ = 0;
  endNode_ = Core::Type<LookaheadId>::max;
  buildLookaheadStructure(tree);
}

void LanguageModelLookahead::buildLookaheadStructure(const Search::LabelTree& tree) {
  if (!read()) {
    log("building look-ahead structure...");
    buildFromLabelTree(tree);
    if (write())
      log("successfully write look-ahead into cache");
    else // error ?
      warning("failed to write look-ahead into cache");
  } else
    log("look-ahead was read from mapped cache");

  // runtime structure
  buildBatchRequest(tree);
  log("table size (%d entries): %zd bytes", nEntries_, sizeof(ContextLookahead)+nEntries_*sizeof(Score));
  log() << "history-limit: " << historyLimit_;
}

void LanguageModelLookahead::buildFromLabelTree(const Search::LabelTree& tree) {
  u32 nTreeNodes = tree.numNodes(), 
      nTreeExits = tree.allExits().size();

  nodeId_.resize(nTreeNodes, Core::Type<LookaheadId>::max);
  exit2node_.resize(nTreeExits, Core::Type<LookaheadId>::max);
  std::deque<TreeNodeId> treeNodeQueue;
  nodeId_[tree.root()] = nEntries_++;
  treeNodeQueue.push_back(tree.root());

  // special case: transitRoots (only for converted HMM state tree)
  std::vector<TreeNodeId> transitRoots;

  traverseTree(tree, treeNodeQueue, transitRoots);
  verify(treeNodeQueue.empty());

  if (transitRoots.empty())
    transitNodeEnd_ = nEntries_;
  else
    transitNodeEnd_ = node2successors_.begin()->first;

  while (!transitRoots.empty()) {
    treeNodeQueue.insert(treeNodeQueue.begin(), transitRoots.begin(), transitRoots.end());
    transitRoots.clear();
    traverseTree(tree, treeNodeQueue, transitRoots);
    verify(treeNodeQueue.empty());
  }  

  // final check (all mapped)
  for (TreeNodeId tnId = 0; tnId < nTreeNodes; ++tnId)
    verify(nodeId_[tnId] != Core::Type<LookaheadId>::max);
  std::unordered_set<LookaheadId> endNodes;
  for (u32 teId = 0; teId < nTreeExits; ++teId) {
    verify(exit2node_[teId] != Core::Type<LookaheadId>::max);
    endNodes.insert(exit2node_[teId]);
  }
  // verify node either has successors or is end
  for (LookaheadId id = 0; id < nEntries_; ++id)
    verify(node2successors_.count(id) > 0 || endNodes.count(id) > 0);
}

void LanguageModelLookahead::traverseTree(const Search::LabelTree& tree, std::deque<TreeNodeId>& treeNodeQueue, std::vector<TreeNodeId>& transitRoots) {
  while (!treeNodeQueue.empty()) {
    // breadth first
    TreeNodeId tnId = treeNodeQueue.front(); 
    treeNodeQueue.pop_front();
    LookaheadId lId = nodeId_[tnId];
    verify(lId != Core::Type<LookaheadId>::max);

    // remove loop
    std::vector<TreeNodeId> successors(tree.getSuccessors(tnId));
    while (!successors.empty() && successors.back() == tnId)
      successors.pop_back();
 
    if (successors.size() > 1) {
      for (std::vector<TreeNodeId>::const_iterator iter = successors.begin(); iter != successors.end(); ++iter) { 
        LookaheadId sucId = nodeId_[*iter];
        if (sucId == Core::Type<LookaheadId>::max) {
          createNode(lId, *iter);
          treeNodeQueue.push_back(*iter);
        } else {
          linkNodes(lId, sucId);
        }
      }   
    } else if (successors.size() == 1) {
      TreeNodeId sucTreeNodeId = successors.front();
      LookaheadId sucId = nodeId_[sucTreeNodeId];
      if (sucId == Core::Type<LookaheadId>::max) {
        nodeId_[sucTreeNodeId] = lId;
        treeNodeQueue.push_back(sucTreeNodeId);
      } else { 
        // TODO remove redundancy (although only possible for converted minimized state tree)
        linkNodes(lId, sucId); 
      }
    } 
    
    if (tree.hasExit(tnId)) {
      const std::vector<u32>& exitIds = tree.getExits(tnId);
      verify(!exitIds.empty());
      for (std::vector<u32>::const_iterator eIt = exitIds.begin(); eIt != exitIds.end(); ++eIt) {
        // should only have uniq predecessor ?
        verify(exit2node_[*eIt] == Core::Type<LookaheadId>::max);
        exit2node_[*eIt] = lId;
        TreeNodeId transitRoot = tree.getExit(*eIt).transitRoot;
        if (nodeId_[transitRoot] == Core::Type<LookaheadId>::max) {
          nodeId_[transitRoot] = 0;
          transitRoots.push_back(transitRoot);
        }
      }
    }
  }
}

void LanguageModelLookahead::createNode(LookaheadId lId, TreeNodeId tnId) {
  LookaheadId newId = nEntries_++;
  nodeId_[tnId] = newId;
  linkNodes(lId, newId);
}

void LanguageModelLookahead::linkNodes(LookaheadId predId, LookaheadId sucId) {
  // no self link
  if (predId == sucId)
    return;

  std::vector<LookaheadId>& successors = node2successors_[predId];
  for (std::vector<LookaheadId>::const_iterator iter = successors.begin(); iter != successors.end(); ++iter)
    if (*iter == sucId)
      return;
  successors.push_back(sucId);

  // check no cycle ?
}

// build BatchRequest: always use lemma
// Note: proncunciation score is not considered here (put into acoustic score of word ends)
//       since the pronunciation variants normalization should happen only after merged later
void LanguageModelLookahead::buildBatchRequest(const Search::LabelTree& tree) {
  const std::vector<Search::LabelTree::Exit>& exits = tree.allExits();
  require(!batchRequest_);
  verify(exits.size() == exit2node_.size());

  Lm::BatchRequest batch;
  batch.reserve(exits.size());
  // loop over all exits and create request for direct node
  for (u32 idx = 0; idx < exit2node_.size(); ++idx) {
    const Bliss::Lemma* lemma = exits[idx].lemma;
    if (!lemma) {
      verify(idx == tree.getEndExitIdx());
      endNode_ = exit2node_[idx];
      continue;
    }
    // TODO whether to apply within-class prob should also be configurable
    Lm::Request request(lemma->syntacticTokenSequence(), exit2node_[idx]);
    for (u32 ti = 0; ti < request.tokens.length(); ++ti)
      request.offset += lm_->scale() * request.tokens[ti]->classEmissionScore();
    batch.push_back(request);
  }
  batchRequest_ = lm_->compileBatchRequest(batch);
}

Lm::History LanguageModelLookahead::getReducedHistory(const Lm::History& history) const { 
  // full order depending on the LM
  if (historyLimit_ >= 0)
    return lm_->reducedHistory(history, historyLimit_);
  else
    return history;
}

// reversely push scores
void LanguageModelLookahead::computeScores(const Lm::History &history, std::vector<Score> &scores) const {
  require(scores.size() == nEntries_);
  std::fill(scores.begin(), scores.end(), Core::Type<Score>::max);

  lm_->getBatch(history, batchRequest_, scores);

  Node2Successors::const_iterator beginIter = node2successors_.cbegin(), 
                                  endIter = node2successors_.find(0);
  bool hasTransitNode = (transitNodeEnd_ != nEntries_);
  if (hasTransitNode)
    beginIter = node2successors_.find(transitNodeEnd_);

  // no lm-lookahead score to roots (mapped to node-0) 
  // either not as hypothesis or considered as previous word
  scores[0] = 0.0;
  for (Node2Successors::const_iterator iter = beginIter; iter != endIter; ++iter) {
    Score s = scores[iter->first];
    for (std::vector<LookaheadId>::const_iterator sucIter = iter->second.cbegin(); sucIter != iter->second.cend(); ++sucIter) { 
      verify_(*sucIter < nEntries_);
      if (scores[*sucIter] < s)
        s = scores[*sucIter];
    }
    scores[iter->first] = s;
  } 
 
  // only for FanIn/Out (backwards compatibility): slight different scoring as AdvancedTreeSearch
  // not well-ordered, thus recursive (can be optimized, but not important)
  if (hasTransitNode)
    for (Node2Successors::const_iterator iter = node2successors_.cbegin(); iter != beginIter; ++iter)
      computeNodeScore(iter->first, scores);

  // end label also no lookahead score
  if (endNode_ != Core::Type<LookaheadId>::max)
    scores[endNode_] = 0.0;
}

void LanguageModelLookahead::computeNodeScore(LookaheadId lId, std::vector<Score> &scores) const {
  if (scores[lId] != Core::Type<Score>::max)
    return;

  Node2Successors::const_iterator iter = node2successors_.find(lId);
  // this should be an end node, but has inf score ? (can lm assign inf score to some exit ?)
  if (iter == node2successors_.end())
    return;

  Score s = Core::Type<Score>::max;
  for (std::vector<LookaheadId>::const_iterator sucIter = iter->second.cbegin(); sucIter != iter->second.cend(); ++sucIter) {
    verify_(*sucIter < nEntries_);
    computeNodeScore(*sucIter, scores);
    if (scores[*sucIter] < s)
      s = scores[*sucIter];
  }
  scores[lId] = s;
}

// image I/O
u32 LanguageModelLookahead::getChecksum() const {
  return nEntries_ + transitNodeEnd_ + nodeId_.size() + exit2node_.size() + node2successors_.size();
}

bool LanguageModelLookahead::write() {
  std::string archive = paramCacheArchive(config);
  if (archive.empty())
    return false;
  Core::MappedArchiveWriter out = Core::Application::us()->getCacheArchiveWriter(archive, archiveEntry());
  if(!out.good())
    return false;

  log("writing lm-lookahead cache");

  out << formatVersion << getChecksum(); 
  out << nEntries_ << transitNodeEnd_;
  out << nodeId_ << exit2node_;
  out << node2successors_;

  return out.good();
}

bool LanguageModelLookahead::read() {
  std::string archive = paramCacheArchive(config);
  if(archive.empty())
    return false;
  Core::MappedArchiveReader in = Core::Application::us()->getCacheArchiveReader(archive, archiveEntry());
  if(!in.good())
    return false;

  log("reading lm-lookahead cache");

  u32 fmtvs, checksum;
  in >> fmtvs >> checksum;
  if (fmtvs != formatVersion) {
    warning() << "wrong compressed format, need " << formatVersion << " got " << fmtvs;
    return false;
  }

  // read first to compute checksum
  in >> nEntries_ >> transitNodeEnd_;
  in >> nodeId_ >> exit2node_;
  in >> node2successors_;

  if (checksum != getChecksum()) {
    warning("wrong checksum of content");
    return false;
  } else {
    log() << "reading ready";
  }

  return in.good();
}

