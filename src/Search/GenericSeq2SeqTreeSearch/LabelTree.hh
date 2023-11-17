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

#ifndef LABEL_TREE_HH
#define LABEL_TREE_HH

#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <Nn/LabelScorer.hh>
#include <Search/Types.hh>

// static tree structure based on a broad-level lexicon (flexible modeling units)
// transcription output unit (orthography) is represented as a path through tree
// - each tree node holds an AM label: path - phoneme/subword sequence
// - each tree exit holds the LM token or token sequence
// - variants: just different paths/exits
// simplest case: the same label unit set is used for all transcription, AM and LM, e.g. subword, 
//                then the tree has one entry node linking to all leaf nodes (~ vocab size)

namespace Search {

typedef u32 NodeId;

// simplified AM label-based transition model (no skip allowed)
struct TransitionPenalty {
  Score loop, forward, exit;

  TransitionPenalty(Score l, Score f, Score e): loop(l), forward(f), exit(e) {}
  TransitionPenalty(): loop(0), forward(0), exit(0) {}
};

class LabelTree {
  public:
    // use ID instead of pointer (init vs. runtime simplicity) ?
    struct Exit {
      const Bliss::LemmaPronunciation* pronunciation;
      const Bliss::Lemma* lemma;
      NodeId transitRoot; // entry root to next tree

      Exit(const Bliss::LemmaPronunciation* p, const Bliss::Lemma* l, NodeId r) :
          pronunciation(p), lemma(l), transitRoot(r) {}
    };

    // config params
    static const Core::Choice labelUnitChoice;
    static const Core::ParameterChoice paramLabelUnit;
    static const Core::ParameterBool paramSkipUnknownLabel;
    static const Core::ParameterBool paramSkipSilence;
    static const Core::ParameterBool paramUseTransitionPenalty;

  public:
    LabelTree(const Core::Configuration& config,
              Core::Ref<const Am::AcousticModel> acousticModel,
              Bliss::LexiconRef lexicon,
              Core::Ref<Nn::LabelScorer> labelScorer);
    ~LabelTree() {}

    NodeId root() const { return root_; }
    bool isRoot(NodeId nId) const { return nId == root_; }
    u32 numNodes() const { return numNodes_; }
    const std::vector<Exit>& allExits() const { return exits_; }
    bool hasPronunciation() const { return hasPronunciation_; }

    NodeId silence() const { return silence_; }
    bool isSilence(NodeId nId) const { return nId == silence_; }

    bool isHMMTree() const { return labelUnit_ == labelHMM; }

    // additional for old model compatbility
    const std::unordered_set<NodeId>& forceEndNodes() const { return forceEndNodes_; }
    bool useTransitionPenalty() const { return useTransitionPenalty_; }
    Score getTransitionPenalty(NodeId source, NodeId target) const;
    Score getExitPenalty(NodeId nId) const;
    // skip transition penalty in label expansion if only exit pentaly is used
    void setExpansionPenalty(bool allowLoop);
    bool useExpansionPenalty() const { return useTransitionPenalty_ && useExpansionPenalty_; }

    // construct the tree 
    void build();

    // preprocessing at initialization time (not in image for more flexibility)
    // add label self to successors for loop 
    void activateLoop();
    // add/adjust end label to the tree
    void activateEndLabel(Index endIdx, const Bliss::Lemma* sentEndLemma, bool useNullLemma=false);
    u32 getEndNodeId() const { return endNodeId_; }
    u32 getEndExitIdx() const { return endExitIdx_; }

    // ---- search interface (all inline) ----
    Index getLabelIndex(NodeId nId) const; // node - label classId
    bool hasSuccessors(NodeId nId) const;
    const std::vector<NodeId>& getSuccessors(NodeId nId) const;
    bool hasExit(NodeId nId) const;
    const std::vector<u32>& getExits(NodeId nId) const;
    const Exit& getExit(u32 eIdx) const;
    // ---------------------------------------

    // images I/O (further optimize to avoid init overhead while maintain runtime efficiency ?)
    bool write();
    bool read();

  private:
    std::string archiveEntry() const { return "label-tree-image"; }
    u32 getChecksum(bool configSetting) const;

    // ---- label tree construction with different label unit ----
    // Note: determinization should be ok, but no minimization except HMM tree
 
    // HMM state (convert from persistent state tree)
    void buildHMMLabelTree();

    // phoneme (mono-phon)
    void buildPhonemeLabelTree();

    // sub-word (whitespace separated as orth in lexicon)
    void buildSubwordLabelTree();

    // word (complete orth)
    void buildWordLabelTree();
    // -----------------------------------------------------------

    // find/create target label (with model index) in the tree for source label
    NodeId extendLabel(NodeId source, Index label);
    NodeId createNewLabel(Index label);

    // find/create exit
    u32 addExitToNode(NodeId, const Bliss::LemmaPronunciation*, const Bliss::Lemma*, NodeId);
    u32 createNewExit(const Bliss::LemmaPronunciation*, const Bliss::Lemma*, NodeId);

    // additional transition penalities
    void makeTransitionPenalty();

    // exit flag for each node
    void makeNodeExitFlag();

    // clear all structures (call at init or re-load another ?)
    void clear();

  private:
    Core::Configuration config_;
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Bliss::LexiconRef lexicon_;
    Core::Ref<Nn::LabelScorer> labelScorer_;

    std::string archive_;
    Core::DependencySet dependencies_;

    enum LabelUnitType {
      labelHMM,
      labelPhoneme,
      labelSubword, // whitespace separated as orth in lexicon
      labelWord     // complete orth
    } labelUnit_;

    std::vector<u32> empty_;

    NodeId root_;
    NodeId numNodes_;
    bool hasPronunciation_; // whether exit has lemmaPronunciation
    bool skipUnknownLabel_;
    bool skipSilence_;
    NodeId silence_;

    // ---- search structures ----
    // TODO storage and access optimization ?

    // old model compatibility
    // only for uncoarticulated word end
    std::unordered_set<NodeId> forceEndNodes_;
    // simple 1st-order transition penalties (e.g. TDP)
    bool useTransitionPenalty_;
    bool useExpansionPenalty_;
    // 0:root ; 1:default; >=2:specials (TODO >2 configurable)
    std::vector<TransitionPenalty> transitions_;
    std::unordered_map<NodeId, u32> node2transition_;
  
    // nodeId to successor nodeIds mapping
    typedef std::unordered_map<NodeId, std::vector<NodeId> > Node2Successors;
    Node2Successors node2successors_;

    // exits
    std::vector<Exit> exits_;
    u32 endExitIdx_;
    u32 endNodeId_;

    // nodeId to exit mapping
    std::vector<bool> hasExit_;
    typedef std::unordered_map<NodeId, std::vector<u32> > Node2Exits;
    Node2Exits node2exits_;

    // nodeId to AM label classIndex  mapping
    std::vector<Index> node2labelIndex_;
    // ---------------------------
};


inline Index LabelTree::getLabelIndex(NodeId nId) const {
  return node2labelIndex_.at(nId);
}

inline bool LabelTree::hasSuccessors(NodeId nId) const {
  return !getSuccessors(nId).empty();
}

inline const std::vector<NodeId>& LabelTree::getSuccessors(NodeId nId) const {
  Node2Successors::const_iterator iter = node2successors_.find(nId);
  if (iter == node2successors_.end())
    return empty_;
  return iter->second;
}

inline bool LabelTree::hasExit(NodeId nId) const {
  return hasExit_.at(nId);
}

inline const std::vector<u32>& LabelTree::getExits(NodeId nId) const {
  Node2Exits::const_iterator iter = node2exits_.find(nId); 
  if (iter == node2exits_.end())
    return empty_;
  return iter->second;
}

inline const LabelTree::Exit& LabelTree::getExit(u32 eIdx) const {
  return exits_.at(eIdx);
}

// Note: no efficiency consideration here (just for old model compatibility)
inline Score LabelTree::getTransitionPenalty(NodeId source, NodeId target) const {
  std::unordered_map<NodeId, u32>::const_iterator iter = node2transition_.find(source);
  u32 tIdx = (iter == node2transition_.end()) ? 1 : iter->second; // defualt transition index 1
  if (source == target)
    return transitions_[tIdx].loop;
  else
    return transitions_[tIdx].forward;
}

// getExitPenalty
inline Score LabelTree::getExitPenalty(NodeId nId) const {
  std::unordered_map<NodeId, u32>::const_iterator iter = node2transition_.find(nId);
  u32 tIdx = (iter == node2transition_.end()) ? 1 : iter->second; // defualt transition index 1
  return transitions_[tIdx].exit;
}

} // namespace

#endif
