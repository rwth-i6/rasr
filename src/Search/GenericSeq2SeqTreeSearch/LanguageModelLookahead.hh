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

#ifndef SEQ2SEQ_LANGUAGEMODELLOOKAHEAD_HH
#define SEQ2SEQ_LANGUAGEMODELLOOKAHEAD_HH

#include <Search/LanguageModelLookahead.hh>
#include "LabelTree.hh"

// compressed LM lookahead structure on LabelTree
// after construction basically the same as Search/LanguageModelLookahead

namespace Seq2SeqTreeSearch {

class LanguageModelLookahead : public Search::LanguageModelLookahead {
    typedef Search::LanguageModelLookahead Precursor;
    typedef Search::NodeId TreeNodeId;

  public:
    LanguageModelLookahead(const Core::Configuration& config, Lm::Score wpScale,
                           Core::Ref<Lm::LanguageModel> lm,
                           const Search::LabelTree& tree);
    
    Lm::History getReducedHistory(const Lm::History& h) const;

  private:
    void buildLookaheadStructure(const Search::LabelTree& tree);
    void buildFromLabelTree(const Search::LabelTree& tree);
    void traverseTree(const Search::LabelTree&, std::deque<TreeNodeId>&, std::vector<TreeNodeId>&);
    void createNode(LookaheadId, TreeNodeId);
    void linkNodes(LookaheadId predId, LookaheadId sucId);

    void buildBatchRequest(const Search::LabelTree&);

    void computeScores(const Lm::History &history, std::vector<Score> &scores) const;
    void computeNodeScore(LookaheadId nId, std::vector<Score> &scores) const;

    // image I/O
    std::string archiveEntry() const { return "lm-lookahead-image"; }
    u32 getChecksum() const;
    bool read();
    bool write();

  private:
    // important structures hold in Precursor
    // nodeId_: TreeNodeId to LookaheadNodeId mapping
    // nEntries_: number of lookahad nodes
    
    // needed to guaranteen score pushing order (well-ordered before transitNode)
    LookaheadId transitNodeEnd_;
    typedef std::map<LookaheadId, std::vector<LookaheadId>, std::greater<LookaheadId>> Node2Successors;
    Node2Successors node2successors_;
    std::vector<LookaheadId> exit2node_; // direct node of each exit
    
    LookaheadId endNode_; // node for end label
};

} // namespace

#endif
