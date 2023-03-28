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

#ifndef SEQ2SEQ_ALIGNER_HH
#define SEQ2SEQ_ALIGNER_HH

#include <Core/Component.hh>
#include <Fsa/Static.hh>
#include <Nn/LabelHistoryManager.hh>
#include <Search/Histogram.hh>
#include <Search/Types.hh>
#include "SearchSpaceStatistics.hh"

namespace Am {
class AcousticModel;
}
namespace Speech {
class Alignment;
class ModelCombination;
}
namespace Nn {
class LabelScorer;
}

namespace Search {

struct AlignTrace : public Core::ReferenceCounted {
  Fsa::LabelId labelId;
  Index step;
  Score score;
  Core::Ref<AlignTrace> predecessor;

  // TODO sibling trace for lattice

  AlignTrace(const Core::Ref<AlignTrace>& pre, Fsa::LabelId lId, Index stp, Score sco) :
      labelId(lId), step(stp), score(sco), predecessor(pre) {}
};

struct AlignLabelHypothesis {
  Fsa::StateId stateId;
  Fsa::LabelId labelId; // input label of incoming arc: usually allophoneStateIndex
  Nn::LabelHistory labelHistory; // for label scorer scoring
  Score score;
  Core::Ref<AlignTrace> trace;

  bool isBlank;
  bool isLoop;
  u32 position; // relative position

  AlignLabelHypothesis(Fsa::StateId sId, const Nn::LabelHistory& lh, Score sco):
      stateId(sId), labelId(Fsa::InvalidLabelId), labelHistory(lh), score(sco), 
      isBlank(false), isLoop(false), position(0) {}
};

// Integrated alignment interface and search space
// So far: Viterbi only
class Seq2SeqAligner : public Core::Component {
  public:
    Seq2SeqAligner(const Core::Configuration& c);
    ~Seq2SeqAligner() {}

    void initialize(const Speech::ModelCombination& modelCombination);
    void restart(Fsa::ConstAutomatonRef model);
    void align();

    bool reachedFinalState() const { return bool(bestEndTrace_); }
    void setAlignment(Speech::Alignment& alignment, bool outputLabelId=false); // best path only

    // TODO Baum-Welch
    // getAlignmentFsa: full lattice

  protected:
    void addStartupHypothesis();
    void alignNext();
    void expand();
    void activateOrUpdate(const AlignLabelHypothesis& lh, Index label);
    void prune();
    void pruneLabel(Score threshold);
    Score quantileScore(Score minScore, Score maxScore, u32 nHyps);
    void extendLabelHistory();
    void createTrace();

    void getBestEndTrace();
    u32 getStateDepth(Fsa::StateId sId);

    void debugPrint(std::string msg, bool newStep=false);

  protected:
    // log search space statistics
    Seq2SeqTreeSearch::SearchSpaceStatistics statistics_;
    Core::XmlChannel statisticsChannel_;

    // FSA fully defines the label topology, i.e. allowed path
    Core::Ref<const Fsa::StaticAutomaton> model_;
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Core::Ref<Nn::LabelScorer> labelScorer_;

    Core::Ref<AlignTrace> bestEndTrace_;
    Index step_;
    Index blankLabelIndex_;

    bool useRelativePosition_;
    u32 relativePositionClip_;

    // reverse depth to reach the finalState
    std::vector<u32> stateDepth_;

    // --- search and pruning ---
    Score bestScore_;
    Score labelPruning_;
    u32 labelPruningLimit_;
    Histogram histogram_;
 
    s32 labelRecombinationLimit_;

    typedef std::vector<AlignLabelHypothesis> LabelHypothesesList;
    LabelHypothesesList labelHypotheses_;
    LabelHypothesesList newLabelHypotheses_;

    typedef std::unordered_map<size_t, u32> LabelHashMap;
    typedef std::unordered_map<Fsa::StateId, LabelHashMap> LabelHypothesesMap;
    LabelHypothesesMap labelHypothesesMap_;
    // --------------------------

    bool debug_;
};

} // namespace

#endif // SEQ2SEQ_ALIGNER_HH
