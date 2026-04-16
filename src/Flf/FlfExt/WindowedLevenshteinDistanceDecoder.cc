#include <Core/Application.hh>
#include <Core/Choice.hh>
#include <Core/ProgressIndicator.hh>
#include <Core/Utility.hh>
#include <Core/Vector.hh>
#include <Flf/CenterFrameConfusionNetworkBuilder.hh>
#include <Flf/FlfCore/Basic.hh>
#include <Flf/FlfCore/TopologicalOrderQueue.hh>
#include <Flf/FlfCore/Utility.hh>
#include <Flf/PivotArcConfusionNetworkBuilder.hh>
#include <Flf/Segment.hh>

#include "WindowedLevenshteinDistanceDecoder.hh"

namespace Flf {

// -------------------------------------------------------------------------
ConditionalPosterior::Value::Value(Fsa::LabelId label, Score condPosteriorScore, Score tuplePosteriorScore)
        : label(label),
          condPosteriorScore(condPosteriorScore),
          tuplePosteriorScore(tuplePosteriorScore) {}

class ConditionalPosterior::Internal {
    friend class ConditionalPosteriorBuilder;

public:
    static const ConditionalPosterior::Value ZeroValue;

public:
    struct Tree {
        struct Node {
            Fsa::LabelId label;
            u32          begin, end;
            Node(Fsa::LabelId label, u32 begin, u32 end)
                    : label(label),
                      begin(begin),
                      end(end) {}
        };
        typedef std::vector<Node> NodeList;

        u32                             labelOffset;
        NodeList                        nodes;
        ConditionalPosterior::ValueList values;

        Tree()
                : labelOffset(0) {}

        ConditionalPosterior::ValueRange lookupValueRange(const LabelIdList& labels) const {
            ValueRange               result(values.end(), values.end());
            NodeList::const_iterator itNode = nodes.end() - 1;
            for (LabelIdList::const_iterator itLabel = labels.begin() + labelOffset, endLabel = labels.end() - 1;
                 itLabel != endLabel; ++itLabel) {
                const Fsa::LabelId       label      = *itLabel;
                NodeList::const_iterator itNextNode = nodes.begin() + itNode->begin, endNextNode = nodes.begin() + itNode->end;
                for (; (itNextNode != endNextNode) && (itNextNode->label < label); ++itNextNode)
                    ;
                if ((itNextNode == endNextNode) || (itNextNode->label > label))
                    // label sequence not found
                    return std::make_pair(values.end(), values.end());
                itNode = itNextNode;
                verify(itNode->label == label);
            }
            return std::make_pair(values.begin() + itNode->begin, values.begin() + itNode->end);
        }

        const ConditionalPosterior::Value& lookupValue(Fsa::LabelId label, const ConditionalPosterior::ValueRange& range) const {
            ValueIterator itValue = range.first, endValue = range.second;
            for (; (itValue != endValue) && (itValue->label < label); ++itValue)
                ;
            if ((itValue == endValue) || (itValue->label > label))
                return ConditionalPosterior::Internal::ZeroValue;
            verify(itValue->label == label);
            return *itValue;
        }

        inline const ConditionalPosterior::Value& lookupValue(const LabelIdList& labels) const {
            ConditionalPosterior::ValueRange valueRange = lookupValueRange(labels);
            return lookupValue(labels.back(), valueRange);
        }
    };
    typedef std::vector<Tree> TreeList;

private:
    ConstLatticeRef          l_;
    ConstConfusionNetworkRef cn_;
    u32                      windowSize_;
    TreeList                 trees_;
    Internal() {}

private:
    Internal(ConstLatticeRef l, u32 windowSize)
            : l_(l),
              windowSize_(windowSize) {}

public:
    inline u32 windowSize() const {
        return windowSize_;
    }

    ConstConfusionNetworkRef cn() const {
        return cn_;
    }

    const TreeList& trees() const {
        return trees_;
    }

    inline ConditionalPosterior::ValueRange lookupValueRange(u32 position, const LabelIdList& labels) const {
        return trees_[position].lookupValueRange(labels);
    }

    inline const ConditionalPosterior::Value& lookupValue(u32 position, const LabelIdList& labels) const {
        return trees_[position].lookupValue(labels);
    }

    /*
     * Format:
     *
     * # slot n
     * "w_{n-m} ... w_{n-1} w_n"  "p(w_n| w_{n-m} ... w_{n-1})" "p(w_{n-m} ... w_{n-1} w_n)"
     * ...
     *
     */
    void dump(std::ostream& os, const ConditionalPosterior::Internal::Tree& tree) const {
        if (tree.nodes.empty())
            return;
        const Fsa::Alphabet&                                                                   alphabet   = *l_->getInputAlphabet();
        Tree::NodeList::const_iterator                                                         beginNode  = tree.nodes.begin();
        ConditionalPosterior::ValueList::const_iterator                                        beginValue = tree.values.begin();
        std::vector<std::string>                                                               symbols(windowSize_, "$");
        std::vector<std::pair<Tree::NodeList::const_iterator, Tree::NodeList::const_iterator>> nodeRanges(windowSize_ - tree.labelOffset);
        nodeRanges.front() = std::make_pair(tree.nodes.end() - 1, tree.nodes.end());
        for (u32 j = 1; 0 < j;) {
            symbols[j - 1 + tree.labelOffset] = (j == 1) ? "$" : alphabet.symbol(nodeRanges[j - 1].first->label);
            for (; j < nodeRanges.size(); ++j) {
                const Tree::Node& prevNode = *nodeRanges[j - 1].first;
                verify(prevNode.begin != prevNode.end);
                nodeRanges[j]                 = std::make_pair(beginNode + prevNode.begin, beginNode + prevNode.end);
                symbols[j + tree.labelOffset] = alphabet.symbol(nodeRanges[j].first->label);
            }
            for (ConditionalPosterior::ValueList::const_iterator itValue  = beginValue + nodeRanges.back().first->begin,
                                                                 endValue = beginValue + nodeRanges.back().first->end;
                 itValue != endValue; ++itValue) {
                for (std::vector<std::string>::const_iterator itSymbol = symbols.begin() + 1; itSymbol != symbols.end(); ++itSymbol)
                    os << *itSymbol << " ";
                os << alphabet.symbol(itValue->label) << "\t"
                   << itValue->condPosteriorScore << "\t"
                   << itValue->tuplePosteriorScore << std::endl;
            }
            for (--j; 0 < j;) {
                if (++nodeRanges[j].first == nodeRanges[j].second)
                    --j;
                else {
                    ++j;
                    break;
                }
            }
        }
    }

    void dump(std::ostream& os) const {
        os << "# window-size " << windowSize_ << "/ context length " << (windowSize_ - 1) << std::endl;
        os << "# w_{n-m} ... w_{n-1} w_n\tp(w_n| w_{n-m} ... w_{n-1})\tp(w_{n-m} ... w_{n-1} w_n)" << std::endl;
        for (u32 i = 0; i < trees_.size(); ++i) {
            os << "# slot " << i << std::endl;
            dump(os, trees_[i]);
        }
    }
};
const ConditionalPosterior::Value ConditionalPosterior::Internal::ZeroValue =
        ConditionalPosterior::Value(Fsa::InvalidLabelId, Semiring::Zero, Semiring::Zero);
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class ConditionalPosteriorBuilder;
typedef Core::Ref<ConditionalPosteriorBuilder> ConditionalPosteriorBuilderRef;

class ConditionalPosteriorBuilder : public Core::ReferenceCounted {
private:
    struct Arc;
    struct Node;

    struct Arc {
        u32          slotId, lastSlotId;
        Fsa::LabelId label;
        Node *       source, *target;
        f64          arcScore, normScore;
        // lexical sort
        struct SortLexically {
            bool operator()(const Arc& arc1, const Arc& arc2) const {
                if (arc1.label == Fsa::Epsilon)
                    if (arc2.label == Fsa::Epsilon)
                        return arc1.slotId > arc2.slotId;
                    else
                        return true;
                else if (arc2.label == Fsa::Epsilon)
                    return false;
                else
                    return (arc1.slotId > arc2.slotId) ||
                           ((arc1.slotId == arc2.slotId) && (arc1.label < arc2.label));
            }
        };
    };
    typedef std::vector<Arc>        ArcList;
    typedef std::vector<const Arc*> ConstArcPtrList;

    struct Node {
        bool active;
        u32  minSlotId, maxSlotId;
        f64  fwdScore, bwdScore;
        u32  beginIdx, endIdx;
        u32 *beginBwdIdxPtr, *endBwdIdxPtr;
        // temporary
        s32 gapHypId, recombinationHypId;
        Node()
                : active(false),
                  minSlotId(0),
                  maxSlotId(0),
                  fwdScore(0.0),
                  bwdScore(0.0),
                  beginIdx(0),
                  endIdx(0),
                  beginBwdIdxPtr(0),
                  endBwdIdxPtr(0),
                  gapHypId(-1),
                  recombinationHypId(-1) {}
        ~Node() {
            delete[] beginBwdIdxPtr;
        }
    };
    typedef std::vector<Node> NodeList;

    typedef std::vector<Node*> Slot;
    typedef std::vector<Slot>  SlotList;

    struct Hypothesis {
        Node*           node;
        f64             score;
        u32             arcPtrIdx;
        ConstArcPtrList arcPtrs;
        Hypothesis(Node* node, Score score, u32 nReserve)
                : node(node),
                  score(score),
                  arcPtrIdx(0) {
            arcPtrs.reserve(nReserve);
        }
    };
    typedef std::vector<Hypothesis>     HypothesisList;
    typedef std::vector<HypothesisList> SummationSpace;

private:
    u32      windowSize_;
    bool     compact_;
    bool     prune_;
    Score    pruneProbabilityMassThreshold_;
    u32      pruneMaxSlotSize_;
    ArcList  arcs_;
    NodeList nodes_;
    SlotList slots_;

private:
    /*
     * Build internal data structure:
     * to a slot s all arcs are assigned that overspan s
     */
    class Filter;
    class DummyFilter;
    class EpsilonRemovalFilter;
    class PruningFilter;
    void buildSlots(ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn, ConditionalPosterior::Internal& condPosteriors);

    /*
     * Estimate the conditional/tuple posteriors by a slot-synchronous search over the lattice
     */
    inline bool lexicalEqual(u32 slotId, Fsa::LabelId label, const Arc& arc) const {
        if (arc.slotId == slotId)
            return arc.label == label;
        else
            return Fsa::Epsilon == label;
    }
    void estimatePosteriors(ConditionalPosterior::Internal& condPosteriors);

public:
    ConditionalPosteriorBuilder(u32 windowSize, bool compact)
            : windowSize_(windowSize),
              compact_(compact),
              prune_(false) {}

    /*
     * Pruning requires a compact CN (necessary due to practical/technical issues; see below)
     */
    void setPruning(Score pruneProbabilityMassThreshold, u32 pruneMaxSlotSize) {
        verify((0.0 < pruneProbabilityMassThreshold) && (0 < pruneMaxSlotSize));
        compact_ = prune_              = true;
        pruneProbabilityMassThreshold_ = pruneProbabilityMassThreshold;
        pruneMaxSlotSize_              = pruneMaxSlotSize;
    }

    void reset() {
        slots_.clear();
        arcs_.clear();
        nodes_.clear();
    }

    ConditionalPosterior::Internal* build(ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn) {
        ConditionalPosterior::Internal* condPost = new ConditionalPosterior::Internal(l, windowSize_);
        buildSlots(l, fb, cn, *condPost);
        estimatePosteriors(*condPost);
        reset();
        return condPost;
    }

    static ConditionalPosteriorBuilderRef create(u32 windowSize, bool compact = true) {
        return ConditionalPosteriorBuilderRef(new ConditionalPosteriorBuilder(windowSize, compact));
    }
};

class ConditionalPosteriorBuilder::Filter {
public:
    virtual ~Filter() {}
    virtual bool         keep(const Fsa::LabelId label) const                  = 0;
    virtual Fsa::LabelId map(const Fsa::LabelId label, const u32 slotId) const = 0;
};

class ConditionalPosteriorBuilder::DummyFilter : public ConditionalPosteriorBuilder::Filter {
public:
    virtual bool keep(const Fsa::LabelId label) const {
        return true;
    }
    virtual Fsa::LabelId map(const Fsa::LabelId label, const u32 slotId) const {
        return label;
    }
};

class ConditionalPosteriorBuilder::EpsilonRemovalFilter : public ConditionalPosteriorBuilder::Filter {
public:
    virtual bool keep(const Fsa::LabelId label) const {
        return (label == Fsa::Epsilon) ? false : true;
    }
    virtual Fsa::LabelId map(const Fsa::LabelId label, const u32 slotId) const {
        return label;
    }
};

class ConditionalPosteriorBuilder::PruningFilter : public ConditionalPosteriorBuilder::Filter {
public:
    ConstConfusionNetworkRef cn;
    const Fsa::LabelId       invalidLabel;
    PruningFilter(ConstConfusionNetworkRef cn, Fsa::LabelId invalidLabel)
            : cn(cn),
              invalidLabel(invalidLabel) {}
    virtual bool keep(const Fsa::LabelId label) const {
        return (label == Fsa::Epsilon) ? false : true;
    }
    virtual Fsa::LabelId map(const Fsa::LabelId label, const u32 slotId) const {
        if (slotId < cn->size()) {
            const ConfusionNetwork::Slot& slot = (*cn)[slotId];
            if (slot.size() <= 8) {
                ConfusionNetwork::Slot::const_iterator it = slot.begin(), end = slot.end();
                verify_(it->label != invalidLabel);
                for (; (it != end) && (it->label < label); ++it)
                    ;
                return ((it == end) || (it->label != label)) ? invalidLabel : label;
            }
            else {
                Fsa::LabelId arcLabel;
                s32          l = 0, r = slot.end() - slot.begin() - 1, m;
                while (l <= r) {
                    // m = (l + r) / 2;
                    m = (s32)((u32)(l + r) >> 1);
                    verify_((l <= r) && (0 <= m) && (m < slot.size()));
                    arcLabel = (slot.begin() + m)->label;
                    verify_(arcLabel != invalidLabel);
                    if (label > arcLabel)
                        l = m + 1;
                    else if (label < arcLabel)
                        r = m - 1;
                    else
                        return label;
                }
                return invalidLabel;
            }
        }
        else
            return label;
    }
};

namespace {
struct CnProbabilityWeakOrder {
    ScoreId posteriorId;
    CnProbabilityWeakOrder(ScoreId posteriorId)
            : posteriorId(posteriorId) {}
    bool operator()(const ConfusionNetwork::Arc& a1, const ConfusionNetwork::Arc& a2) const {
        return a1.scores->get(posteriorId) > a2.scores->get(posteriorId);
    }
};
}  // namespace
void ConditionalPosteriorBuilder::buildSlots(ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn, ConditionalPosterior::Internal& condPosteriors) {
    verify(cn && cn->hasMap());
    const ConfusionNetwork::MapProperties& mapProperties = *cn->mapProperties;
    /*
     * Make CN compact by removing epsilon arcs and pure epsilon slots, if requested.
     * Store an instance of the (compact) CN.
     */
    Fsa::StateId* slotIdMap = new Fsa::StateId[cn->size()];
    u32           nSlots    = 0;
    if (compact_) {
        ScoreId           posteriorId      = Semiring::InvalidId;
        ConfusionNetwork* compactCnPtr     = new ConfusionNetwork;
        compactCnPtr->semiring             = cn->semiring;
        compactCnPtr->alphabet             = cn->alphabet;
        compactCnPtr->normalizedProperties = cn->normalizedProperties;
        compactCnPtr->reserve(cn->size() + 1);
        compactCnPtr->push_back(ConfusionNetwork::Slot());
        if (prune_) {
            verify(cn->isNormalized());
            compactCnPtr->normalizedProperties = cn->normalizedProperties;
            posteriorId                        = compactCnPtr->normalizedProperties->posteriorId;
            verify(Fsa::InvalidLabelId > Fsa::LastLabelId);
        }
        Fsa::StateId* itSlotIdMap = slotIdMap;
        for (ConfusionNetwork::const_iterator itSlot = cn->begin(), endSlot = cn->end(); itSlot != endSlot; ++itSlot, ++itSlotIdMap) {
            ConfusionNetwork::Slot& compactSlot = compactCnPtr->back();
            compactSlot.reserve(itSlot->size());
            bool keepSlot = false;
            for (ConfusionNetwork::Slot::const_iterator itArc = itSlot->begin(), endArc = itSlot->end(); itArc != endArc; ++itArc)
                if (itArc->label != Fsa::Epsilon) {
                    keepSlot = true;
                    compactSlot.push_back(*itArc);
                }
            if (keepSlot) {
                if (prune_) {
                    // Use Fsa::LastLabelId as "filler" for pruned probability mass
                    verify(compactSlot.back().label != Fsa::LastLabelId);
                    std::sort(compactSlot.begin(), compactSlot.end(), CnProbabilityWeakOrder(posteriorId));
                    Score                            sum   = (itSlot->front().label == Fsa::Epsilon) ? itSlot->front().scores->get(posteriorId) : 0.0;
                    ConfusionNetwork::Slot::iterator itArc = compactSlot.begin(), endArc = compactSlot.end();
                    for (u32 i = 0, max = std::min(pruneMaxSlotSize_, u32(compactSlot.size()));
                         (i < max) && (sum < pruneProbabilityMassThreshold_); ++i, ++itArc)
                        sum += itArc->scores->get(posteriorId);
                    if (itArc != endArc) {
                        compactSlot.erase(itArc, endArc);
                        ScoresRef scores = compactCnPtr->semiring->clone(compactCnPtr->semiring->one());
                        scores->set(posteriorId, 1.0 - sum);
                        compactSlot.push_back(ConfusionNetwork::Arc(Fsa::LastLabelId, scores));
                    }
                    std::sort(compactSlot.begin(), compactSlot.end());
                }
                compactCnPtr->push_back(ConfusionNetwork::Slot());
                *itSlotIdMap = nSlots++;
            }
            else
                *itSlotIdMap = Fsa::InvalidStateId;
        }
        compactCnPtr->pop_back();
        condPosteriors.cn_ = ConstConfusionNetworkRef(compactCnPtr);
    }
    else {
        for (Fsa::StateId *itSlotIdMap = slotIdMap, *endSlotIdMap = slotIdMap + cn->size();
             itSlotIdMap != endSlotIdMap; ++itSlotIdMap)
            *itSlotIdMap = nSlots++;
        condPosteriors.cn_ = cn;
    }

    if (nSlots == 0)
        return;

    /*
     * Prepare data structure storing lattice information
     */
    Filter* filter = 0;
    if (compact_) {
        if (prune_)
            filter = new PruningFilter(condPosteriors.cn_, Fsa::LastLabelId);
        else
            filter = new EpsilonRemovalFilter;
    }
    else
        filter = new DummyFilter;
    ConstStateMapRef topologicalSort = sortTopologically(l);
    nodes_.resize(topologicalSort->maxSid + 1);
    arcs_.reserve(nSlots * 20);
    slots_.resize(nSlots);
    /*
     * Connect arcs such that each arcs gets a slot information
     */
    std::vector<Fsa::StateId> S;
    nodes_[l->initialStateId()].active = true;
    S.push_back(l->initialStateId());
    ConstStateMapRef         topologicalOrderMap = findTopologicalOrder(l);
    TopologicalOrderQueueRef Q                   = createTopologicalOrderQueue(l, topologicalOrderMap);
    std::vector<Collector*>  partialFwdCols(nodes_.size(), 0);
    std::vector<u32>         nBwdArcs(nodes_.size(), 0);
    while (!S.empty()) {
        Fsa::StateId sid = S.back();
        S.pop_back();
        Node& node = nodes_[sid];
        verify(node.active);
        const FwdBwd::State& fbState = fb->state(sid);
        node.fwdScore                = fbState.fwdScore;
        node.bwdScore                = fbState.bwdScore;
        node.beginIdx                = arcs_.size();
        Q->insert(sid);
        Collector*& col = partialFwdCols[sid] = createCollector(Fsa::SemiringTypeLog);
        col->feed(0.0);
        while (!Q->empty()) {
            Fsa::StateId sid = Q->top();
            Q->pop();
            Collector*& col = partialFwdCols[sid];
            verify(col);
            f64 partialFwdScore = col->get();
            delete col;
            col                                                         = 0;
            ConstStateRef                                        sr     = l->getState(sid);
            FwdBwd::State::const_iterator                        itFb   = fb->state(sid).begin();
            ConfusionNetwork::MapProperties::Map::const_iterator itSlot = mapProperties.state(sid);
            for (Flf::State::const_iterator a = sr->begin(); a != sr->end(); ++a, ++itFb, ++itSlot) {
                Fsa::StateId targetSid = a->target();
                Fsa::StateId slotId    = Fsa::InvalidStateId;
                if ((itSlot->sid != Fsa::InvalidStateId) && filter->keep(a->input()))
                    // if ((itSlot->sid != Fsa::InvalidStateId) && (!compact_ || (a->input() != Fsa::Epsilon)))
                    slotId = slotIdMap[itSlot->sid];
                if ((slotId == Fsa::InvalidStateId) && (l->getState(targetSid)->isFinal()))
                    slotId = nSlots;
                if (slotId != Fsa::InvalidStateId) {
                    verify_(slotId <= nSlots);
                    arcs_.push_back(Arc());
                    Arc& arc         = arcs_.back();
                    arc.slotId       = slotId;
                    arc.label        = filter->map(a->input(), slotId);
                    arc.arcScore     = partialFwdScore + itFb->arcScore;
                    arc.normScore    = itFb->normScore;
                    arc.source       = &node;
                    Node& targetNode = nodes_[targetSid];
                    arc.target       = &targetNode;
                    ++nBwdArcs[targetSid];
                    if (arc.slotId + 1 > targetNode.minSlotId)
                        targetNode.minSlotId = arc.slotId + 1;
                    if (!nodes_[targetSid].active) {
                        nodes_[targetSid].active = true;
                        S.push_back(targetSid);
                    }
                }
                else {
                    if (a->input() != Fsa::Epsilon)
                        Core::Application::us()->warning(
                                "No slot information available for arc %d--\"%s\"->%d; map label to \"%s\".",
                                sid, l->getInputAlphabet()->symbol(a->input()).c_str(), targetSid,
                                l->getInputAlphabet()->symbol(Fsa::Epsilon).c_str());
                    Collector*& col = partialFwdCols[targetSid];
                    if (!col) {
                        Q->insert(targetSid);
                        col = createCollector(Fsa::SemiringTypeLog);
                    }
                    col->feed(partialFwdScore + itFb->arcScore);
                }
            }
        }
        node.endIdx = arcs_.size();
        std::sort(arcs_.begin() + node.beginIdx, arcs_.begin() + node.endIdx, Arc::SortLexically());
    }
    delete filter;
    filter = 0;
    delete[] slotIdMap;
    slotIdMap = 0;
    /*
     * Re-calculate fwd/bwd-scores (the origianl ones are not valid anymore after an epsilon removal)
     * and determine slot ranges of the arcs
     */
    Collector* col = createCollector(Fsa::SemiringTypeLog);
    {
        std::vector<u32>::const_iterator itNBwdArcs = nBwdArcs.begin();
        for (NodeList::iterator itNode = nodes_.begin(), endNode = nodes_.end(); itNode != endNode; ++itNode, ++itNBwdArcs) {
            Node& node = *itNode;
            if (node.active)
                node.beginBwdIdxPtr = node.endBwdIdxPtr = new u32[*itNBwdArcs];
        }
    }
    for (StateMap::const_reverse_iterator itSid  = topologicalSort->rbegin(),
                                          endSid = topologicalSort->rend();
         itSid != endSid; ++itSid) {
        Node& node = nodes_[*itSid];
        if (node.active && (node.beginIdx != node.endIdx)) {
            node.maxSlotId = 0;
            u32 arcIdx     = node.beginIdx;
            for (ArcList::iterator itArc = arcs_.begin() + node.beginIdx, endArc = arcs_.begin() + node.endIdx; itArc != endArc; ++itArc, ++arcIdx) {
                Arc& arc                  = *itArc;
                *arc.target->endBwdIdxPtr = arcIdx;
                ++arc.target->endBwdIdxPtr;
                col->feed(arc.arcScore + arc.target->bwdScore);
                u32 targetSlotId = arc.lastSlotId = arc.target->minSlotId - 1;
                if (targetSlotId > node.maxSlotId)
                    node.maxSlotId = targetSlotId;
            }
            node.bwdScore = col->get();
            col->reset();
            if (node.maxSlotId == nSlots)
                --node.maxSlotId;
            else
                verify((node.minSlotId <= node.maxSlotId) && (node.maxSlotId < nSlots));
            for (u32 slotId = node.minSlotId; slotId <= node.maxSlotId; ++slotId)
                slots_[slotId].push_back(&node);
        }
    }
    for (StateMap::const_iterator itSid  = topologicalSort->begin() + 1,
                                  endSid = topologicalSort->end();
         itSid != endSid; ++itSid) {
        Node& node = nodes_[*itSid];
        if (node.active) {
            verify(node.beginBwdIdxPtr != node.endBwdIdxPtr);
            for (const u32 *itArcIdx = node.beginBwdIdxPtr, *endArcIdx = node.endBwdIdxPtr; itArcIdx != endArcIdx; ++itArcIdx) {
                Arc& arc = arcs_[*itArcIdx];
                col->feed(arc.arcScore + arc.source->fwdScore);
            }
            node.fwdScore = col->get();
            col->reset();
        }
    }
    /*
     * Preform slot-wise consistency check
     */
    for (u32 slotId = 0; slotId < slots_.size(); ++slotId) {
        const Slot& slot = slots_[slotId];
        verify(!slot.empty());
        for (std::vector<Node*>::const_iterator itNodePtr = slot.begin(); itNodePtr != slot.end(); ++itNodePtr) {
            const Node& node = **itNodePtr;
            for (ArcList::const_iterator itArc = arcs_.begin() + node.beginIdx, endArc = arcs_.begin() + node.endIdx; itArc != endArc; ++itArc) {
                const Arc& arc = *itArc;
                if (slotId <= arc.lastSlotId) {
                    f64 posteriorScore = node.fwdScore + arc.arcScore + arc.target->bwdScore - arc.normScore;
                    col->feed(posteriorScore);
                }
            }
        }
        Score deviation = col->get();
        col->reset();
        if ((deviation <= -0.00995033085316808285) || (0.01005033585350144118 <= deviation))
            Core::Application::us()->warning(
                    "Slot %d is not normalized, expected 0.0 got %.5f (probability mass %.5f)",
                    slotId, deviation, ::exp(-deviation));
    }
    delete col;
}

void ConditionalPosteriorBuilder::estimatePosteriors(ConditionalPosterior::Internal& condPosteriors) {
    if (slots_.empty())
        return;

    // final result are stored here in a tree structure
    condPosteriors.trees_.resize(slots_.size());
    std::vector<ConditionalPosterior::Internal::Tree::NodeList> S(windowSize_);
    {
        std::vector<ConditionalPosterior::Internal::Tree::NodeList>::iterator itS = S.begin();
        itS->reserve(1);
        for (++itS; itS != S.end(); ++itS)
            itS->reserve(256);
    }
    LabelIdList labels(windowSize_, Core::Type<Fsa::LabelId>::max);
    labels.front() = Fsa::Epsilon;
    // the temporary summation space
    SummationSpace sumSpace(windowSize_ + 1);
    LabelIdList    nextLabels(windowSize_ + 1, Core::Type<Fsa::LabelId>::max);
    Collector*     col = createCollector(Fsa::SemiringTypeLog);

    /*
     * Iterate over all slots
     */
    Core::ProgressIndicator pi(Core::form("#slots=%zu", slots_.size()));
    pi.start(slots_.size());
    for (u32 lastSlotId = 0; lastSlotId < slots_.size(); ++lastSlotId) {
        u32 windowSize  = (lastSlotId < windowSize_) ? lastSlotId + 1 : windowSize_;
        u32 contextSize = windowSize - 1;
        u32 slotId      = lastSlotId - contextSize;

        ConditionalPosterior::Internal::Tree& tree = condPosteriors.trees_[lastSlotId];
        tree.labelOffset                           = windowSize_ - windowSize;
        tree.nodes.reserve(1024);
        tree.values.reserve(1024);
        /*
         * Initialize hypotheses
         */
        u32 context = 1;
        {
            Slot&           slot               = slots_[slotId];
            HypothesisList& nextHyps           = sumSpace[context];
            Fsa::LabelId&   nextHypsFirstLabel = nextLabels[context];
            verify(nextHypsFirstLabel == Core::Type<Fsa::LabelId>::max);
            for (std::vector<Node*>::iterator itNodePtr = slot.begin(); itNodePtr != slot.end(); ++itNodePtr) {
                Node* nodePtr = *itNodePtr;
                nextHyps.push_back(Hypothesis(nodePtr, nodePtr->fwdScore, nodePtr->endIdx - nodePtr->beginIdx));
                {
                    Hypothesis&             nextHyp   = nextHyps.back();
                    ArcList::const_iterator itNextArc = arcs_.begin() + nodePtr->beginIdx, endNextArc = arcs_.begin() + nodePtr->endIdx;
                    verify(itNextArc != endNextArc);
                    for (--endNextArc; (itNextArc != endNextArc) && (endNextArc->slotId < slotId); --endNextArc)
                        if (endNextArc->lastSlotId >= slotId)
                            nextHyp.arcPtrs.push_back(&*endNextArc);
                    ++endNextArc;
                    for (; itNextArc != endNextArc; ++itNextArc)
                        if (itNextArc->lastSlotId >= slotId)
                            nextHyp.arcPtrs.push_back(&*itNextArc);
                    const Arc& nextArc = *nextHyps.back().arcPtrs.front();
                    if (nextArc.slotId != slotId) {
                        nextHypsFirstLabel = Fsa::Epsilon;
                    }
                    else {
                        if (nextArc.label < nextHypsFirstLabel)
                            nextHypsFirstLabel = nextArc.label;
                    }
                }
            }
            verify(nextHypsFirstLabel != Core::Type<Fsa::LabelId>::max);
        }
        /*
         * Traverse hypothese
         */
        for (; 0 < context;) {
            /*
             * Build context
             */
            for (; (0 < context) && (context <= contextSize);) {
                if (nextLabels[context] == Core::Type<Fsa::LabelId>::max) {
                    /*
                     * Store tree stage
                     */
                    verify(labels[context - 1] != Core::Type<Fsa::LabelId>::max);
                    S[context - 1].push_back(
                            ConditionalPosterior::Internal::Tree::Node(
                                    labels[context - 1],
                                    tree.nodes.size(),
                                    tree.nodes.size()));
                    tree.nodes.insert(tree.nodes.end(), S[context].begin(), S[context].end());
                    S[context - 1].back().end = tree.nodes.size();
                    S[context].clear();
                    /*
                     * Reduce context
                     */
                    sumSpace[context].clear();
                    --context;
                    --slotId;
                }
                else {
                    /*
                     * Expand context
                     */
                    HypothesisList& hyps  = sumSpace[context];
                    Fsa::LabelId    label = labels[context] = nextLabels[context];

                    Fsa::LabelId&   nextLabel = nextLabels[context] = Core::Type<Fsa::LabelId>::max;
                    HypothesisList& nextHyps                        = sumSpace[context + 1];
                    Fsa::LabelId&   nextHypsFirstLabel              = nextLabels[context + 1];
                    verify(nextHypsFirstLabel == Core::Type<Fsa::LabelId>::max);
                    // reset recombination indicator
                    for (HypothesisList::iterator itHyp = hyps.begin(), endHyp = hyps.end(); itHyp != endHyp; ++itHyp)
                        itHyp->node->gapHypId = itHyp->node->recombinationHypId = -1;

                    // find and stack all nodes that are reachable by an arc with current label
                    for (HypothesisList::iterator itHyp = hyps.begin(), endHyp = hyps.end(); itHyp != endHyp; ++itHyp) {
                        Hypothesis& hyp = *itHyp;

                        if (label == Fsa::Epsilon)
                            verify(hyp.arcPtrIdx == 0);

                        ConstArcPtrList::const_iterator itArcPtr = hyp.arcPtrs.begin() + hyp.arcPtrIdx, endArcPtr = hyp.arcPtrs.end();
                        for (; (itArcPtr != endArcPtr) && lexicalEqual(slotId, label, **itArcPtr); ++itArcPtr, ++hyp.arcPtrIdx) {
                            const Arc& arc        = **itArcPtr;
                            const Arc* nextArcPtr = 0;

                            if (arc.lastSlotId > slotId) {
                                if (hyp.node->recombinationHypId != -1)
                                    dbg(hyp.node);
                                verify(hyp.node->recombinationHypId == -1);

                                if (hyp.node->gapHypId == -1) {
                                    hyp.node->gapHypId = nextHyps.size();
                                    nextHyps.push_back(Hypothesis(hyp.node, hyp.score, hyp.arcPtrs.size()));
                                    Hypothesis& nextHyp = nextHyps.back();
                                    nextHyp.arcPtrs.push_back(&arc);
                                    nextArcPtr = &arc;
                                }
                                else {
                                    Hypothesis& nextHyp = nextHyps[hyp.node->gapHypId];
                                    nextHyp.arcPtrs.push_back(&arc);
                                }
                            }
                            else {
                                verify(arc.lastSlotId == slotId);
                                verify(arc.target->gapHypId == -1);

                                if (arc.target->recombinationHypId == -1) {
                                    arc.target->recombinationHypId = nextHyps.size();
                                    nextHyps.push_back(Hypothesis(arc.target, hyp.score + arc.arcScore, arc.target->endIdx - arc.target->beginIdx));
                                    Hypothesis& nextHyp = nextHyps.back();
                                    ArcList::const_iterator
                                            itNextArc  = arcs_.begin() + arc.target->beginIdx,
                                            endNextArc = arcs_.begin() + arc.target->endIdx;
                                    verify(itNextArc != endNextArc);
                                    for (; itNextArc != endNextArc; ++itNextArc)
                                        if (itNextArc->lastSlotId >= slotId + 1)
                                            nextHyp.arcPtrs.push_back(&*itNextArc);
                                    nextArcPtr = nextHyp.arcPtrs.front();
                                }
                                else {
                                    Hypothesis& nextHyp = nextHyps[arc.target->recombinationHypId];
                                    nextHyp.score       = logAdd(nextHyp.score, hyp.score + arc.arcScore);
                                }
                            }
                            if (nextArcPtr) {
                                const Arc& nextArc = *nextArcPtr;
                                if (nextArc.slotId != slotId + 1) {
                                    nextHypsFirstLabel = Fsa::Epsilon;
                                }
                                else {
                                    if (nextArc.label < nextHypsFirstLabel)
                                        nextHypsFirstLabel = nextArc.label;
                                }
                            }
                        }
                        if (itArcPtr != endArcPtr) {
                            const Arc& arc = **itArcPtr;

                            verify((arc.slotId == slotId) && (arc.label > label));
                            if (arc.label < nextLabel)
                                nextLabel = arc.label;
                        }
                        else
                            verify(hyp.arcPtrIdx == hyp.arcPtrs.size());
                    }

                    ++context;
                    ++slotId;
                }
            }
            if (context > contextSize) {
                verify(context == contextSize + 1);

                /*
                 * Collect statistics and compute conditional probabilities
                 */
                // P(c_1...c_{N}) / P(c_1...c_{N-1})  => P(c_{N}|c_1...c_{N-1})
                verify(nextLabels[contextSize + 1] != Core::Type<Fsa::LabelId>::max);

                HypothesisList& hyps = sumSpace[contextSize + 1];
                verify(labels[contextSize] != Core::Type<Fsa::LabelId>::max);
                S[contextSize].push_back(
                        ConditionalPosterior::Internal::Tree::Node(
                                labels[contextSize], tree.values.size(), tree.values.size()));
                /*
                 * Denominator, reset recombination indicator
                 */
                for (HypothesisList::const_iterator itHyp = hyps.begin(), endHyp = hyps.end(); itHyp != endHyp; ++itHyp) {
                    const Hypothesis& hyp = *itHyp;
                    itHyp->node->gapHypId = itHyp->node->recombinationHypId = -1;
                    for (ConstArcPtrList::const_iterator itArcPtr = hyp.arcPtrs.begin(), endArcPtr = hyp.arcPtrs.end(); itArcPtr != endArcPtr; ++itArcPtr) {
                        const Arc& arc = **itArcPtr;
                        col->feed(hyp.score + arc.arcScore + arc.target->bwdScore - arc.normScore);
                    }
                }
                f64 denominator = col->get();
                col->reset();
                /*
                 * Numerators
                 */
                for (Fsa::LabelId label                            = nextLabels[contextSize + 1], nextLabel;
                     label != Core::Type<Fsa::LabelId>::max; label = nextLabel) {
                    nextLabel = Core::Type<Fsa::LabelId>::max;
                    for (HypothesisList::iterator itHyp = hyps.begin(), endHyp = hyps.end(); itHyp != endHyp; ++itHyp) {
                        Hypothesis&                     hyp      = *itHyp;
                        ConstArcPtrList::const_iterator itArcPtr = hyp.arcPtrs.begin() + hyp.arcPtrIdx, endArcPtr = hyp.arcPtrs.end();
                        for (; (itArcPtr != endArcPtr) && lexicalEqual(slotId, label, **itArcPtr); ++itArcPtr, ++hyp.arcPtrIdx) {
                            const Arc& arc = **itArcPtr;
                            col->feed(hyp.score + arc.arcScore + arc.target->bwdScore - arc.normScore);
                        }
                        if (itArcPtr != endArcPtr) {
                            const Arc& arc = **itArcPtr;
                            verify((arc.slotId == slotId) && (arc.label > label));
                            if (arc.label < nextLabel)
                                nextLabel = arc.label;
                        }
                        else
                            verify(hyp.arcPtrIdx == hyp.arcPtrs.size());
                    }
                    f64 numerator = col->get();
                    col->reset();

                    tree.values.push_back(ConditionalPosterior::Value(label, numerator - denominator, numerator));
                }
                S[contextSize].back().end   = tree.values.size();
                nextLabels[contextSize + 1] = Core::Type<Fsa::LabelId>::max;
                /*
                 * Verify
                 */
                {
                    for (ConditionalPosterior::ValueList::const_iterator
                                 itValue  = tree.values.begin() + S[contextSize].back().begin,
                                 endValue = tree.values.begin() + S[contextSize].back().end;
                         itValue != endValue; ++itValue)
                        col->feed(itValue->condPosteriorScore);
                    Score deviation = col->get();
                    col->reset();
                    if ((deviation <= -0.00995033085316808285) || (0.01005033585350144118 <= deviation))
                        Core::Application::us()->warning(
                                "Conditional posterior distribution not normalized, expected 0.0 got %.5f",
                                deviation);
                }

                /*
                 * Reduce context
                 */

                hyps.clear();
                --context;
                --slotId;
            }
        }
        verify(context == 0);
        /*
         * Build tree root
         */
        {
            verify(S[0].size() == 1);
            tree.nodes.push_back(S[0].front());
            S[0].clear();
        }
        pi.notify();
    }
    pi.finish(false);
    delete col;
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
ConditionalPosterior::ConditionalPosterior(Internal* internal)
        : internal_(internal) {}

ConditionalPosterior::~ConditionalPosterior() {
    delete internal_;
}

u32 ConditionalPosterior::contextSize() const {
    return internal_->windowSize() - 1;
}

void ConditionalPosterior::dump(std::ostream& os) const {
    internal_->dump(os);
}

ConditionalPosterior::ValueRange ConditionalPosterior::posteriors(u32 position, const LabelIdList& labels) const {
    verify(labels.size() == internal_->windowSize());
    return internal_->lookupValueRange(position, labels);
}

const ConditionalPosterior::Value& ConditionalPosterior::posterior(u32 position, const LabelIdList& labels) const {
    verify(labels.size() == internal_->windowSize());
    return internal_->lookupValue(position, labels);
}

ConstConditionalPosteriorRef ConditionalPosterior::create(
        ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn, u32 contextSize, bool compact) {
    ConditionalPosteriorBuilderRef  condPosteriorBuilder = ConditionalPosteriorBuilder::create(contextSize + 1, compact);
    ConditionalPosterior::Internal* internal             = condPosteriorBuilder->build(l, fb, cn);
    return ConstConditionalPosteriorRef(new ConditionalPosterior(internal));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class ConditionalPosteriorsNode : public FilterNode {
public:
    static const Core::ParameterInt  paramContext;
    static const Core::ParameterBool paramCompact;

private:
    Core::XmlChannel           dumpChannel_;
    u32                        contextSize_;
    FwdBwdBuilderRef           fbBuilder_;
    ConfusionNetworkFactoryRef cnBuilder_;
    bool                       compact_;

private:
    ConstLatticeRef filter(ConstLatticeRef l) {
        if (connected(1))
            printSegmentHeader(dumpChannel_, requestSegment(1));
        std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult = fbBuilder_->build(l);
        ConstConfusionNetworkRef                   cn;
        cnBuilder_->build(fbResult.first, fbResult.second);
        cn = cnBuilder_->getCn(Semiring::InvalidId, true);
        ConstConditionalPosteriorRef condPost =
                ConditionalPosterior::create(fbResult.first, fbResult.second, cn, contextSize_, compact_);
        condPost->dump(dumpChannel_);
        return fbResult.first;
    }

public:
    ConditionalPosteriorsNode(const std::string& name, const Core::Configuration& config)
            : FilterNode(name, config),
              dumpChannel_(config, "dump") {}
    virtual ~ConditionalPosteriorsNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        Core::Component::Message msg = log();
        contextSize_                 = paramContext(config);
        msg << "context-size: " << contextSize_ << "\n";
        fbBuilder_ = FwdBwdBuilder::create(select("fb"));
        cnBuilder_ = ConfusionNetworkFactory::create(select("cn"));
        cnBuilder_->dump(msg);
        compact_ = paramCompact(config);
        if (compact_)
            msg << "compact CN before extracting conditional posteriors\n";
    }

    virtual void sync() {
        cnBuilder_->reset();
    }
};
const Core::ParameterInt ConditionalPosteriorsNode::paramContext(
        "context",
        "context size",
        2);
const Core::ParameterBool ConditionalPosteriorsNode::paramCompact(
        "compact",
        "compact CN",
        true);
NodeRef createConditionalPosteriorsNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new ConditionalPosteriorsNode(name, config));
}
}  // namespace Flf

#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT

#include <Fsa/Alphabet.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Hash.hh>
#include <Fsa/Minimize.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Static.hh>

#include "Cache.hh"
#include "Compose.hh"
#include "Copy.hh"
#include "Determinize.hh"
#include "EpsilonRemoval.hh"
#include "Map.hh"

#endif

#define COND_POST_DBG 2

namespace Flf {
struct CostFunction {
    f64 del(Fsa::LabelId ref) const {
        return (ref == Fsa::Epsilon) ? 0.0 : 1.0;
    }

    f64 ins(Fsa::LabelId hyp) const {
        return (hyp == Fsa::Epsilon) ? 0.0 : 1.0;
    }

    f64 sub(Fsa::LabelId hyp, Fsa::LabelId ref) const {
        return (hyp == ref) ? 0.0 : 1.0;
    }
};

// template<class CostFunction>
class WindowedLevenshteinDistanceDecoder;
typedef Core::Ref<WindowedLevenshteinDistanceDecoder> WindowedLevenshteinDistanceDecoderRef;

class WindowedLevenshteinDistanceDecoder : public Core::ReferenceCounted {
public:
    /*
     * Result
     */
    class Result : public Core::ReferenceCounted {
    public:
        struct Word {
            Fsa::LabelId label;
            f64          risk;
            Word()
                    : label(Fsa::InvalidLabelId),
                      risk(Core::Type<f64>::max) {}
            Word(Fsa::LabelId label, f64 risk)
                    : label(label),
                      risk(risk) {}
        };
        typedef std::vector<Word> WordList;

    public:
        f64      bestRisk;
        WordList bestHyp;

        ConstLatticeRef best;
        ConstLatticeRef alignment;
        ConstLatticeRef cost;

        Result()
                : bestRisk(0.0) {}
    };
    typedef Core::Ref<Result>       ResultRef;
    typedef Core::Ref<const Result> ConstResultRef;

protected:
    /*
     * Alignemnt
     */
    struct Alignment {
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
        static ConstSemiringRef Semiring;
        static ScoresRef        CorrectScore;
        static ScoresRef        ErrorScore;

        /*
         * Backpointer storing a lattice alignment
         */
        struct Backpointer;
        typedef Core::Ref<Backpointer> BackpointerRef;
        struct Backpointer : public Core::ReferenceCounted {
            Fsa::LabelId   hypLabel, refLabel;
            BackpointerRef backptr;
            BackpointerRef sideptr;
            Backpointer()
                    : hypLabel(Fsa::InvalidLabelId),
                      refLabel(Fsa::InvalidLabelId) {}
            Backpointer(BackpointerRef backptr, Fsa::LabelId hypLabel, Fsa::LabelId refLabel)
                    : hypLabel(hypLabel),
                      refLabel(refLabel),
                      backptr(backptr) {}

            static BackpointerRef create();
            static BackpointerRef extend(BackpointerRef backptr, Fsa::LabelId hypLabel, Fsa::LabelId refLabel);
            static BackpointerRef add(BackpointerRef trgEnd, BackpointerRef srcBegin);
        };

        /*
         * The alignment window
         */
        struct Cost {
            f64            score;
            BackpointerRef bptr;
        };
#else
        struct Cost {
            f64 score;
        };
#endif
        f64   score;
        Cost* costs;
        Alignment()
                : score(Semiring::Invalid),
                  costs(0) {}
        ~Alignment() {
            delete[] costs;
        }
    };

    /*
     * Alignment: Miscellaneous classes
     */
    struct AlignmentHelper;

    /*
     * Static Search Space
     */
    struct Word {
        Fsa::LabelId label;
        u32          prefixId;
        u32          suffixId;
        u32          tailId;
        f64          condScore;
    };
    typedef Word* WordPtr;

    struct WordSuccessors {
        Word *beginWord, *endWord;
        u32   nSuffixStrings;  // number of incoming tails, i.e. incoming tail-id < nSuffixStrings
    };
    typedef std::vector<WordSuccessors> WordSuccessorsList;

    struct Slot {
        u32                slotId;
        WordSuccessorsList vSuccessorsByPrefix;
        WordSuccessorsList wSuccessorsByPrefix;
    };
    typedef std::vector<Slot> SlotList;

    /*
     * Static Search Space: Miscellaneous classes
     */
    struct PrefixTree;

    /*
     * Dynamic Search Space
     */
    struct WHead {
        Fsa::LabelId wLabel;
        Alignment    alignment;
        WHead(Fsa::LabelId wLabel)
                : wLabel(wLabel) {}
    };
    typedef WHead* WHeadPtr;

    struct WSuffix {
        u32       wSuffixId;
        Alignment sumAlignment;
        WHeadPtr *beginWHeadPtr, *endWHeadPtr;
        WSuffix(u32 wSuffixId, u32 nWHeads)
                : wSuffixId(wSuffixId) {
            beginWHeadPtr = new WHeadPtr[nWHeads];
            endWHeadPtr   = beginWHeadPtr + nWHeads;
            std::fill(beginWHeadPtr, endWHeadPtr, static_cast<WHeadPtr>(0));
        }
        ~WSuffix() {
            for (WHeadPtr* itWHeadPtr = beginWHeadPtr; itWHeadPtr != endWHeadPtr; ++itWHeadPtr)
                delete *itWHeadPtr;
            delete[] beginWHeadPtr;
            beginWHeadPtr = endWHeadPtr = 0;
        }
    };
    typedef WSuffix* WSuffixPtr;

    struct VHead {
        Fsa::LabelId vLabel;
        WSuffixPtr * beginWSuffixPtr, *endWSuffixPtr;
        VHead(Fsa::LabelId vLabel, u32 nWSuffixes)
                : vLabel(vLabel) {
            beginWSuffixPtr = new WSuffixPtr[nWSuffixes];
            endWSuffixPtr   = beginWSuffixPtr + nWSuffixes;
            std::fill(beginWSuffixPtr, endWSuffixPtr, static_cast<WSuffixPtr>(0));
        }
        ~VHead() {
            for (WSuffixPtr* itWSuffixPtr = beginWSuffixPtr; itWSuffixPtr != endWSuffixPtr; ++itWSuffixPtr)
                delete *itWSuffixPtr;
            delete[] beginWSuffixPtr;
            beginWSuffixPtr = endWSuffixPtr = 0;
        }
    };
    typedef VHead* VHeadPtr;

    struct VSuffix {
        u32       vSuffixId;
        VHeadPtr  minVHeadPtr;
        VHeadPtr *beginVHeadPtr, *endVHeadPtr;
        VSuffix(u32 vSuffixId, u32 nWSuffixes)
                : vSuffixId(vSuffixId),
                  minVHeadPtr(0) {
            beginVHeadPtr = new VHeadPtr[nWSuffixes];
            endVHeadPtr   = beginVHeadPtr + nWSuffixes;
            std::fill(beginVHeadPtr, endVHeadPtr, static_cast<VHeadPtr>(0));
        }
        ~VSuffix() {
            for (VHeadPtr* itVHeadPtr = beginVHeadPtr; itVHeadPtr != endVHeadPtr; ++itVHeadPtr)
                delete *itVHeadPtr;
            delete[] beginVHeadPtr;
            beginVHeadPtr = endVHeadPtr = 0;
        }
    };
    typedef VSuffix*                VSuffixPtr;
    typedef std::vector<VSuffixPtr> VSuffixPtrList;

    typedef std::vector<Fsa::LabelId*> LabelIdPtrList;

    struct SearchSpace {
        u32            slotId;
        VSuffixPtrList vSuffixPtrs;
        LabelIdPtrList vSuffixStrings;
        LabelIdPtrList wSuffixStrings;

        void reset() {
            slotId = 0;
            for (VSuffixPtrList::iterator itVSuffixPtr = vSuffixPtrs.begin(), endVSuffixPtr = vSuffixPtrs.end();
                 itVSuffixPtr != endVSuffixPtr; ++itVSuffixPtr)
                delete *itVSuffixPtr;
            vSuffixPtrs.clear();
            for (LabelIdPtrList::iterator itLabelIdPtr = vSuffixStrings.begin(), endLabelIdPtr = vSuffixStrings.end();
                 itLabelIdPtr != endLabelIdPtr; ++itLabelIdPtr)
                delete *itLabelIdPtr;
            vSuffixStrings.clear();
            for (LabelIdPtrList::iterator itLabelIdPtr = wSuffixStrings.begin(), endLabelIdPtr = wSuffixStrings.end();
                 itLabelIdPtr != endLabelIdPtr; ++itLabelIdPtr)
                delete *itLabelIdPtr;
            wSuffixStrings.clear();
        }

        SearchSpace()
                : slotId(0) {}
        ~SearchSpace() {
            reset();
        }
    };

    /*
     * Dynamic Search Space: Miscellaneous classes
     */
    struct PrePruning {
        // probability mass per slot
        Score threshold;
        // max. arcs per slot
        u32 maxSlotSize;
        PrePruning()
                : threshold(1.0),
                  maxSlotSize(Core::Type<u32>::max) {}
    };

    struct Pruning {
        // relative pruning threshold
        Score riskThreshold;
        // do not prune during the first supply-size steps
        u32 supplySize;
        // purge after each interval steps
        u32 interval;
        // count steps until next purging
        u32 count;

        Pruning()
                : riskThreshold(Core::Type<Score>::max),
                  supplySize(Core::Type<u32>::max),
                  interval(Core::Type<u32>::max),
                  count(0) {}
    };

private:
    CostFunction                   costFcn_;
    ConditionalPosteriorBuilderRef condPosteriorBuilder_;

    u32  windowSize_;
    bool vRestricted_;

    Collector*                          scoreColPtr_;
    typedef std::vector<CostCollector*> CostCollectorPtrList;
    CostCollectorPtrList                costColPtrs_;
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
    typedef std::vector<std::pair<Alignment::BackpointerRef, Alignment::BackpointerRef>> BackpointerCollectorList;
    BackpointerCollectorList                                                             bptrCols_;
#endif

    ResultRef   result_;
    SlotList    slots_;
    SearchSpace ss1_, ss2_;
    PrePruning  prePruning_;
    Pruning     vPruning_;

    ConstLatticeRef          l_;
    ConstConfusionNetworkRef cn_;

protected:
    void align(Alignment& trgA, const Alignment& srcA,
               const Fsa::LabelId* wPrefix, const Word& w,
               const Fsa::LabelId* vPrefix, const Word& v) const;
    void extend(SearchSpace& nextSs, const SearchSpace& ss);
    void collect(SearchSpace& ss);
    void prune(SearchSpace& ss);

    ConstLatticeRef buildLattice() const;
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
    ConstLatticeRef buildAlignmentLattice(Alignment::BackpointerRef finalBptr) const;
    ConstLatticeRef buildCostLattice(ConstLatticeRef alignment) const;
    void            trace(const VHead& vHead);
#endif

    void resetDecoder();
    void resetSearchSpace();
    void initDecoder();
    void initSearchSpace(ConditionalPosterior::Internal& condPost);

    void search();
    void dumpPartial(std::ostream& os) const;

public:
    WindowedLevenshteinDistanceDecoder() {
        windowSize_             = 3;
        vRestricted_            = false;
        scoreColPtr_            = 0;
        prePruning_.threshold   = Core::Type<Score>::max;
        prePruning_.maxSlotSize = Core::Type<u32>::max;
        vPruning_.supplySize    = Core::Type<u32>::max;
        vPruning_.interval      = Core::Type<u32>::max;
        vPruning_.riskThreshold = Core::Type<Score>::max;
    }
    ~WindowedLevenshteinDistanceDecoder() {
        resetDecoder();
    }

    void dump(std::ostream& os) const {
        if (vRestricted_)
            os << "restrict hypothesis space\n";
        if ((prePruning_.threshold >= 1.0) && (prePruning_.maxSlotSize == Core::Type<u32>::max)) {
            os << "pre-pruning deactivated\n";
        }
        else {
            if (prePruning_.threshold < 1.0)
                os << "pre-pruning-threshold(probability mass): " << prePruning_.threshold << "\n";
            if (prePruning_.maxSlotSize != Core::Type<u32>::max)
                os << "pre-pruning-threshold(max. slot size): " << prePruning_.maxSlotSize << "\n";
        }
        if ((vPruning_.interval == Core::Type<u32>::max) || (vPruning_.supplySize == Core::Type<u32>::max)) {
            os << "pruning deactivated\n";
        }
        else {
            os << "pruning-interval:  " << vPruning_.interval << "\n";
            os << "prune first at " << (vPruning_.supplySize + vPruning_.interval - 1) << "\n";
            if (vPruning_.riskThreshold != Core::Type<Score>::max)
                os << "pruning-threshold(risk distance): " << vPruning_.riskThreshold << "\n";
        }
    }

    void setContextSize(u32 d) {
        windowSize_ = 2 * d + 1;
        resetDecoder();
    }

    void setVRestricted(bool restricted) {
        vRestricted_ = restricted;
    }

    void setPrePruningThresholds(Score threshold, u32 maxSlotSize) {
        prePruning_.threshold   = threshold;
        prePruning_.maxSlotSize = maxSlotSize;
    }

    void setPruningThreshold(Score riskThreshold) {
        vPruning_.riskThreshold = riskThreshold;
    }

    void setPruningInterval(u32 interval, u32 supplySize = Core::Type<u32>::max) {
        vPruning_.interval   = (interval < 1) ? 1 : interval;
        vPruning_.supplySize = (supplySize == Core::Type<u32>::max) ? windowSize_ : supplySize;
    }

    ConstResultRef decode(ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn);

    static WindowedLevenshteinDistanceDecoderRef create() {
        return WindowedLevenshteinDistanceDecoderRef(new WindowedLevenshteinDistanceDecoder);
    }
};

#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef WindowedLevenshteinDistanceDecoder::Alignment::Backpointer::create() {
    return Alignment::BackpointerRef(new Alignment::Backpointer);
}

WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef WindowedLevenshteinDistanceDecoder::Alignment::Backpointer::extend(
        WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef backptr, Fsa::LabelId hypLabel, Fsa::LabelId refLabel) {
    return BackpointerRef(new Backpointer(backptr, hypLabel, refLabel));
}

WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef WindowedLevenshteinDistanceDecoder::Alignment::Backpointer::add(
        WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef trgLast, WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef srcFirst) {
    require(!(trgLast->sideptr) && (srcFirst));
    trgLast->sideptr = srcFirst;
    for (trgLast = trgLast->sideptr; trgLast->sideptr; trgLast = trgLast->sideptr)
        ;
    verify((trgLast) && !(trgLast->sideptr));
    return trgLast;
}

ConstSemiringRef WindowedLevenshteinDistanceDecoder::Alignment::Semiring     = ConstSemiringRef();
ScoresRef        WindowedLevenshteinDistanceDecoder::Alignment::CorrectScore = ScoresRef();
ScoresRef        WindowedLevenshteinDistanceDecoder::Alignment::ErrorScore   = ScoresRef();
#endif

struct WindowedLevenshteinDistanceDecoder::PrefixTree {
    typedef std::pair<Fsa::LabelId, u32> Element;
    typedef std::vector<Element>         Node;
    typedef std::vector<Node>            NodeList;
    typedef std::pair<u32, u32>          Leave;
    typedef std::vector<Leave>           LeaveList;

    typedef std::pair<Node::const_iterator, Node::const_iterator> Range;

    NodeList  nodes;
    LeaveList leaves;

    PrefixTree() {
        reset();
    }

    void reset() {
        leaves.clear();
        nodes.clear();
        nodes.push_back(Node());
    }

    u32 nLeaves() const {
        return leaves.size();
    }

    const Leave& lookupAndCount(Fsa::LabelId* itLabel, Fsa::LabelId* endLabel) {
        u32 nodeId = 0;
        --endLabel;
        for (; itLabel != endLabel; ++itLabel) {
            Fsa::LabelId label = *itLabel;
            Node&        node  = nodes[nodeId];
            for (Node::const_iterator itChild = node.begin(), endChild = node.end();; ++itChild) {
                if (itChild == endChild) {
                    nodeId = nodes.size();
                    node.push_back(std::make_pair(label, nodeId));
                    nodes.push_back(Node());
                    break;
                }
                else if (itChild->first == label) {
                    nodeId = itChild->second;
                    break;
                }
            }
        }
        u32 leaveId = Core::Type<u32>::max;
        {
            Fsa::LabelId label = *itLabel;
            Node&        node  = nodes[nodeId];
            for (Node::const_iterator itChild = node.begin(), endChild = node.end();; ++itChild) {
                if (itChild == endChild) {
                    leaveId = leaves.size();
                    leaves.push_back(std::make_pair(leaveId, 0));
                    node.push_back(std::make_pair(label, leaveId));
                    break;
                }
                else if (itChild->first == label) {
                    leaveId = itChild->second;
                    break;
                }
            }
        }
        Leave& leave = leaves[leaveId];
        ++leave.second;
        return leave;
    }

    const Leave& lookupExisting(Fsa::LabelId* itLabel, Fsa::LabelId* endLabel) {
        u32 nodeId = 0;
        for (; itLabel != endLabel; ++itLabel) {
            Fsa::LabelId label = *itLabel;
            const Node&  node  = nodes[nodeId];
            for (Node::const_iterator itChild = node.begin(), endChild = node.end();; ++itChild) {
                verify(itChild != endChild);
                if (itChild->first == label) {
                    nodeId = itChild->second;
                    break;
                }
            }
        }
        verify(nodeId < leaves.size());
        return leaves[nodeId];
    }
};

/*
   for string w[0..n], n = 0, 2, 4, ...

   head:    w[0]
   prefix:  w[0..n-1]
   tail:    w[n]
   suffix:  w[1..n]

   window:  size=n+1
   present: w[(n+1)/2]
   future:  w[(n+1)/2+1..n]
   past:    w[0..(n+1)/2-1]
 */

/*
 * extend/align operator
 */
void WindowedLevenshteinDistanceDecoder::align(
        WindowedLevenshteinDistanceDecoder::Alignment&       trgA,
        const WindowedLevenshteinDistanceDecoder::Alignment& srcA,
        const Fsa::LabelId* wPrefix, const Word& w,
        const Fsa::LabelId* vPrefix, const Word& v) const {
    trgA.score = srcA.score + w.condScore;
    if (windowSize_ == 1) {
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
        f64 subCost         = costFcn_.sub(w.label, v.label);
        trgA.costs[0].score = srcA.costs[0].score + subCost;
        trgA.costs[0].bptr  = Alignment::Backpointer::extend(srcA.costs[0].bptr, w.label, v.label);
#else
        trgA.costs[0].score = srcA.costs[0].score + costFcn_.sub(w.label, v.label);
#endif
    }
    else {
        const Fsa::LabelId     wHyp      = wPrefix[windowSize_ / 2];
        const Alignment::Cost *itSrcCost = srcA.costs + 1, *endSrcCost = srcA.costs + windowSize_;
        const Fsa::LabelId*    itVLabel  = vPrefix;
        Alignment::Cost*       itTrgCost = trgA.costs;
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
        f64 subCost, insCost, delCost;
        {
            subCost = (itSrcCost - 1)->score + costFcn_.sub(wHyp, *itVLabel);
            insCost = itSrcCost->score + costFcn_.ins(wHyp);
            if (subCost <= insCost) {
                itTrgCost->score = subCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend((itSrcCost - 1)->bptr, wHyp, *itVLabel);
            }
            else {
                itTrgCost->score = insCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend(itSrcCost->bptr, wHyp, Fsa::Epsilon);
            }
        }
        for (++itSrcCost, ++itVLabel, ++itTrgCost; itSrcCost != endSrcCost; ++itSrcCost, ++itVLabel, ++itTrgCost) {
            subCost = (itSrcCost - 1)->score + costFcn_.sub(wHyp, *itVLabel);
            insCost = itSrcCost->score + costFcn_.ins(wHyp);
            delCost = (itTrgCost - 1)->score + costFcn_.del(*itVLabel);
            if ((subCost <= delCost) && (subCost <= insCost)) {
                itTrgCost->score = subCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend((itSrcCost - 1)->bptr, wHyp, *itVLabel);
            }
            else if (delCost <= insCost) {
                itTrgCost->score = delCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend((itTrgCost - 1)->bptr, Fsa::Epsilon, *itVLabel);
            }
            else {
                itTrgCost->score = insCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend(itSrcCost->bptr, wHyp, Fsa::Epsilon);
            }
        }
        {
            subCost = (itSrcCost - 1)->score + costFcn_.sub(wHyp, v.label);
            delCost = (itTrgCost - 1)->score + costFcn_.del(v.label);
            if (subCost <= delCost) {
                itTrgCost->score = subCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend((itSrcCost - 1)->bptr, wHyp, v.label);
            }
            else {
                itTrgCost->score = delCost;
                itTrgCost->bptr  = Alignment::Backpointer::extend((itTrgCost - 1)->bptr, Fsa::Epsilon, v.label);
            }
        }
#else
        itTrgCost->score = std::min(
                (itSrcCost - 1)->score + costFcn_.sub(wHyp, *itVLabel),
                itSrcCost->score + costFcn_.ins(wHyp));
        for (++itSrcCost, ++itVLabel, ++itTrgCost; itSrcCost != endSrcCost; ++itSrcCost, ++itVLabel, ++itTrgCost)
            itTrgCost->score = std::min(
                    (itTrgCost - 1)->score + costFcn_.del(*itVLabel),
                    std::min(
                            (itSrcCost - 1)->score + costFcn_.sub(wHyp, *itVLabel),
                            itSrcCost->score + costFcn_.ins(wHyp)));
        itTrgCost->score = std::min(
                (itTrgCost - 1)->score + costFcn_.del(v.label),
                (itSrcCost - 1)->score + costFcn_.sub(wHyp, v.label));
#endif
#if COND_POST_DBG >= 3
        std::cerr << Core::form("probability %1.5f -> %1.5f", ::exp(-srcA.score), ::exp(-trgA.score)) << std::endl;
        const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
        std::cerr << "align:" << std::endl;
        std::cerr << "W  :         ";
        for (const Fsa::LabelId *itLabel = wPrefix, *endLabel = wPrefix + windowSize_ - 1; itLabel != endLabel; ++itLabel)
            std::cerr << alphabet.symbol(*itLabel) << " ";
        std::cerr << alphabet.symbol(w.label) << " (hyp=" << alphabet.symbol(wHyp) << ")" << std::endl;
        std::cerr << "V  :         ";
        for (const Fsa::LabelId *itLabel = vPrefix, *endLabel = vPrefix + windowSize_ - 1; itLabel != endLabel; ++itLabel)
            std::cerr << alphabet.symbol(*itLabel) << " ";
        std::cerr << alphabet.symbol(v.label) << std::endl;
        std::cerr << "t-1: ";
        for (const Alignment::Cost *itCost = srcA.costs, *endCost = srcA.costs + windowSize_; itCost != endCost; ++itCost)
            std::cerr << Core::form("%2.5f ", itCost->score);
        std::cerr << std::endl;
        std::cerr << "t  :         ";
        for (const Alignment::Cost *itCost = trgA.costs, *endCost = trgA.costs + windowSize_; itCost != endCost; ++itCost)
            std::cerr << Core::form("%2.5f ", itCost->score);
        std::cerr << std::endl;
        std::cerr << "t-1: ";
        for (const Alignment::Cost *itCost = srcA.costs, *endCost = srcA.costs + windowSize_; itCost != endCost; ++itCost)
            std::cerr << Core::form("%2.5f ", ((itCost->score == 0.0) ? 0.0 : ::exp(::log(itCost->score) - srcA.score)));
        std::cerr << std::endl;
        std::cerr << "t  :         ";
        for (const Alignment::Cost *itCost = trgA.costs, *endCost = trgA.costs + windowSize_; itCost != endCost; ++itCost)
            std::cerr << Core::form("%2.5f ", ((itCost->score == 0.0) ? 0.0 : ::exp(::log(itCost->score) - trgA.score)));
        std::cerr << std::endl;
        std::cerr << std::endl;
#endif
    }
}

/*
 * extend
 */
void WindowedLevenshteinDistanceDecoder::extend(WindowedLevenshteinDistanceDecoder::SearchSpace& nextSs, const WindowedLevenshteinDistanceDecoder::SearchSpace& ss) {
#if COND_POST_DBG >= 3
    dbg("extend, slot-id=" << ss.slotId);
    const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
#endif
    const Slot& slot = slots_[ss.slotId];
    // const u32 nVSuffixes = slot.vSuccessorsByPrefix.size();
    // verify(ss.vSuffixPtrs.size() == nVSuffixes);
    // const u32 nWSuffixes = slot.wSuccessorsByPrefix.size();
    // initialize next search space
    Slot& nextSlot = slots_[slot.slotId + 1];
    nextSs.slotId  = nextSlot.slotId;
    verify(nextSs.vSuffixPtrs.empty());
    const u32 nNextVSuffixes = nextSlot.vSuccessorsByPrefix.size();
    const u32 nNextWSuffixes = nextSlot.wSuccessorsByPrefix.size();
    nextSs.vSuffixPtrs.insert(nextSs.vSuffixPtrs.end(), nNextVSuffixes, 0);
    verify(nextSs.vSuffixStrings.empty());
    nextSs.vSuffixStrings.insert(nextSs.vSuffixStrings.end(), nNextVSuffixes, 0);
    verify(nextSs.wSuffixStrings.empty());
    nextSs.wSuffixStrings.insert(nextSs.wSuffixStrings.end(), nNextWSuffixes, 0);
    // iterate over v-suffix => next-v-prefix
    for (VSuffixPtrList::const_iterator itVSuffixPtr = ss.vSuffixPtrs.begin(), endVSuffixPtr = ss.vSuffixPtrs.end();
         itVSuffixPtr != endVSuffixPtr; ++itVSuffixPtr)
        if (*itVSuffixPtr) {
            // v-suffix
            const VSuffix& vSuffix = **itVSuffixPtr;
            verify(vSuffix.minVHeadPtr);
            const VHead& vHead = *vSuffix.minVHeadPtr;
            verify(vSuffix.vSuffixId < slot.vSuccessorsByPrefix.size());
            const WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix[vSuffix.vSuffixId];
            // v-suffix-string = next-v-prefix-string
            verify(vSuffix.vSuffixId < ss.vSuffixStrings.size());
            const Fsa::LabelId* nextVPrefixString = ss.vSuffixStrings[vSuffix.vSuffixId];
            // iterate over next-v-tail
            for (const Word* itNextVTail = vSuccessors.beginWord; itNextVTail != vSuccessors.endWord; ++itNextVTail) {
                const Word& nextVTail = *itNextVTail;
                verify(nextVTail.suffixId < nNextVSuffixes);
                const u32 nNextVHeads = nextSlot.vSuccessorsByPrefix[nextVTail.suffixId].nSuffixStrings;
                // next-v-suffix
                verify(nextVTail.suffixId < nextSs.vSuffixPtrs.size());
                VSuffixPtr& nextVSuffixPtr = nextSs.vSuffixPtrs[nextVTail.suffixId];
                if (!nextVSuffixPtr)
                    nextVSuffixPtr = new VSuffix(nextVTail.suffixId, nNextVHeads);
                // next-v-head
                Fsa::LabelId vHeadLabel;
                if (windowSize_ > 1) {
                    // next-v-suffix-string
                    verify(nextVPrefixString && (nextVTail.suffixId < nextSs.vSuffixStrings.size()));
                    Fsa::LabelId*& nextVSuffixString = nextSs.vSuffixStrings[nextVTail.suffixId];
                    if (!nextVSuffixString) {
                        Fsa::LabelId* itTrg = nextVSuffixString = new Fsa::LabelId[windowSize_ - 1];
                        for (const Fsa::LabelId *itSrc = nextVPrefixString + 1, *endSrc = nextVPrefixString + windowSize_ - 1;
                             itSrc != endSrc; ++itSrc, ++itTrg)
                            *itTrg = *itSrc;
                        *itTrg = nextVTail.label;
                    }
                    vHeadLabel = nextVPrefixString[0];
                }
                else
                    vHeadLabel = nextVTail.label;
#if COND_POST_DBG >= 3
                dbg("v-head: " << alphabet.symbol(vHeadLabel));
#endif
                verify(nextVTail.tailId < nNextVHeads);
                VHeadPtr& nextVHeadPtr = nextVSuffixPtr->beginVHeadPtr[nextVTail.tailId];
                if (!nextVHeadPtr)
                    nextVHeadPtr = new VHead(vHeadLabel, nNextWSuffixes);
                // iterate over w-suffix => next-w-prefix
                for (const WSuffixPtr *itWSuffixPtr = vHead.beginWSuffixPtr, *endWSuffixPtr = vHead.endWSuffixPtr;
                     itWSuffixPtr != endWSuffixPtr; ++itWSuffixPtr)
                    if (*itWSuffixPtr) {
                        const WSuffix& wSuffix = **itWSuffixPtr;
                        verify(wSuffix.wSuffixId < slot.wSuccessorsByPrefix.size());
                        const WordSuccessors& wSuccessors = slot.wSuccessorsByPrefix[wSuffix.wSuffixId];
                        // w-suffix-string = next-w-prefix-string
                        verify(wSuffix.wSuffixId < ss.wSuffixStrings.size());
                        const Fsa::LabelId* nextWPrefixString = ss.wSuffixStrings[wSuffix.wSuffixId];
                        // iterate over next w-tail
                        for (const Word* itNextWTail = wSuccessors.beginWord; itNextWTail != wSuccessors.endWord; ++itNextWTail) {
                            const Word& nextWTail = *itNextWTail;
                            verify(nextWTail.suffixId < nNextWSuffixes);
                            const u32 nNextWHeads = nextSlot.wSuccessorsByPrefix[nextWTail.suffixId].nSuffixStrings;
                            // next-w-suffix
                            WSuffixPtr& nextWSuffixPtr = nextVHeadPtr->beginWSuffixPtr[nextWTail.suffixId];
                            if (!nextWSuffixPtr)
                                nextWSuffixPtr = new WSuffix(nextWTail.suffixId, nNextWHeads);
                            // next-w-head
                            Fsa::LabelId wHeadLabel;
                            if (windowSize_ > 1) {
                                // next-w-suffix-string = next-w-prefix-string
                                verify(nextWPrefixString && (nextWTail.suffixId < nextSs.wSuffixStrings.size()));
                                Fsa::LabelId*& nextWSuffixString = nextSs.wSuffixStrings[nextWTail.suffixId];
                                if (!nextWSuffixString) {
                                    Fsa::LabelId* itTrg = nextWSuffixString = new Fsa::LabelId[windowSize_ - 1];
                                    for (const Fsa::LabelId *itSrc = nextWPrefixString + 1, *endSrc = nextWPrefixString + windowSize_ - 1;
                                         itSrc != endSrc; ++itSrc, ++itTrg)
                                        *itTrg = *itSrc;
                                    *itTrg = nextWTail.label;
                                }
                                wHeadLabel = nextWPrefixString[0];
                            }
                            else
                                wHeadLabel = nextWTail.label;
#if COND_POST_DBG >= 3
                            dbg("w-head: " << alphabet.symbol(wHeadLabel));
#endif
                            verify(nextWTail.tailId < nNextWHeads);
                            WHeadPtr& nextWHeadPtr = nextWSuffixPtr->beginWHeadPtr[nextWTail.tailId];
                            verify(!nextWHeadPtr);
                            nextWHeadPtr                  = new WHead(wHeadLabel);
                            nextWHeadPtr->alignment.costs = new Alignment::Cost[windowSize_];
                            align(nextWHeadPtr->alignment, wSuffix.sumAlignment, nextWPrefixString, nextWTail, nextVPrefixString, nextVTail);
                        }
                    }
                    else
                        defect();
            }
        }
}

/*
 * collect
 */
void WindowedLevenshteinDistanceDecoder::collect(WindowedLevenshteinDistanceDecoder::SearchSpace& ss) {
#if COND_POST_DBG >= 3
    dbg("collect, slot-id=" << ss.slotId);
    const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
#endif
    // const Slot &slot = slots_[ss.slotId];
    // const u32 nVSuffixes = slot.vSuccessorsByPrefix.size();
    // verify(ss.vSuffixPtrs.size() == nVSuffixes);
    // const u32 nWSuffixes = slot.wSuccessorsByPrefix.size();

    // find best v-head
    Fsa::LabelId bestLabel = Fsa::InvalidLabelId;
    f64          score, cost, risk, bestRisk = Core::Type<f64>::max;
    const u32    vScoreId = windowSize_ / 2;
    // iterate over v-suffix
    for (VSuffixPtrList::iterator itVSuffixPtr = ss.vSuffixPtrs.begin(), endVSuffixPtr = ss.vSuffixPtrs.end();
         itVSuffixPtr != endVSuffixPtr; ++itVSuffixPtr)
        if (*itVSuffixPtr) {
            const VSuffix& vSuffix  = **itVSuffixPtr;
            Collector&     scoreCol = *scoreColPtr_;
            CostCollector& costCol  = *costColPtrs_[vScoreId];
            // iterate over v-head
            for (const VHeadPtr* itVHeadPtr = vSuffix.beginVHeadPtr; itVHeadPtr != vSuffix.endVHeadPtr; ++itVHeadPtr)
                if (*itVHeadPtr) {
                    const VHead& vHead = **itVHeadPtr;
                    // iterate over w-suffix
                    for (const WSuffixPtr* itWSuffixPtr = vHead.beginWSuffixPtr; itWSuffixPtr != vHead.endWSuffixPtr; ++itWSuffixPtr) {
                        verify(*itWSuffixPtr);
                        const WSuffix& wSuffix = **itWSuffixPtr;
                        // iterate over w-head
                        for (const WHeadPtr* itWHeadPtr = wSuffix.beginWHeadPtr; itWHeadPtr != wSuffix.endWHeadPtr; ++itWHeadPtr) {
                            verify(*itWHeadPtr);
                            const WHead& wHead = **itWHeadPtr;
                            scoreCol.feed(wHead.alignment.score);
                            costCol.feed(wHead.alignment.score, wHead.alignment.costs[vScoreId].score);
                        }
                    }
                    score = scoreCol.get();
                    if ((score <= -0.00995033085316808285) || (0.01005033585350144118 <= score))
                        Core::Application::us()->warning(
                                "Not normalized, expected 0.0 got %.5f",
                                score);
                    scoreCol.reset();
                    cost = costCol.get(score);
                    costCol.reset();
                    risk = (cost == 0.0) ? 0.0 : ::exp(::log(cost) - score);
                    if (risk < bestRisk) {
                        bestLabel = vHead.vLabel;
                        bestRisk  = risk;
                    }
                }
        }
#if COND_POST_DBG >= 3
    dbg("best-v-head: " << alphabet.symbol(bestLabel) << "/" << bestRisk);
#endif
    verify(bestLabel != Fsa::InvalidLabelId);
    // iterate over v-suffix
    for (VSuffixPtrList::iterator itVSuffixPtr = ss.vSuffixPtrs.begin(), endVSuffixPtr = ss.vSuffixPtrs.end();
         itVSuffixPtr != endVSuffixPtr; ++itVSuffixPtr)
        if (*itVSuffixPtr) {
            (*itVSuffixPtr)->minVHeadPtr = 0;
#if COND_POST_DBG >= 3
            std::cerr << "v-suffix: ";
            const Fsa::LabelId* vSuffixString = ss.vSuffixStrings[(*itVSuffixPtr)->vSuffixId];
            for (const Fsa::LabelId *itLabel = vSuffixString, *endLabel = vSuffixString + windowSize_ - 1; itLabel != endLabel; ++itLabel)
                std::cerr << alphabet.symbol(*itLabel) << " ";
            std::cerr << std::endl;
#endif
            // iterate over v-head
            for (VHeadPtr* itVHeadPtr = (*itVSuffixPtr)->beginVHeadPtr; itVHeadPtr != (*itVSuffixPtr)->endVHeadPtr; ++itVHeadPtr)
                if (*itVHeadPtr) {
                    if ((*itVHeadPtr)->vLabel == bestLabel) {
                        verify(!(*itVSuffixPtr)->minVHeadPtr);
                        (*itVSuffixPtr)->minVHeadPtr = *itVHeadPtr;
                        VHead& vHead                 = **itVHeadPtr;
                        // iterate over w-suffix
                        for (WSuffixPtr* itWSuffixPtr = vHead.beginWSuffixPtr; itWSuffixPtr != vHead.endWSuffixPtr; ++itWSuffixPtr) {
                            WSuffix& wSuffix = **itWSuffixPtr;
#if COND_POST_DBG >= 3
                            std::cerr << "  w-suffix: ";
                            const Fsa::LabelId* wSuffixString = ss.wSuffixStrings[wSuffix.wSuffixId];
                            for (const Fsa::LabelId *itLabel = wSuffixString, *endLabel = wSuffixString + windowSize_ - 1; itLabel != endLabel; ++itLabel)
                                std::cerr << alphabet.symbol(*itLabel) << " ";
                            std::cerr << std::endl;
#endif
                            // sum over w-head
                            for (WHeadPtr* itWHeadPtr = wSuffix.beginWHeadPtr; itWHeadPtr != wSuffix.endWHeadPtr; ++itWHeadPtr) {
                                const WHead&     wHead = **itWHeadPtr;
                                const Alignment& a     = wHead.alignment;
                                verify(a.costs);
                                scoreColPtr_->feed(a.score);
                                const Alignment::Cost* itCost = a.costs;
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
                                BackpointerCollectorList::iterator itBptrCol = bptrCols_.begin();
                                for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin(), endCostColPtr = costColPtrs_.end();
                                     itCostColPtr != endCostColPtr; ++itCostColPtr, ++itBptrCol, ++itCost) {
                                    (*itBptrCol).second = Alignment::Backpointer::add((*itBptrCol).second, itCost->bptr);
                                    (*itCostColPtr)->feed(a.score, itCost->score);
                                }
#else
                                for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin(), endCostColPtr = costColPtrs_.end();
                                     itCostColPtr != endCostColPtr; ++itCostColPtr, ++itCost)
                                    (*itCostColPtr)->feed(a.score, itCost->score);
#endif
                            }
                            f64 scoreSum = wSuffix.sumAlignment.score = scoreColPtr_->get();
                            scoreColPtr_->reset();
                            Alignment::Cost* itWSumCost = wSuffix.sumAlignment.costs = new Alignment::Cost[windowSize_];
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
                            BackpointerCollectorList::iterator itBptrCol = bptrCols_.begin();
                            for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin(), endCostColPtr = costColPtrs_.end();
                                 itCostColPtr != endCostColPtr; ++itCostColPtr, ++itBptrCol, ++itWSumCost) {
                                itWSumCost->bptr = (*itBptrCol).first->sideptr;
                                (*itBptrCol).first->sideptr.reset();
                                (*itBptrCol).second = (*itBptrCol).first;
                                itWSumCost->score   = (*itCostColPtr)->get(scoreSum);
                                (*itCostColPtr)->reset();
                            }
#else
                            for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin(), endCostColPtr = costColPtrs_.end();
                                 itCostColPtr != endCostColPtr; ++itCostColPtr, ++itWSumCost) {
                                itWSumCost->score = (*itCostColPtr)->get(scoreSum);
                                (*itCostColPtr)->reset();
                            }
#endif
                            // delete w-heads
                            for (WHeadPtr* itWHeadPtr = wSuffix.beginWHeadPtr; itWHeadPtr != wSuffix.endWHeadPtr; ++itWHeadPtr)
                                delete *itWHeadPtr;
                            delete[] wSuffix.beginWHeadPtr;
                            wSuffix.beginWHeadPtr = wSuffix.endWHeadPtr = 0;
                        }
                    }
                    else {
                        delete *itVHeadPtr;
                        *itVHeadPtr = 0;
                    }
                }
            if (!(*itVSuffixPtr)->minVHeadPtr) {
                delete *itVSuffixPtr;
                *itVSuffixPtr = 0;
            }
        }
    // store best v-head
    result_->bestRisk = bestRisk;
    if (((windowSize_ > 1) && (ss.slotId >= windowSize_)) || (ss.slotId > windowSize_))
        result_->bestHyp.push_back(Result::Word(bestLabel, bestRisk));
#if COND_POST_DBG >= 3
    dumpPartial(std::cout);
#endif
}

/*
 * prune
 */
void WindowedLevenshteinDistanceDecoder::prune(WindowedLevenshteinDistanceDecoder::SearchSpace& ss) {
    if (windowSize_ == 1)
        return;
    if (vPruning_.riskThreshold == Core::Type<Score>::max)
        return;
    if (ss.slotId < vPruning_.supplySize)
        return;
    if (++vPruning_.count < vPruning_.interval)
        return;
    vPruning_.count = 0;
#if COND_POST_DBG >= 3
    dbg("prune, slot-id=" << ss.slotId);
    const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
#endif
    /*
     * risk based pruning of v-suffixes
     */
    f64       score, cost, risk;
    const u32 vScoreId      = windowSize_ / 2;
    const f64 riskThreshold = result_->bestRisk + vPruning_.riskThreshold;
    // iterate over v-suffix
    for (VSuffixPtrList::iterator itVSuffixPtr = ss.vSuffixPtrs.begin(), endVSuffixPtr = ss.vSuffixPtrs.end();
         itVSuffixPtr != endVSuffixPtr; ++itVSuffixPtr)
        if (*itVSuffixPtr) {
            Collector&     scoreCol = *scoreColPtr_;
            CostCollector& costCol  = *costColPtrs_[vScoreId];
            risk                    = Core::Type<f64>::max;
            verify((*itVSuffixPtr)->minVHeadPtr);
            // iterate over w-suffix
            for (WSuffixPtr *itWSuffixPtr = (*itVSuffixPtr)->minVHeadPtr->beginWSuffixPtr, *endWSuffixPtr = (*itVSuffixPtr)->minVHeadPtr->endWSuffixPtr;
                 itWSuffixPtr != endWSuffixPtr; ++itWSuffixPtr) {
                WSuffix& wSuffix = **itWSuffixPtr;
                scoreCol.feed(wSuffix.sumAlignment.score);
                costCol.feed(wSuffix.sumAlignment.score, wSuffix.sumAlignment.costs[vScoreId].score);
            }
            score = scoreCol.get();
            if ((score <= -0.00995033085316808285) || (0.01005033585350144118 <= score))
                Core::Application::us()->warning(
                        "Not normalized, expected 0.0 got %.5f",
                        score);
            scoreCol.reset();
            cost = costCol.get(score);
            costCol.reset();
            risk = (cost == 0.0) ? 0.0 : ::exp(::log(cost) - score);
            verify(risk != Core::Type<f64>::max);
            if (risk > riskThreshold) {
                delete *itVSuffixPtr;
                *itVSuffixPtr = 0;
            }
        }
}

/*
 * dump (partial) hypotheses
 */
void WindowedLevenshteinDistanceDecoder::dumpPartial(std::ostream& os) const {
    const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
    os << "partial result (#words=" << result_->bestHyp.size() << ",risk=" << result_->bestRisk << "):" << std::endl;
    for (Result::WordList::const_iterator itWord = result_->bestHyp.begin(), endWord = result_->bestHyp.end();
         itWord != endWord; ++itWord)
        os << Core::form("  %6.2f %s", itWord->risk, alphabet.symbol(itWord->label).c_str()) << std::endl;
    os << std::endl;
}

/*
 * build linear lattice from result using the corresponding CN for time information
 */
ConstLatticeRef WindowedLevenshteinDistanceDecoder::buildLattice() const {
    verify(result_->bestHyp.size() == cn_->size());
    ConstSemiringRef  semiring = cn_->semiring;
    StaticBoundaries* b        = new StaticBoundaries;
    StaticLattice*    s        = new StaticLattice;
    s->setDescription(Core::form("mbr(%s,risk=%.3f,window-size=%d)", l_->describe().c_str(), result_->bestRisk, windowSize_));
    s->setType(Fsa::TypeAcceptor);
    s->setProperties(Fsa::PropertyAcyclic | Fsa::PropertyLinear, Fsa::PropertyAll);
    s->setInputAlphabet(cn_->alphabet);
    s->setSemiring(cn_->semiring);
    s->setBoundaries(ConstBoundariesRef(b));
    s->setInitialStateId(0);
    ConfusionNetwork::const_iterator itSlot        = cn_->begin();
    Time                             lastStartTime = 0, lastEndTime = 0;
    Flf::State*                      sp  = 0;
    Fsa::StateId                     sid = 0;
    for (WindowedLevenshteinDistanceDecoder::Result::WordList::const_iterator itWord = result_->bestHyp.begin(), endWord = result_->bestHyp.end();
         itWord != endWord; ++itWord, ++itSlot) {
        Fsa::LabelId label = itWord->label;
        if ((label != Fsa::Epsilon) && (label != Fsa::LastLabelId)) {
            ConfusionNetwork::Slot::const_iterator itArc = itSlot->begin();
            for (; itArc->label != label; ++itArc)
                verify(itArc != itSlot->end());
            if (lastEndTime < itArc->begin) {
                sp = new Flf::State(sid++);
                s->setState(sp);
                b->set(sp->id(), Boundary(lastEndTime));
                sp->newArc(sid, semiring->one(), Fsa::Epsilon, Fsa::Epsilon);
                lastEndTime = itArc->begin;
            }
            else
                lastEndTime = std::max((lastEndTime + itArc->begin) / 2, lastStartTime + 1);
            sp = new Flf::State(sid++);
            s->setState(sp);
            b->set(sp->id(), Boundary(lastEndTime));
            sp->newArc(sid, itArc->scores, label, label);
            lastStartTime = lastEndTime;
            lastEndTime   = std::max(itArc->begin + itArc->duration, lastStartTime + 1);
        }
        else if (label == Fsa::LastLabelId)
            Core::Application::us()->warning(
                    "Pre-pruning filler label; discard word. "
                    "The pre-pruning filler is probably the result of too heavy pre-pruning.");
#if COND_POST_DBG >= 3
        // dbg
        Fsa::LabelId bestCnLabel = Fsa::InvalidLabelId;
        Score        bestCnScore = 0.0, sum = 0.0;
        for (ConfusionNetwork::Slot::const_iterator itArc = itSlot->begin(), endArc = itSlot->end();
             itArc != endArc; ++itArc) {
            Score cnScore = itArc->scores->get(cn_->normalizedProperties->posteriorId);
            if (cnScore > bestCnScore) {
                bestCnLabel = itArc->label;
                bestCnScore = cnScore;
            }
            sum += cnScore;
        }
        if (1.0 - sum > bestCnScore) {
            bestCnLabel = Fsa::Epsilon;
            bestCnScore = 1.0 - sum;
        }
        std::cerr << cn_->alphabet->symbol(label) << ", "
                  << cn_->alphabet->symbol(bestCnLabel) << ":" << bestCnScore;
        if (bestCnLabel != label)
            std::cerr << " <- ATTENTION";
        std::cerr << std::endl;
#endif
    }
    if (lastEndTime <= lastStartTime)
        lastEndTime = lastStartTime + 1;
    sp = new Flf::State(sid++);
    s->setState(sp);
    b->set(sp->id(), Boundary(lastEndTime));
    sp->setFinal(semiring->one());
    return ConstLatticeRef(s);
}

#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
struct WindowedLevenshteinDistanceDecoder::AlignmentHelper {
    typedef WindowedLevenshteinDistanceDecoder::Alignment::Backpointer    Backpointer;
    typedef WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef BackpointerRef;

    struct Arc {
        Fsa::StateId   fromSid, toSid;
        BackpointerRef bptr;
        Arc(Fsa::StateId fromSid, Fsa::StateId toSid, BackpointerRef bptr)
                : fromSid(fromSid),
                  toSid(toSid),
                  bptr(bptr) {}
    };
    typedef std::vector<Arc> ArcList;

    template<class P>
    struct PointerToSizeT {
        size_t operator()(const P p) const {
            return reinterpret_cast<const size_t>(p);
        }
    };
    typedef Fsa::Hash<Backpointer*, PointerToSizeT<Backpointer*>> BackpointerPtrHash;
    typedef std::vector<BackpointerRef>                           BackpointerRefList;
};

ConstLatticeRef WindowedLevenshteinDistanceDecoder::buildAlignmentLattice(WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef finalBptr) const {
    /*
     * Build alignment graph
     */
    AlignmentHelper::ArcList            alignment;
    AlignmentHelper::BackpointerRefList Q;
    AlignmentHelper::BackpointerPtrHash H;
    Fsa::StateId                        initialSid = H.insert(0);
    Fsa::StateId                        finalSid   = H.insert(finalBptr.get());
    Q.push_back(finalBptr);
    while (!Q.empty()) {
        Alignment::BackpointerRef bptr = Q.back();
        Q.pop_back();
        Fsa::StateId toSid = H.find(bptr.get());
        for (; bptr; bptr = bptr->sideptr) {
            Fsa::StateId fromSid;
            if (bptr->backptr) {
                std::pair<u32, bool> result = H.insertExisting(bptr->backptr.get());
                fromSid                     = result.first;
                if (!result.second)
                    Q.push_back(bptr->backptr);
            }
            else
                fromSid = initialSid;
            alignment.push_back(AlignmentHelper::Arc(fromSid, toSid, bptr));
        }
    }
    /*
     * Minimize alignment graph
     */
    Fsa::StaticAlphabet*            staticDummyAlphabet = new Fsa::StaticAlphabet;
    Fsa::ConstAlphabetRef           dummyAlphabet       = Fsa::ConstAlphabetRef(staticDummyAlphabet);
    Core::Ref<Fsa::StaticAutomaton> dummyFsa            = Core::ref(new Fsa::StaticAutomaton);
    {
        Fsa::StaticAlphabet&  alphabet = *staticDummyAlphabet;
        Fsa::StaticAutomaton& fsa      = *dummyFsa;
        fsa.setType(Fsa::TypeAcceptor);
        fsa.setProperties(Fsa::PropertyAcyclic, Fsa::PropertyAll);
        fsa.setInputAlphabet(dummyAlphabet);
        fsa.setSemiring(Fsa::TropicalSemiring);
        std::string::size_type symbolSize  = sizeof(Fsa::LabelId) + sizeof(Fsa::LabelId);
        char*                  symbol      = new char[symbolSize];
        const Fsa::Weight      tropicalOne = Fsa::TropicalSemiring->one();
        for (Fsa::StateId sid = 0; sid < H.size(); ++sid)
            fsa.newState(sid);
        fsa.setInitialStateId(initialSid);
        fsa.fastState(finalSid)->setFinal(tropicalOne);
        for (AlignmentHelper::ArcList::const_iterator itArc = alignment.begin(); itArc != alignment.end(); ++itArc) {
            const AlignmentHelper::Arc&                                       bptrArc = *itArc;
            const WindowedLevenshteinDistanceDecoder::Alignment::Backpointer& bptr    = *bptrArc.bptr;
            ::memcpy(symbol, &bptr.hypLabel, sizeof(Fsa::LabelId));
            ::memcpy(symbol + sizeof(Fsa::LabelId), &bptr.refLabel, sizeof(Fsa::LabelId));
            Fsa::LabelId label = alphabet.addSymbol(std::string(symbol, symbolSize));
            fsa.fastState(bptrArc.fromSid)->newArc(bptrArc.toSid, tropicalOne, label);
        }
        delete[] symbol;
        dummyFsa = Fsa::staticCopy(Fsa::normalize(Fsa::staticCopy(Fsa::removeEpsilons(Fsa::staticCopy(Fsa::minimize(dummyFsa))))));
    }
    /*
     * Convert alignment graph
     */
    StaticLattice* s = new StaticLattice;
    {
        const Fsa::StaticAlphabet&  alphabet = *staticDummyAlphabet;
        const Fsa::StaticAutomaton& fsa      = *dummyFsa;
        if (!Alignment::Semiring) {
            Alignment::Semiring     = Semiring::create(Fsa::SemiringTypeTropical, 1, ScoreList(1, 1.0), KeyList(1, "penalty"));
            Alignment::CorrectScore = Alignment::Semiring->one();
            Alignment::ErrorScore   = Alignment::Semiring->clone(Alignment::Semiring->one());
            Alignment::ErrorScore->set(0, 1.0);
        }
        s->setDescription("bayes-risk-alignment");
        s->setType(Fsa::TypeTransducer);
        s->setProperties(Fsa::PropertyAcyclic, Fsa::PropertyAll);
        s->setInputAlphabet(cn_->alphabet);
        s->setOutputAlphabet(cn_->alphabet);
        s->setSemiring(Alignment::Semiring);
        s->setInitialStateId(fsa.initialStateId());
        for (Fsa::StateId sid = 0; sid < fsa.size(); ++sid) {
            const Fsa::State* dummySp = fsa.fastState(sid);
            if (dummySp) {
                State* sp = new State(dummySp->id());
                s->setState(sp);
                if (dummySp->isFinal())
                    sp->setFinal(Alignment::CorrectScore);
                for (Fsa::State::const_iterator a = dummySp->begin(), a_end = dummySp->end(); a != a_end; ++a) {
                    verify((Fsa::FirstLabelId <= a->input()) && (a->input() <= Fsa::LastLabelId));
                    std::string  symbol = alphabet.symbol(a->input());
                    Fsa::LabelId hypLabel;
                    Fsa::LabelId refLabel;
                    ::memcpy(&hypLabel, symbol.c_str(), sizeof(Fsa::LabelId));
                    ::memcpy(&refLabel, symbol.c_str() + sizeof(Fsa::LabelId), sizeof(Fsa::LabelId));
                    ScoresRef score = (hypLabel == refLabel) ? Alignment::CorrectScore : Alignment::ErrorScore;
                    sp->newArc(a->target(), score, hypLabel, refLabel);
                }
            }
        }
    }
    return ConstLatticeRef(s);
}

/*
 * Attention: Non-determinism in the input lattice can cause duplicated pathes in the cost lattice;
 * can be avoided by making the input lattice deterministic
 */
ConstLatticeRef WindowedLevenshteinDistanceDecoder::buildCostLattice(ConstLatticeRef alignment) const {
    // prepare lattice
    StaticLatticeRef    s = StaticLatticeRef(new StaticLattice);
    StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
    s->setBoundaries(b);
    copy(projectInput(l_), s.get(), b.get());
    s->setSemiring(Alignment::Semiring);
    ScoresRef one = s->semiring()->one();
    for (Fsa::StateId sid = 0; sid < s->size(); ++sid) {
        State* sp = s->fastState(sid);
        if (sp) {
            if (sp->isFinal())
                sp->weight_ = one;
            for (State::iterator a = sp->begin(), a_end = sp->end(); a != a_end; ++a)
                a->weight_ = one;
        }
    }
    ConstLatticeRef l = fastRemoveEpsilons(persistent(minimize(fastRemoveEpsilons(s))));
    // prepare alignment
    alignment = persistent(fastRemoveEpsilons(alignment));
    // compose lattice and alignment
    StaticLatticeRef    costS = StaticLatticeRef(new StaticLattice);
    StaticBoundariesRef costB = StaticBoundariesRef(new StaticBoundaries);
    costS->setBoundaries(costB);
    copy(composeSequencing(l, alignment), costS.get(), costB.get());
    trimInPlace(costS);
    return costS;
}

void WindowedLevenshteinDistanceDecoder::trace(const VHead& vHead) {
    const u32                                                     vScoreId = windowSize_ / 2;
    WindowedLevenshteinDistanceDecoder::Alignment::BackpointerRef bptr, lastBptr;
    for (WSuffixPtr* itWSuffixPtr = vHead.beginWSuffixPtr; itWSuffixPtr != vHead.endWSuffixPtr; ++itWSuffixPtr) {
        WSuffix& wSuffix = **itWSuffixPtr;
        if (!bptr)
            lastBptr = bptr = wSuffix.sumAlignment.costs[vScoreId].bptr;
        else
            lastBptr = Alignment::Backpointer::add(lastBptr, wSuffix.sumAlignment.costs[vScoreId].bptr);
    }
    verify(bptr);
    result_->alignment = buildAlignmentLattice(bptr);
    result_->cost      = buildCostLattice(result_->alignment);
}
#endif

/*
 * reset static part of decoder; only influenced by window-size
 */
void WindowedLevenshteinDistanceDecoder::resetDecoder() {
    condPosteriorBuilder_.reset();
    delete scoreColPtr_;
    scoreColPtr_ = 0;
    for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin();
         itCostColPtr != costColPtrs_.end(); ++itCostColPtr)
        delete *itCostColPtr;
    costColPtrs_.clear();
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
    bptrCols_.clear();
#endif
}

/*
 * reset dynamic part of decoder;
 * free all memory
 */
void WindowedLevenshteinDistanceDecoder::resetSearchSpace() {
    ss1_.reset();
    ss2_.reset();
    for (SlotList::iterator itSlot = slots_.begin(); itSlot != slots_.end(); ++itSlot) {
        for (WordSuccessorsList::iterator itWordSuccessor  = itSlot->vSuccessorsByPrefix.begin(),
                                          endWordSuccessor = itSlot->vSuccessorsByPrefix.end();
             itWordSuccessor != endWordSuccessor; ++itWordSuccessor)
            delete[] itWordSuccessor->beginWord;
        for (WordSuccessorsList::iterator itWordSuccessor  = itSlot->wSuccessorsByPrefix.begin(),
                                          endWordSuccessor = itSlot->wSuccessorsByPrefix.end();
             itWordSuccessor != endWordSuccessor; ++itWordSuccessor)
            delete[] itWordSuccessor->beginWord;
    }
    slots_.clear();
    vPruning_.count = 0;
}

/*
 * initialize static part of decoder
 */
void WindowedLevenshteinDistanceDecoder::initDecoder() {
    if (!condPosteriorBuilder_) {
        condPosteriorBuilder_ = ConditionalPosteriorBuilder::create(windowSize_);
        if ((prePruning_.threshold < 1.0) || (prePruning_.maxSlotSize != Core::Type<u32>::max))
            condPosteriorBuilder_->setPruning(prePruning_.threshold, prePruning_.maxSlotSize);
        scoreColPtr_ = createCollector(Fsa::SemiringTypeLog);
        costColPtrs_.insert(costColPtrs_.begin(), windowSize_, 0);
        for (CostCollectorPtrList::iterator itCostColPtr = costColPtrs_.begin();
             itCostColPtr != costColPtrs_.end(); ++itCostColPtr)
            *itCostColPtr = CostCollector::create();
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
        bptrCols_.resize(windowSize_);
        for (BackpointerCollectorList::iterator itBptrCol = bptrCols_.begin();
             itBptrCol != bptrCols_.end(); ++itBptrCol)
            itBptrCol->first = itBptrCol->second = Alignment::Backpointer::create();
#endif
    }  // else decoder is already initialized
}

/*
 * initalize static and dynamic search space
 */
void WindowedLevenshteinDistanceDecoder::initSearchSpace(ConditionalPosterior::Internal& condPost) {
#if COND_POST_DBG >= 3
    dbg("start initialization");
    const Fsa::Alphabet& alphabet = *l_->getInputAlphabet();
#endif
    if (windowSize_ == 1) {
        /*
         * Initialize static search space
         */

        slots_.resize(condPost.trees().size() + 2);
        u32 slotId = 0;
        // init
        {
            Slot& slot  = slots_.front();
            slot.slotId = slotId++;
            slot.wSuccessorsByPrefix.resize(1);
            WordSuccessors& wSuccessors      = slot.wSuccessorsByPrefix.front();
            wSuccessors.nSuffixStrings       = 1;
            wSuccessors.beginWord            = new Word[1];
            wSuccessors.endWord              = wSuccessors.beginWord + 1;
            wSuccessors.beginWord->label     = Fsa::Epsilon;
            wSuccessors.beginWord->condScore = 0.0;
            wSuccessors.beginWord->prefixId  = 0;
            wSuccessors.beginWord->suffixId  = 0;
            wSuccessors.beginWord->tailId    = 0;
            slot.vSuccessorsByPrefix.resize(1);
            WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix.front();
            vSuccessors.nSuffixStrings  = 1;
            vSuccessors.nSuffixStrings  = vSuccessors.nSuffixStrings;
            vSuccessors.beginWord       = new Word[1];
            vSuccessors.endWord         = vSuccessors.beginWord + 1;
            ::memcpy((void*)vSuccessors.beginWord, (void*)wSuccessors.beginWord, sizeof(Word));
        }
        // work
        u32 nWordsInLastSlot = 1;
        for (; slotId < condPost.trees().size() + 1; ++slotId) {
            // dbg
#if COND_POST_DBG >= 3
            std::cerr << "slot " << slotId << ", w/v-successors-by-prefix" << std::endl;
#endif

            const ConditionalPosterior::ValueList& values = condPost.trees()[slotId - 1].values;
            Slot&                                  slot   = slots_[slotId];
            slot.slotId                                   = slotId;
            slot.wSuccessorsByPrefix.resize(1);
            WordSuccessors& wSuccessors = slot.wSuccessorsByPrefix.front();
            Word*           itWord = wSuccessors.beginWord = new Word[values.size()];
            wSuccessors.endWord                            = wSuccessors.beginWord + values.size();
            wSuccessors.nSuffixStrings                     = nWordsInLastSlot;

            // dbg
#if COND_POST_DBG >= 3
            std::cerr << 0 << ". v/w-successors (#words=" << values.size() << ", #suffix-strings=" << wSuccessors.nSuffixStrings << ")" << std::endl;
#endif

            u32 wordId = 0;
            for (ConditionalPosterior::ValueList::const_iterator itValue  = values.begin(),
                                                                 endValue = values.end();
                 itValue != endValue; ++itValue, ++itWord, ++wordId) {
                itWord->label     = itValue->label;
                itWord->prefixId  = 0;
                itWord->suffixId  = 0;
                itWord->tailId    = wordId;
                itWord->condScore = itValue->condPosteriorScore;

                // dbg
#if COND_POST_DBG >= 3
                std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", itWord->prefixId, itWord->suffixId, itWord->tailId)
                          << " " << alphabet.symbol(itWord->label) << std::endl;
#endif
            }
            slot.vSuccessorsByPrefix.resize(1);
            WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix.front();
            vSuccessors.nSuffixStrings  = nWordsInLastSlot;
            vSuccessors.beginWord       = new Word[values.size()];
            vSuccessors.endWord         = vSuccessors.beginWord + values.size();
            ::memcpy((void*)vSuccessors.beginWord, (void*)wSuccessors.beginWord, sizeof(Word) * values.size());
            nWordsInLastSlot = values.size();

            // dbg
#if COND_POST_DBG >= 3
            std::cerr << std::endl;
#endif
        }
        // dummy, flush flush last symbol
        {
            Slot& slot  = slots_.back();
            slot.slotId = slotId++;
            verify(slot.slotId == slots_.size() - 1);
            slot.wSuccessorsByPrefix.resize(1);
            WordSuccessors& wSuccessors      = slot.wSuccessorsByPrefix.front();
            wSuccessors.nSuffixStrings       = nWordsInLastSlot;
            wSuccessors.beginWord            = new Word[1];
            wSuccessors.endWord              = wSuccessors.beginWord + 1;
            wSuccessors.beginWord->label     = Fsa::InvalidLabelId;
            wSuccessors.beginWord->condScore = 0.0;
            wSuccessors.beginWord->prefixId  = 0;
            wSuccessors.beginWord->suffixId  = 0;
            wSuccessors.beginWord->tailId    = 0;
            slot.vSuccessorsByPrefix.resize(1);
            WordSuccessors& vSuccessors  = slot.vSuccessorsByPrefix.front();
            vSuccessors.nSuffixStrings   = nWordsInLastSlot;
            vSuccessors.beginWord        = new Word[1];
            vSuccessors.endWord          = vSuccessors.beginWord + 1;
            vSuccessors.beginWord->label = Fsa::InvalidLabelId;
        }
        verify(slotId == slots_.size());

        /*
         * Initialize dynamic search space
         */
        SearchSpace& ss = ss1_;
        ss.vSuffixStrings.resize(1, 0);
        ss.wSuffixStrings.resize(1, 0);
        ss.vSuffixPtrs.resize(1);
        ss.vSuffixPtrs.front() = new VSuffix(0, 1);
        VSuffix& vSuffix       = *ss.vSuffixPtrs.front();
        vSuffix.minVHeadPtr = vSuffix.beginVHeadPtr[0] = new VHead(Fsa::Epsilon, 1);
        VHead& vHead                                   = *vSuffix.beginVHeadPtr[0];
        vHead.beginWSuffixPtr[0]                       = new WSuffix(0, 1);
        WSuffix& wSuffix                               = *vHead.beginWSuffixPtr[0];
        wSuffix.sumAlignment.score                     = 0.0;
        wSuffix.sumAlignment.costs                     = new Alignment::Cost[1];
        wSuffix.sumAlignment.costs[0].score            = 0.0;
    }
    else {
        slots_.resize(condPost.trees().size() + windowSize_);
        const s32     contextSize      = windowSize_ - 1;
        Fsa::LabelId* labels           = new Fsa::LabelId[windowSize_];
        Fsa::LabelId *beginPrefixLabel = labels, *endPrefixLabel = labels + windowSize_ - 1;
        Fsa::LabelId *beginSuffixLabel = labels + 1, *endSuffixLabel = labels + windowSize_;
        PrefixTree    prefixTree1, prefixTree2;
        {
            /*
             * Initialize static search space/
             * link partial hypothesis
             */

            std::fill(labels, labels + windowSize_, Fsa::Epsilon);
            PrefixTree *lastPrefixTreePtr = &prefixTree1, *prefixTreePtr = &prefixTree2;
            lastPrefixTreePtr->lookupAndCount(beginSuffixLabel, endSuffixLabel);
            verify(lastPrefixTreePtr->nLeaves() == 1);
            std::vector<std::pair<u32, u32>> S(windowSize_);
            s32                              suffixIdOffset = contextSize;
            u32                              slotId         = 0;
            // init=windowSize-1, work=cnSize-windowSize+1
            for (; slotId < condPost.trees().size(); ++slotId) {
                // dbg
#if COND_POST_DBG >= 3
                std::cerr << "slot " << slotId << ", w-successors-by-prefix" << std::endl;
#endif

                Slot& slot  = slots_[slotId];
                slot.slotId = slotId;
                slot.wSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());
                if (vRestricted_)
                    slot.vSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());

                const ConditionalPosterior::Internal::Tree::NodeList& nodes  = condPost.trees()[slotId].nodes;
                const ConditionalPosterior::ValueList&                values = condPost.trees()[slotId].values;
                S[suffixIdOffset]                                            = std::make_pair(nodes.back().begin, nodes.back().end);
                for (s32 suffixId = suffixIdOffset; suffixIdOffset <= suffixId;) {
                    Fsa::LabelId* itLabel = labels + suffixId;
                    for (; suffixId < contextSize; ++suffixId, ++itLabel) {
                        std::pair<u32, u32>&                              range = S[suffixId];
                        const ConditionalPosterior::Internal::Tree::Node& node  = nodes[range.first];
                        *itLabel                                                = node.label;
                        S[suffixId + 1]                                         = std::make_pair(node.begin, node.end);
                    }
                    {
                        const std::pair<u32, u32> prefixIds =
                                lastPrefixTreePtr->lookupExisting(beginPrefixLabel, endPrefixLabel);
                        WordSuccessors&      wSuccessors = slot.wSuccessorsByPrefix[prefixIds.first];
                        std::pair<u32, u32>& range       = S[suffixId];
                        u32                  nWords      = range.second - range.first;
                        Word*                itWord = wSuccessors.beginWord = new Word[nWords];
                        wSuccessors.endWord                                 = wSuccessors.beginWord + nWords;
                        wSuccessors.nSuffixStrings                          = prefixIds.second;

                        // dbg
#if COND_POST_DBG >= 3
                        std::cerr << prefixIds.first << ". w-successors (#words=" << nWords << ", #suffix-strings=" << prefixIds.second << ")" << std::endl;
#endif

                        for (ConditionalPosterior::ValueList::const_iterator itValue  = values.begin() + range.first,
                                                                             endValue = values.begin() + range.second;
                             itValue != endValue; ++itValue, ++itWord) {
                            *itLabel = itValue->label;
                            const std::pair<u32, u32> suffixIds =
                                    prefixTreePtr->lookupAndCount(beginSuffixLabel, endSuffixLabel);
                            itWord->label     = itValue->label;
                            itWord->prefixId  = prefixIds.first;
                            itWord->suffixId  = suffixIds.first;
                            itWord->tailId    = suffixIds.second - 1;
                            itWord->condScore = itValue->condPosteriorScore;

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", itWord->prefixId, itWord->suffixId, itWord->tailId);
                            for (Fsa::LabelId* itLabel = beginPrefixLabel; itLabel != endSuffixLabel; ++itLabel)
                                std::cerr << " " << alphabet.symbol(*itLabel);
                            std::cerr << std::endl;
#endif
                        }
                        if (vRestricted_) {
                            WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix[prefixIds.first];
                            vSuccessors.nSuffixStrings  = wSuccessors.nSuffixStrings;
                            vSuccessors.beginWord       = new Word[nWords];
                            vSuccessors.endWord         = vSuccessors.beginWord + nWords;
                            ::memcpy((void*)vSuccessors.beginWord, (void*)wSuccessors.beginWord, sizeof(Word) * nWords);
                        }
                    }
                    for (--suffixId; suffixIdOffset <= suffixId;) {
                        std::pair<u32, u32>& range = S[suffixId];
                        if (++range.first == range.second)
                            --suffixId;
                        else
                            break;
                    }
                }
                if (suffixIdOffset > 0)
                    --suffixIdOffset;
                std::swap(lastPrefixTreePtr, prefixTreePtr);
                prefixTreePtr->reset();
            }
            // resolve suffix

            std::vector<PrefixTree::Range> SSuffix(windowSize_ - 1);
            s32                            suffixIdEnd = windowSize_ - 1;
            u32                            nFinalTails = 0;
            // work=windowSize-2
            for (; slotId < condPost.trees().size() + windowSize_ - 1; ++slotId, --suffixIdEnd) {
                // dbg
#if COND_POST_DBG >= 3
                std::cerr << "slot " << slotId << ", w-successors-by-prefix" << std::endl;
#endif

                Slot& slot  = slots_[slotId];
                slot.slotId = slotId;
                slot.wSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());
                if (vRestricted_)
                    slot.vSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());
                const PrefixTree::NodeList&  nodes  = lastPrefixTreePtr->nodes;
                const PrefixTree::LeaveList& leaves = lastPrefixTreePtr->leaves;
                SSuffix[0]                          = std::make_pair(nodes.front().begin(), nodes.front().end());
                for (s32 suffixId = 0; 0 <= suffixId;) {
                    Fsa::LabelId* itLabel = labels + suffixId;
                    for (; suffixId < suffixIdEnd - 1; ++suffixId, ++itLabel) {
                        const PrefixTree::Range& range = SSuffix[suffixId];
                        *itLabel                       = range.first->first;
                        const PrefixTree::Node& node   = nodes[range.first->second];
                        SSuffix[suffixId + 1]          = std::make_pair(node.begin(), node.end());
                    }
                    {
                        PrefixTree::Range& range = SSuffix[suffixId];
                        for (; range.first != range.second; ++range.first) {
                            *itLabel                               = range.first->first;
                            const std::pair<u32, u32>& prefixIds   = leaves[range.first->second];
                            WordSuccessors&            wSuccessors = slot.wSuccessorsByPrefix[prefixIds.first];
                            wSuccessors.beginWord                  = new Word[1];
                            wSuccessors.endWord                    = wSuccessors.beginWord + 1;
                            wSuccessors.nSuffixStrings             = prefixIds.second;

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << prefixIds.first << ". w-successors (#words=" << 1 << ", #suffix-strings=" << prefixIds.second << ")" << std::endl;
#endif
                            wSuccessors.beginWord->label     = Fsa::Epsilon;
                            wSuccessors.beginWord->condScore = 0.0;
                            if (suffixIdEnd > 1) {
                                const PrefixTree::Leave& suffixIds =
                                        prefixTreePtr->lookupAndCount(beginSuffixLabel, beginSuffixLabel + suffixIdEnd - 1);
                                wSuccessors.beginWord->prefixId = prefixIds.first;
                                wSuccessors.beginWord->suffixId = suffixIds.first;
                                wSuccessors.beginWord->tailId   = suffixIds.second - 1;
                            }
                            else {
                                verify(suffixIdEnd == 1);
                                wSuccessors.beginWord->prefixId = prefixIds.first;
                                wSuccessors.beginWord->suffixId = 0;
                                wSuccessors.beginWord->tailId   = nFinalTails++;
                            }

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", wSuccessors.beginWord->prefixId, wSuccessors.beginWord->suffixId, wSuccessors.beginWord->tailId);
                            {
                                Fsa::LabelId* itLabel = beginPrefixLabel;
                                for (; itLabel != beginSuffixLabel + suffixIdEnd - 1; ++itLabel)
                                    std::cerr << " " << alphabet.symbol(*itLabel);
                                for (; itLabel != endSuffixLabel; ++itLabel)
                                    std::cerr << " $";
                                std::cerr << std::endl;
                            }
#endif

                            if (vRestricted_) {
                                WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix[prefixIds.first];
                                vSuccessors.nSuffixStrings  = wSuccessors.nSuffixStrings;
                                vSuccessors.beginWord       = new Word[1];
                                vSuccessors.endWord         = vSuccessors.beginWord + 1;
                                ::memcpy((void*)vSuccessors.beginWord, (void*)wSuccessors.beginWord, sizeof(Word));
                            }
                        }
                    }
                    for (--suffixId; 0 <= suffixId;) {
                        PrefixTree::Range& range = SSuffix[suffixId];
                        if (++range.first == range.second)
                            --suffixId;
                        else
                            break;
                    }
                }
                std::swap(lastPrefixTreePtr, prefixTreePtr);
                prefixTreePtr->reset();
            }
            // dummy=1
            {
                Slot& slot  = slots_.back();
                slot.slotId = slotId++;
                verify(slot.slotId == slots_.size() - 1);
                slot.wSuccessorsByPrefix.resize(1);
                WordSuccessors& wSuccessors = slot.wSuccessorsByPrefix.front();
                wSuccessors.nSuffixStrings  = nFinalTails;

                // dbg
#if COND_POST_DBG >= 3
                std::cerr << 0 << ". w-successors (#words=" << 1 << ", #suffix-strings=" << nFinalTails << ")" << std::endl;
#endif

                wSuccessors.beginWord            = new Word[1];
                wSuccessors.endWord              = wSuccessors.beginWord + 1;
                wSuccessors.beginWord->label     = Fsa::InvalidLabelId;
                wSuccessors.beginWord->condScore = 0.0;
                wSuccessors.beginWord->prefixId  = 0;
                wSuccessors.beginWord->suffixId  = 0;
                wSuccessors.beginWord->tailId    = 0;

                // dbg
#if COND_POST_DBG >= 3
                std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", wSuccessors.beginWord->prefixId, wSuccessors.beginWord->suffixId, wSuccessors.beginWord->tailId);
                {
                    Fsa::LabelId* itLabel = beginPrefixLabel;
                    for (; itLabel != endSuffixLabel; ++itLabel)
                        std::cerr << " $";
                    std::cerr << std::endl;
                }
#endif

                if (vRestricted_) {
                    slot.vSuccessorsByPrefix.resize(1);
                    WordSuccessors& vSuccessors = slot.vSuccessorsByPrefix.front();
                    vSuccessors.nSuffixStrings  = nFinalTails;
                    vSuccessors.nSuffixStrings  = vSuccessors.nSuffixStrings;
                    vSuccessors.beginWord       = new Word[1];
                    vSuccessors.endWord         = vSuccessors.beginWord + 1;
                    ::memcpy((void*)vSuccessors.beginWord, (void*)wSuccessors.beginWord, sizeof(Word));
                }
            }
            verify(slotId == slots_.size());
            verify(lastPrefixTreePtr->nLeaves() == 0);
            lastPrefixTreePtr->reset();

            // dbg
#if COND_POST_DBG >= 3
            std::cerr << "finished initializing w" << std::endl;
#endif
        }
        if (!vRestricted_) {
            /*
             * Initialize static search space/
             * Link word tuples
             */

            std::vector<LabelIdList> labelsInSlot(condPost.trees().size());
            {
                Core::Vector<bool> unique;
                Fsa::LabelId       maxLabel = 0;
                for (u32 i = 0; i < labelsInSlot.size(); ++i) {
                    LabelIdList&                           labels = labelsInSlot[i];
                    bool                                   hasEps = false;
                    const ConditionalPosterior::ValueList& values = condPost.trees()[i].values;
                    for (ConditionalPosterior::ValueList::const_iterator itValue = values.begin(), endValue = values.end();
                         itValue != endValue; ++itValue)
                        if (itValue->label == Fsa::Epsilon)
                            hasEps = true;
                        else if (itValue->label > maxLabel)
                            maxLabel = itValue->label;
                    verify(maxLabel != Fsa::InvalidLabelId);
                    unique.grow(maxLabel + 1, true);
                    for (ConditionalPosterior::ValueList::const_iterator itValue = values.begin(), endValue = values.end();
                         itValue != endValue; ++itValue)
                        if ((itValue->label >= 0) && (unique[itValue->label])) {
                            unique[itValue->label] = false;
                            labels.push_back(itValue->label);
                        }
                    for (LabelIdList::const_iterator itLabel = labels.begin(), endLabel = labels.end();
                         itLabel != endLabel; ++itLabel) {
                        verify(!unique[*itLabel]);
                        unique[*itLabel] = true;
                    }
                    if (hasEps)
                        labels.push_back(Fsa::Epsilon);
                }
            }
            std::fill(labels, labels + windowSize_, Fsa::Epsilon);
            PrefixTree *lastPrefixTreePtr = &prefixTree1, *prefixTreePtr = &prefixTree2;
            lastPrefixTreePtr->lookupAndCount(beginSuffixLabel, endSuffixLabel);
            verify(lastPrefixTreePtr->nLeaves() == 1);
            std::vector<std::pair<LabelIdList::const_iterator, LabelIdList::const_iterator>> S(windowSize_);
            s32                                                                              suffixIdOffset = contextSize;
            u32                                                                              slotId         = 0;
            for (; slotId < condPost.trees().size(); ++slotId) {
                // dbg
#if COND_POST_DBG >= 3
                std::cerr << "slot " << slotId << ", v-successors-by-prefix" << std::endl;
#endif

                Slot& slot = slots_[slotId];
                slot.vSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());
                const LabelIdList& nextLabelsInSlot = labelsInSlot[slotId + suffixIdOffset + 1 - windowSize_];
                S[suffixIdOffset]                   = std::make_pair(nextLabelsInSlot.begin(), nextLabelsInSlot.end());
                for (s32 suffixId = suffixIdOffset; suffixIdOffset <= suffixId;) {
                    Fsa::LabelId* itLabel = labels + suffixId;
                    for (; suffixId < contextSize; ++suffixId, ++itLabel) {
                        std::pair<LabelIdList::const_iterator, LabelIdList::const_iterator>& labelRange = S[suffixId];
                        *itLabel                                                                        = *labelRange.first;
                        const LabelIdList& nextLabelsInSlot                                             = labelsInSlot[slotId + suffixId + 2 - windowSize_];
                        S[suffixId + 1]                                                                 = std::make_pair(nextLabelsInSlot.begin(), nextLabelsInSlot.end());
                    }
                    {
                        const std::pair<u32, u32> prefixIds =
                                lastPrefixTreePtr->lookupExisting(beginPrefixLabel, endPrefixLabel);
                        WordSuccessors&                                                      vSuccessors = slot.vSuccessorsByPrefix[prefixIds.first];
                        std::pair<LabelIdList::const_iterator, LabelIdList::const_iterator>& labelRange  = S[suffixId];
                        u32                                                                  nWords      = labelRange.second - labelRange.first;
                        Word*                                                                itWord = vSuccessors.beginWord = new Word[nWords];
                        vSuccessors.endWord                                                                                 = vSuccessors.beginWord + nWords;
                        vSuccessors.nSuffixStrings                                                                          = prefixIds.second;

                        // dbg
#if COND_POST_DBG >= 3
                        std::cerr << prefixIds.first << ". v-successors (#words=" << nWords << ", #suffix-strings=" << prefixIds.second << ")" << std::endl;
#endif

                        for (LabelIdList::const_iterator itLabelInSlot  = labelRange.first,
                                                         endLabelInSlot = labelRange.second;
                             itLabelInSlot != endLabelInSlot; ++itLabelInSlot, ++itWord) {
                            *itLabel = *itLabelInSlot;
                            const std::pair<u32, u32> suffixIds =
                                    prefixTreePtr->lookupAndCount(beginSuffixLabel, endSuffixLabel);
                            itWord->label     = *itLabelInSlot;
                            itWord->prefixId  = prefixIds.first;
                            itWord->suffixId  = suffixIds.first;       // suffix
                            itWord->tailId    = suffixIds.second - 1;  // ith word with given suffix
                            itWord->condScore = Semiring::Zero;

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", itWord->prefixId, itWord->suffixId, itWord->tailId);
                            for (Fsa::LabelId* itLabel = beginPrefixLabel; itLabel != endSuffixLabel; ++itLabel)
                                std::cerr << " " << alphabet.symbol(*itLabel);
                            std::cerr << std::endl;
#endif
                        }
                    }
                    for (--suffixId; suffixIdOffset <= suffixId;) {
                        std::pair<LabelIdList::const_iterator, LabelIdList::const_iterator>& labelRange = S[suffixId];
                        if (++labelRange.first == labelRange.second)
                            --suffixId;
                        else
                            break;
                    }
                }
                if (suffixIdOffset > 0)
                    --suffixIdOffset;
                std::swap(lastPrefixTreePtr, prefixTreePtr);
                prefixTreePtr->reset();
            }
            // resolve suffix

            std::vector<PrefixTree::Range> SSuffix(windowSize_ - 1);
            s32                            suffixIdEnd = contextSize;
            u32                            nFinalTails = 0;
            for (; slotId < condPost.trees().size() + windowSize_ - 1; ++slotId, --suffixIdEnd) {
                // dbg
#if COND_POST_DBG >= 3
                std::cerr << "slot " << slotId << ", v-successors-by-prefix" << std::endl;
#endif

                Slot& slot = slots_[slotId];
                slot.vSuccessorsByPrefix.resize(lastPrefixTreePtr->nLeaves());
                const PrefixTree::NodeList&  nodes  = lastPrefixTreePtr->nodes;
                const PrefixTree::LeaveList& leaves = lastPrefixTreePtr->leaves;
                SSuffix[0]                          = std::make_pair(nodes.front().begin(), nodes.front().end());
                for (s32 suffixId = 0; 0 <= suffixId;) {
                    Fsa::LabelId* itLabel = labels + suffixId;
                    for (; suffixId < suffixIdEnd - 1; ++suffixId, ++itLabel) {
                        const PrefixTree::Range& range = SSuffix[suffixId];
                        *itLabel                       = range.first->first;
                        const PrefixTree::Node& node   = nodes[range.first->second];
                        SSuffix[suffixId + 1]          = std::make_pair(node.begin(), node.end());
                    }
                    {
                        PrefixTree::Range& range = SSuffix[suffixId];
                        for (; range.first != range.second; ++range.first) {
                            *itLabel                               = range.first->first;
                            const std::pair<u32, u32>& prefixIds   = leaves[range.first->second];
                            WordSuccessors&            vSuccessors = slot.vSuccessorsByPrefix[prefixIds.first];
                            vSuccessors.beginWord                  = new Word[1];
                            vSuccessors.endWord                    = vSuccessors.beginWord + 1;
                            vSuccessors.nSuffixStrings             = prefixIds.second;

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << prefixIds.first << ". w-successors (#words=" << 1 << ", #suffix-strings=" << prefixIds.second << ")" << std::endl;
#endif

                            vSuccessors.beginWord->label     = Fsa::Epsilon;
                            vSuccessors.beginWord->condScore = 0.0;
                            if (suffixIdEnd > 1) {
                                const PrefixTree::Leave& suffixIds =
                                        prefixTreePtr->lookupAndCount(beginSuffixLabel, beginSuffixLabel + suffixIdEnd - 1);
                                vSuccessors.beginWord->prefixId = prefixIds.first;
                                vSuccessors.beginWord->suffixId = suffixIds.first;
                                vSuccessors.beginWord->tailId   = suffixIds.second - 1;
                            }
                            else {
                                verify(suffixIdEnd == 1);
                                vSuccessors.beginWord->prefixId = prefixIds.first;
                                vSuccessors.beginWord->suffixId = 0;
                                vSuccessors.beginWord->tailId   = nFinalTails++;
                            }

                            // dbg
#if COND_POST_DBG >= 3
                            std::cerr << Core::form("  prefix=%2d, suffix=%2d, tail=%2d:", vSuccessors.beginWord->prefixId, vSuccessors.beginWord->suffixId, vSuccessors.beginWord->tailId);
                            {
                                Fsa::LabelId* itLabel = beginPrefixLabel;
                                for (; itLabel != beginSuffixLabel + suffixIdEnd - 1; ++itLabel)
                                    std::cerr << " " << alphabet.symbol(*itLabel);
                                for (; itLabel != endSuffixLabel; ++itLabel)
                                    std::cerr << " $";
                                std::cerr << std::endl;
                            }
#endif
                        }
                    }
                    for (--suffixId; 0 <= suffixId;) {
                        PrefixTree::Range& range = SSuffix[suffixId];
                        if (++range.first == range.second)
                            --suffixId;
                        else
                            break;
                    }
                }
                std::swap(lastPrefixTreePtr, prefixTreePtr);
                prefixTreePtr->reset();
            }
            {
                Slot& slot  = slots_.back();
                slot.slotId = slotId++;
                slot.vSuccessorsByPrefix.resize(1);
                WordSuccessors& vSuccessors      = slot.vSuccessorsByPrefix.front();
                vSuccessors.nSuffixStrings       = nFinalTails;
                vSuccessors.beginWord            = new Word[1];
                vSuccessors.endWord              = vSuccessors.beginWord + 1;
                vSuccessors.beginWord->label     = Fsa::Epsilon;
                vSuccessors.beginWord->condScore = 0.0;
                vSuccessors.beginWord->prefixId  = 0;
                vSuccessors.beginWord->suffixId  = 0;
                vSuccessors.beginWord->tailId    = 0;
            }
            verify(slotId == slots_.size());
            lastPrefixTreePtr->reset();
        }
        delete[] labels;
        /*
         * Initialize dynamic search space
         */

        SearchSpace& ss = ss1_;
        ss.vSuffixStrings.resize(1);
        ss.vSuffixStrings[0] = new Fsa::LabelId[windowSize_ - 1];
        std::fill(ss.vSuffixStrings[0], ss.vSuffixStrings[0] + windowSize_ - 1, Fsa::Epsilon);
        ss.wSuffixStrings.resize(1);
        ss.wSuffixStrings[0] = new Fsa::LabelId[windowSize_ - 1];
        std::fill(ss.wSuffixStrings[0], ss.wSuffixStrings[0] + windowSize_ - 1, Fsa::Epsilon);
        ss.vSuffixPtrs.resize(1);
        ss.vSuffixPtrs[0]   = new VSuffix(0, 1);
        VSuffix& vSuffix    = *ss.vSuffixPtrs[0];
        vSuffix.minVHeadPtr = vSuffix.beginVHeadPtr[0] = new VHead(Fsa::Epsilon, 1);
        VHead& vHead                                   = *vSuffix.beginVHeadPtr[0];
        vHead.beginWSuffixPtr[0]                       = new WSuffix(0, 0);
        WSuffix& wSuffix                               = *vHead.beginWSuffixPtr[0];
        wSuffix.sumAlignment.score                     = 0.0;
        wSuffix.sumAlignment.costs                     = new Alignment::Cost[windowSize_];
        for (Alignment::Cost *itCost = wSuffix.sumAlignment.costs, *endCost = wSuffix.sumAlignment.costs + windowSize_;
             itCost != endCost; ++itCost)
            itCost->score = 0.0;
    }
}

/*
 * perform search; needs to be initialized before
 */
void WindowedLevenshteinDistanceDecoder::search() {
    SearchSpace *           ss = &ss1_, *nextSs = &ss2_;
    Core::ProgressIndicator pi(Core::form("decode(%zu,window=%u)", slots_.size() - 1, windowSize_));
    pi.start(slots_.size());
    for (u32 slotId = 0; slotId < slots_.size() - 1; ++slotId) {
        extend(*nextSs, *ss);
        std::swap(ss, nextSs);
        nextSs->reset();
        collect(*ss);
        prune(*ss);
        pi.notify();
    }
    pi.finish(false);
    verify(result_->bestHyp.size() == cn_->size());
    result_->best = buildLattice();
#ifdef WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT
    verify((ss->vSuffixPtrs.size() == 1) && (ss->vSuffixPtrs.front()->minVHeadPtr));
    trace(*ss->vSuffixPtrs.front()->minVHeadPtr);
#endif
}

/*
 * decode lattice
 */
WindowedLevenshteinDistanceDecoder::ConstResultRef WindowedLevenshteinDistanceDecoder::decode(ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn) {
    l_ = l;
    initDecoder();
    ConditionalPosterior::Internal* condPost = condPosteriorBuilder_->build(l, fb, cn);
    cn_                                      = condPost->cn();
    ConstResultRef result = result_ = ResultRef(new Result);
    if (!condPost->trees().empty()) {
        initSearchSpace(*condPost);
        search();
        resetSearchSpace();
    }
    delete condPost;
    l_.reset();
    cn_.reset();
    result_.reset();
    return result;
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class WindowedLevenshteinDistanceDecoderNode : public Node {
public:
    static const Core::ParameterInt    paramContext;
    static const Core::ParameterBool   paramRestricted;
    static const Core::ParameterFloat  paramThreshold;
    static const Core::ParameterInt    paramMaxSlotSize;
    static const Core::ParameterInt    paramSupply;
    static const Core::ParameterInt    paramInterval;
    static const Core::ParameterString paramConfKey;

private:
    Core::XmlChannel dumpChannel_;

    u32                                   n_;
    Key                                   confidenceKey_;
    FwdBwdBuilderRef                      fbBuilder_;
    ConfusionNetworkFactoryRef            cnBuilder_;
    WindowedLevenshteinDistanceDecoderRef decoder_;

    WindowedLevenshteinDistanceDecoder::ConstResultRef mbrResult_;
    ConstLatticeRef                                    union_;
    ConstFwdBwdRef                                     fb_;
    ConstConfusionNetworkRef                           cn_;

    ConstSemiringRef lastSemiring_;
    ScoreId          confidenceId_;

private:
    ScoreId getConfidenceId(ConstSemiringRef semiring) {
        if (!lastSemiring_ || (semiring.get() != lastSemiring_.get()) || !(*semiring == *lastSemiring_)) {
            lastSemiring_ = semiring;
            if (!confidenceKey_.empty()) {
                confidenceId_ = semiring->id(confidenceKey_);
                if (confidenceId_ == Semiring::InvalidId)
                    warning("Semiring \"%s\" has no dimension labeled \"%s\".",
                            semiring->name().c_str(), confidenceKey_.c_str());
            }
        }
        return confidenceId_;
    }

    void dump(const WindowedLevenshteinDistanceDecoder::Result& mbrResult) {
        if (!dumpChannel_.isOpen())
            return;
        const Fsa::Alphabet& alphabet = *union_->getInputAlphabet();
        dumpChannel_ << Core::XmlOpen("minimum-bayes-risk");
        dumpChannel_ << Core::XmlFull("risk", mbrResult.bestRisk);
        dumpChannel_ << Core::XmlOpen("hypothesis");
        for (WindowedLevenshteinDistanceDecoder::Result::WordList::const_iterator
                     itWord  = mbrResult.bestHyp.begin(),
                     endWord = mbrResult.bestHyp.end();
             itWord != endWord; ++itWord)
            if (itWord->label != Fsa::Epsilon)
                dumpChannel_ << Core::form("%6.2f\t%s\n", itWord->risk, alphabet.symbol(itWord->label).c_str());
        dumpChannel_ << Core::XmlClose("hypothesis");
        dumpChannel_ << Core::XmlClose("minimum-bayes-risk");
    }

    void decode() {
        if (!mbrResult_) {
            ConstLatticeRefList lats(n_);
            for (u32 i = 0; i < n_; ++i)
                lats[i] = requestLattice(i);
            std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult =
                    (n_ == 1) ? fbBuilder_->build(lats.front()) : fbBuilder_->build(lats);
            union_ = fbResult.first;
            fb_    = fbResult.second;
            cnBuilder_->build(union_, fb_);
            std::pair<ConstConfusionNetworkRef, ConstLatticeRef> cnResult =
                    cnBuilder_->getNormalizedCn(getConfidenceId(union_->semiring()), true);
            cn_        = cnResult.first;
            mbrResult_ = decoder_->decode(union_, fb_, cn_);
            dump(*mbrResult_);
        }
    }

    u32 s32ToU32(s32 i) const {
        return (i == Core::Type<s32>::max) ? Core::Type<u32>::max : u32(i);
    }

public:
    WindowedLevenshteinDistanceDecoderNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config),
              dumpChannel_(config, "dump"),
              n_(0) {
        confidenceId_ = Semiring::InvalidId;
    }
    virtual ~WindowedLevenshteinDistanceDecoderNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        for (n_ = 0; connected(n_); ++n_)
            ;
        if (n_ == 0)
            criticalError("At least one incoming lattice at port 0 required.");
        Core::Component::Message msg = log();
        if (n_ > 1)
            msg << "Combine " << n_ << " lattices.\n\n";
        KeyList   requiredKeys;
        ScoreList requiredScales;
        confidenceKey_ = paramConfKey(config);
        if (!confidenceKey_.empty()) {
            msg << "Confidence key is \"" << confidenceKey_ << "\"\n";
            requiredKeys.push_back(confidenceKey_);
            requiredScales.push_back(0.0);
        }
        fbBuilder_ = FwdBwdBuilder::create(select("fb"), requiredKeys, requiredScales);
        cnBuilder_ = ConfusionNetworkFactory::create(select("cn"));
        msg << "CN builder:\n";
        cnBuilder_->dump(msg);
        decoder_ = WindowedLevenshteinDistanceDecoder::create();
        decoder_->setContextSize(paramContext(config));
        const Core::Configuration configSearchSpace(config, "search-space");
        decoder_->setVRestricted(paramRestricted(configSearchSpace));
        // pre-pruning
        const Core::Configuration configPrePruning(config, "pre-pruning");
        decoder_->setPrePruningThresholds(
                paramThreshold(configPrePruning, Core::Type<Score>::max),
                s32ToU32(paramMaxSlotSize(configPrePruning, Core::Type<s32>::max)));
        // pruning
        const Core::Configuration configPruning(config, "pruning");
        decoder_->setPruningInterval(
                s32ToU32(paramInterval(configPruning, Core::Type<s32>::max)),
                s32ToU32(paramSupply(configPruning, Core::Type<s32>::max)));
        decoder_->setPruningThreshold(paramThreshold(configPruning, Core::Type<Score>::max));
        msg << "Bayes risk decoder:\n";
        decoder_->dump(msg);
    }

    virtual void finalize() {}

    virtual ConstLatticeRef sendLattice(Port to) {
        decode();
        switch (to) {
            case 0:
                return mbrResult_->best;
            case 1:
                return union_;
            case 2:
                return mbrResult_->alignment;
            case 3:
                return mbrResult_->cost;
            default:
                defect();
                return ConstLatticeRef();
        }
    }

    virtual void sync() {
        mbrResult_.reset();
        cn_.reset();
        fb_.reset();
        union_.reset();
        cnBuilder_->reset();
    }
};
const Core::ParameterInt WindowedLevenshteinDistanceDecoderNode::paramContext(
        "context",
        "context size",
        2);
const Core::ParameterBool WindowedLevenshteinDistanceDecoderNode::paramRestricted(
        "restricted",
        "restricted",
        false);
const Core::ParameterFloat WindowedLevenshteinDistanceDecoderNode::paramThreshold(
        "threshold",
        "threshold");
const Core::ParameterInt WindowedLevenshteinDistanceDecoderNode::paramMaxSlotSize(
        "max-slot-size",
        "max. number of arcs in CN slot");
const Core::ParameterInt WindowedLevenshteinDistanceDecoderNode::paramSupply(
        "supply",
        "first action after supply steps");
const Core::ParameterInt WindowedLevenshteinDistanceDecoderNode::paramInterval(
        "interval",
        "action at each interval steps");
const Core::ParameterString WindowedLevenshteinDistanceDecoderNode::paramConfKey(
        "confidence-key",
        "store confidence score",
        "");
NodeRef createWindowedLevenshteinDistanceDecoderNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new WindowedLevenshteinDistanceDecoderNode(name, config));
}
// -------------------------------------------------------------------------

}  // namespace Flf
