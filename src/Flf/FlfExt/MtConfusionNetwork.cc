#include <Core/Application.hh>
#include <Core/Channel.hh>
#include <Flf/ConfusionNetwork.hh>
#include <Flf/ConfusionNetworkIo.hh>
#include <Flf/FlfCore/Basic.hh>
#include <Flf/FlfCore/LatticeInternal.hh>
#include <Flf/FlfCore/Utility.hh>
#include <Flf/Lexicon.hh>
#include <Flf/RescoreInternal.hh>

#include "MtConfusionNetwork.hh"

namespace Flf {

// -------------------------------------------------------------------------
class MtCnFeatureLattice;
typedef Core::Ref<MtCnFeatureLattice> MtCnFeatureLatticeRef;

class MtCnFeatureLattice : public RescoreLattice {
    typedef RescoreLattice Precursor;

public:
    struct FeatureIds {
        // CN
        ScoreId cnPosteriorId;
        // lattice features
        ScoreId confidenceId;
        ScoreId scoreId;
        ScoreId slotEntropyId;
        ScoreId slotId;
        ScoreId nonEpsSlotId;
        Score   epsSlotThreshold;
        FeatureIds()
                : cnPosteriorId(Semiring::InvalidId),
                  confidenceId(Semiring::InvalidId),
                  scoreId(Semiring::InvalidId),
                  slotEntropyId(Semiring::InvalidId),
                  slotId(Semiring::InvalidId),
                  nonEpsSlotId(Semiring::InvalidId),
                  epsSlotThreshold(1.0) {}
    };

public:
    ConstConfusionNetworkRef cn_;
    FeatureIds               ids_;
    ConstConfusionNetworkRef normalizedCn_;
    bool                     needsPosterior_;
    StateIdList              slotIdToNonEpsilonSlotIdMap_;
    u32                      nNonEpsSlots_;

    Fsa::ConstAlphabetRef alphabet_;
    Lexicon::SymbolMap    symbolMap_;

public:
    MtCnFeatureLattice(ConstLatticeRef l, ConstConfusionNetworkRef cn, RescoreMode rescoreMode, const FeatureIds& ids)
            : Precursor(l, rescoreMode),
              cn_(cn),
              ids_(ids) {
        verify(cn_->hasMap());
        if ((ids_.confidenceId != Semiring::InvalidId) || (ids_.scoreId != Semiring::InvalidId) || (ids_.slotEntropyId != Semiring::InvalidId) || (ids_.nonEpsSlotId != Semiring::InvalidId))
            normalizedCn_ = (cn_->isNormalized()) ? cn_ : normalizeCn(cn_, ids_.cnPosteriorId);
        needsPosterior_ = (ids_.confidenceId != Semiring::InvalidId) || (ids_.scoreId != Semiring::InvalidId);
        nNonEpsSlots_   = Core::Type<u32>::max;
        if (ids_.nonEpsSlotId != Semiring::InvalidId) {
            slotIdToNonEpsilonSlotIdMap_.resize(cn->size(), Fsa::InvalidStateId);
            ScoreId               nonEpsSlotId       = 0;
            StateIdList::iterator itNonEpsilonSlotId = slotIdToNonEpsilonSlotIdMap_.begin();
            for (ConfusionNetwork::const_iterator itSlot = cn->begin(), endSlot = cn->end(), itNormalizedSlot = normalizedCn_->begin();
                 itSlot != endSlot; ++itSlot, ++itNormalizedSlot, ++itNonEpsilonSlotId) {
                if ((itNormalizedSlot->front().label == Fsa::Epsilon) && (itNormalizedSlot->front().scores->get(ids_.cnPosteriorId) >= ids_.epsSlotThreshold))
                    continue;
                ConfusionNetwork::Slot::const_iterator itArc = itSlot->begin(), endArc = itSlot->end();
                for (; (itArc != endArc) && (itArc->label == Fsa::Epsilon); ++itArc)
                    ;
                if (itArc != endArc)
                    *itNonEpsilonSlotId = nonEpsSlotId++;
            }
            nNonEpsSlots_ = nonEpsSlotId;
            alphabet_     = l->getInputAlphabet();
            symbolMap_    = Lexicon::us()->symbolMap(Lexicon::us()->alphabetId(alphabet_, true));

#if 0
            // Test word prefixing
            {
                ConfusionNetwork &modCn = const_cast<ConfusionNetwork&>(*cn);
                StateIdList::iterator itNonEpsilonSlotId = slotIdToNonEpsilonSlotIdMap_.begin();
                for (ConfusionNetwork::iterator itSlot = modCn.begin(), endSlot = modCn.end(); itSlot != endSlot; ++itSlot, ++itNonEpsilonSlotId)
                    for (ConfusionNetwork::Slot::iterator itArc = itSlot->begin(), endArc = itSlot->end(); itArc != endArc; ++itArc)
                        if ((itArc->label != Fsa::Epsilon) && (*itNonEpsilonSlotId != Fsa::InvalidStateId))
                            itArc->label = symbolMap_.index(Core::form("%d_%s", *itNonEpsilonSlotId, alphabet_->symbol(itArc->label).c_str()));

                writeConfusionNetworkAsText(std::cout, cn, ConstSegmentRef());
                std::cout << std::endl;
            }
#endif
        }
    }
    virtual ~MtCnFeatureLattice() {}

    virtual void rescore(State* sp) const {
        ConfusionNetwork::MapProperties::Map::const_iterator itMap = cn_->mapProperties->state(sp->id());
        for (State::iterator a = sp->begin(); a != sp->end(); ++a, ++itMap) {
            const ConfusionNetwork::MapProperties::Mapping& toCn = *itMap;
            if (toCn.sid == Fsa::InvalidStateId) {
                if (ids_.confidenceId != Semiring::InvalidId)
                    a->weight_->set(ids_.confidenceId, 1.0);
                if (ids_.scoreId != Semiring::InvalidId)
                    a->weight_->set(ids_.confidenceId, Semiring::One);
                if (ids_.slotEntropyId != Semiring::InvalidId)
                    a->weight_->set(ids_.slotEntropyId, Semiring::Invalid);
                if (ids_.slotId != Semiring::InvalidId)
                    a->weight_->set(ids_.slotId, Semiring::Invalid);
                if (ids_.nonEpsSlotId != Semiring::InvalidId)
                    a->weight_->set(ids_.nonEpsSlotId, Semiring::Invalid);
            }
            else {
                if (needsPosterior_) {
                    Score posterior = normalizedCn_->normalizedProperties->posteriorScore((*normalizedCn_)[toCn.sid], a->input());
                    if (ids_.confidenceId != Semiring::InvalidId)
                        a->weight_->set(ids_.confidenceId, posterior);
                    if (ids_.scoreId != Semiring::InvalidId)
                        a->weight_->set(ids_.scoreId, -::log(posterior));
                }
                if (ids_.slotEntropyId != Semiring::InvalidId) {
                    const ConfusionNetwork::Slot& slot = (*normalizedCn_)[toCn.sid];
                    Score                         e    = 0.0;
                    for (ConfusionNetwork::Slot::const_iterator itArc = slot.begin(), endArc = slot.end(); itArc != endArc; ++itArc) {
                        Score p = itArc->scores->get(ids_.cnPosteriorId);
                        e += p * ::log(p);
                    }
                    e = -e;
                    a->weight_->set(ids_.slotEntropyId, e);
                }
                if (ids_.slotId != Semiring::InvalidId) {
                    a->weight_->set(ids_.slotId, Score(toCn.sid));
                }
                if (ids_.nonEpsSlotId != Semiring::InvalidId) {
                    if (a->input() == Fsa::Epsilon) {
                        verify_(((*cn_)[toCn.sid].begin() + toCn.aid)->label == Fsa::Epsilon);
                        a->weight_->set(ids_.nonEpsSlotId, Semiring::Invalid);
                    }
                    else {
                        verify_(((*cn_)[toCn.sid].begin() + toCn.aid)->label != Fsa::Epsilon);
                        Fsa::StateId nonEpsSlotId = slotIdToNonEpsilonSlotIdMap_[toCn.sid];
                        if (nonEpsSlotId == Fsa::InvalidStateId) {
                            a->weight_->set(ids_.nonEpsSlotId, Semiring::Invalid);
                            a->input_ = Fsa::Epsilon;
                        }
                        else {
                            a->weight_->set(ids_.nonEpsSlotId, Score(nonEpsSlotId));
                            // prefix label with slot id
                            a->input_ = symbolMap_.index(Core::form("%d_%s", nonEpsSlotId, alphabet_->symbol(a->input_).c_str()));
                        }
                    }
                }
            }
        }
    }

    virtual std::string describe() const {
        return Core::form("addMtCnFeatures(%s)", fsa_->describe().c_str());
    }
};

class MtCnFeatureNode : public RescoreNode {
    typedef RescoreNode Precursor;

public:
    static const Core::ParameterString paramPosteriorKey;

    static const Core::ParameterString paramKey;
    static const Core::ParameterFloat  paramThreshold;

private:
    Core::Channel alignedBestChannel_;

    std::string cnPosteriorKey_;

    std::string confidenceKey_;
    std::string scoreKey_;
    std::string slotEntropyKey_;
    std::string slotKey_;
    std::string nonEpsSlotKey_;

    mutable ConstSemiringRef               lastCnSemiring_;
    mutable ConstSemiringRef               lastSemiring_;
    mutable MtCnFeatureLattice::FeatureIds lastIds_;

protected:
    struct TraceElement {
        Score        score;
        Fsa::StateId bptr;
        Fsa::StateId aid;
        TraceElement()
                : score(Semiring::Max),
                  bptr(Fsa::InvalidStateId),
                  aid(Fsa::InvalidStateId) {}
    };
    typedef std::vector<TraceElement> Traceback;

    void dumpAlignedBest(
            std::ostream&                          os,
            ConstLatticeRef                        l,
            const ConfusionNetwork::MapProperties& mapProperties,
            const StateIdList&                     slotIdToNonEpsilonSlotIdMap,
            u32                                    nNonEpsSlots,
            ConstSegmentRef                        segment) {
        const Fsa::Alphabet& alphabet        = *l->getInputAlphabet();
        const Semiring&      semiring        = *l->semiring();
        ConstStateMapRef     topologicalSort = sortTopologically(l);
        Fsa::StateId         initialSid      = topologicalSort->front();
        Traceback            traceback(topologicalSort->maxSid + 1);
        traceback[initialSid].score = 0.0;
        TraceElement bestTrace;
        for (u32 i = 0; i < topologicalSort->size(); ++i) {
            Fsa::StateId        sid          = (*topologicalSort)[i];
            const TraceElement& currentTrace = traceback[sid];
            ConstStateRef       sr           = l->getState(sid);
            if (sr->isFinal()) {
                Score score = currentTrace.score + semiring.project(sr->weight());
                if (score < bestTrace.score) {
                    bestTrace.score = score;
                    bestTrace.bptr  = sid;
                }
            }
            Fsa::StateId aid = 0;
            for (State::const_iterator a = sr->begin(); a != sr->end(); ++a, ++aid) {
                Score         score = currentTrace.score + semiring.project(a->weight());
                TraceElement& trace = traceback[a->target()];
                if (score < trace.score) {
                    trace.score = score;
                    trace.bptr  = sid;
                    trace.aid   = aid;
                }
            }
        }
        verify(bestTrace.bptr != Fsa::InvalidStateId);
        LabelIdList  result(nNonEpsSlots, Fsa::Epsilon);
        Fsa::StateId bestSid = bestTrace.bptr;
        while (bestSid != initialSid) {
            const TraceElement& trace  = traceback[bestSid];
            ConstStateRef       sr     = l->getState(trace.bptr);
            Fsa::StateId        slotId = mapProperties.slotArc(trace.bptr, trace.aid).sid;

            if (slotId != Fsa::InvalidStateId) {
                ScoreId nonEpsSlotId = slotIdToNonEpsilonSlotIdMap[slotId];
                if (nonEpsSlotId != Fsa::InvalidStateId) {
                    const Arc& arc = *(l->getState(trace.bptr)->begin() + trace.aid);
                    verify(nonEpsSlotId < result.size());
                    result[nonEpsSlotId] = arc.input();
                }
            }
            bestSid = trace.bptr;
        }
        printSegmentHeader(os, segment);
        os << nNonEpsSlots << "\t";
        for (LabelIdList::const_iterator itLabel = result.begin(), endLabel = result.end(); itLabel != endLabel; ++itLabel)
            os << alphabet.symbol(*itLabel) << " ";
        os << std::endl
           << std::endl;
    }

    ConstLatticeRef rescore(ConstLatticeRef l) {
        ConstConfusionNetworkRef cn = requestCn(1);
        if (!l)
            return ConstLatticeRef();
        if (!cn) {
            warning("No CN provided for lattice \"%s\"; skip lattice",
                    l->describe().c_str());
            return ConstLatticeRef();
        }
        if (!cn->hasMap())
            criticalError("CN for lattice \"%s\" does not provide a mapping.",
                          l->describe().c_str());
        if (!lastCnSemiring_ || (lastCnSemiring_.get() != cn->semiring.get())) {
            lastCnSemiring_        = cn->semiring;
            lastIds_.cnPosteriorId = lastCnSemiring_->id(cnPosteriorKey_);
            /*
            if (lastIds_.cnPosteriorId == Semiring::InvalidId)
                criticalError("CN semiring \"%s\" does not provide posterior score dimension \"%s\"",
                              lastCnSemiring_->name().c_str(), cnPosteriorKey_.c_str());
            */
        }
        if (!lastSemiring_ || (lastSemiring_.get() != l->semiring().get())) {
            lastSemiring_          = l->semiring();
            lastIds_.confidenceId  = lastSemiring_->id(confidenceKey_);
            lastIds_.scoreId       = lastSemiring_->id(scoreKey_);
            lastIds_.slotEntropyId = lastSemiring_->id(slotEntropyKey_);
            lastIds_.slotId        = lastSemiring_->id(slotKey_);
            lastIds_.nonEpsSlotId  = lastSemiring_->id(nonEpsSlotKey_);
        }
        MtCnFeatureLatticeRef e       = MtCnFeatureLatticeRef(new MtCnFeatureLattice(l, cn, rescoreMode, lastIds_));
        ConstSegmentRef       segment = connected(2) ? requestSegment(2) : ConstSegmentRef();
        if (alignedBestChannel_.isOpen() && (e->nNonEpsSlots_ != Core::Type<u32>::max))
            dumpAlignedBest(alignedBestChannel_, l, *e->cn_->mapProperties, e->slotIdToNonEpsilonSlotIdMap_, e->nNonEpsSlots_, segment);
        return e;
    }

public:
    MtCnFeatureNode(const std::string& name, const Core::Configuration& config)
            : Precursor(name, config),
              alignedBestChannel_(config, "best") {}
    ~MtCnFeatureNode() {}

    void init(const std::vector<std::string>& arguments) {
        if (!connected(0))
            criticalError("Need a data source at port 0.");
        if (!connected(1))
            criticalError("Need a CN at port 1.");
        Core::Component::Message msg(log());
        cnPosteriorKey_ = paramPosteriorKey(select("cn"));
        if (cnPosteriorKey_.empty())
            msg << "CN posterior key: " << cnPosteriorKey_ << "\n";
        // else criticalError("Definition of a key pointing at a posterior probability distribution in the CN is mandatory.");
        msg << "Store the following CN features:\n";
        confidenceKey_ = paramKey(select("confidence"));
        if (!confidenceKey_.empty())
            msg << "  - confidence to dimension \"" << confidenceKey_ << "\"\n";
        scoreKey_ = paramKey(select("score"));
        if (!scoreKey_.empty())
            msg << "  - score to dimension \"" << scoreKey_ << "\"\n";
        slotEntropyKey_ = paramKey(select("entropy"));
        if (!slotEntropyKey_.empty())
            msg << "  - slot entropy to dimension \"" << slotEntropyKey_ << "\"\n";
        slotKey_ = paramKey(select("slot"));
        if (!slotKey_.empty())
            msg << "  - slot number to dimension \"" << slotKey_ << "\"\n";
        nonEpsSlotKey_ = paramKey(select("non-eps-slot"));
        if (!nonEpsSlotKey_.empty()) {
            lastIds_.epsSlotThreshold = paramThreshold(select("non-eps-slot"), 1.0);
            msg << "  - non-epsilon-slot number to dimension \"" << slotKey_ << "\"\n";
            msg << "    epsilon-slot threshold is \"" << lastIds_.epsSlotThreshold << "\"\n";
        }
    }
};
const Core::ParameterString MtCnFeatureNode::paramPosteriorKey(
        "posterior-key",
        "posterior key",
        "");
const Core::ParameterString MtCnFeatureNode::paramKey(
        "key",
        "key",
        "");
const Core::ParameterFloat MtCnFeatureNode::paramThreshold(
        "threshold",
        "threshold");
NodeRef createMtCnFeatureNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new MtCnFeatureNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
void evgenyEpsSlots(ConstConfusionNetworkRef cnRef, Score threshold) {
    if (!cnRef)
        return;
    if (!cnRef->isNormalized())
        Core::Application::us()->criticalError("Epsilon slot removal does only work for normalized CNs.");
    ScoreId                    posteriorId = (threshold != Core::Type<Score>::max) ? cnRef->normalizedProperties->posteriorId : Semiring::InvalidId;
    ConfusionNetwork&          cn          = const_cast<ConfusionNetwork&>(*cnRef);
    ConfusionNetwork::iterator itTo        = cn.begin();
    for (ConfusionNetwork::iterator itFrom = cn.begin(), endSlot = cn.end(); itFrom != endSlot; ++itFrom) {
        const ConfusionNetwork::Slot& from = *itFrom;
        if ((from.front().label != Fsa::Epsilon) || ((from.size() > 1) && ((posteriorId == Semiring::InvalidId) || (from.front().scores->get(posteriorId) < threshold)))) {
            if (itTo != itFrom)
                *itTo = from;
            ++itTo;
        }
    }
    cn.erase(itTo, cn.end());
    u32                i         = 0;
    Lexicon::SymbolMap symbolMap = Lexicon::us()->symbolMap(Lexicon::us()->alphabetId(cn.alphabet, true));
    for (ConfusionNetwork::iterator itSlot = cn.begin(), endSlot = cn.end(); itSlot != endSlot; ++itSlot, ++i)
        for (ConfusionNetwork::Slot::iterator itArc = itSlot->begin(), endArc = itSlot->end(); itArc != endArc; ++itArc)
            if (itArc->label != Fsa::Epsilon)
                itArc->label = symbolMap.index(Core::form("%d_%s", i, cn.alphabet->symbol(itArc->label).c_str()));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class MtCnPruningNode : public Node {
    typedef Node Precursor;

public:
    static const Core::ParameterFloat paramThreshold;
    static const Core::ParameterInt   paramMaxSlotSize;
    static const Core::ParameterBool  paramNormalize;
    static const Core::ParameterBool  paramRemoveEpsSlots;

protected:
    bool  prune_;
    Score threshold_;
    u32   maxSlotSize_;
    bool  normalize_;
    bool  rmEpsSlots_;
    Score epsSlotThreshold_;

public:
    MtCnPruningNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config) {}
    virtual ~MtCnPruningNode() {}
    virtual void init(const std::vector<std::string>& arguments) {
        threshold_ = paramThreshold(config);
        verify(threshold_ > 0.0);
        maxSlotSize_ = paramMaxSlotSize(config);
        verify(maxSlotSize_ > 0);
        normalize_                   = paramNormalize(config);
        prune_                       = (threshold_ != Core::Type<Score>::max) || (maxSlotSize_ != Core::Type<u32>::max);
        rmEpsSlots_                  = paramRemoveEpsSlots(config);
        epsSlotThreshold_            = paramThreshold(select("eps-slot-removal"));
        Core::Component::Message msg = log();
        if (prune_) {
            msg << "Prune";
            if (threshold_ != Core::Type<Score>::max)
                msg << ", threshold = " << threshold_;
            if (maxSlotSize_ != Core::Type<u32>::max)
                msg << ", max. slot size = " << maxSlotSize_;
            msg << "\n";
            if (normalize_)
                msg << "Re-normalize slot-wise posterior prob. dist. after pruning.\n";
        }
        if (rmEpsSlots_) {
            msg << "Remove epsilon slots";
            if (epsSlotThreshold_ != Core::Type<Score>::max)
                msg << ", threshold = " << epsSlotThreshold_;
            msg << "\n";
        }
    }
};
const Core::ParameterFloat MtCnPruningNode::paramThreshold(
        "threshold",
        "probability threshold",
        Core::Type<Score>::max);
const Core::ParameterInt MtCnPruningNode::paramMaxSlotSize(
        "max-slot-size",
        "max. slot size",
        Core::Type<u32>::max);
const Core::ParameterBool MtCnPruningNode::paramNormalize(
        "normalize",
        "normalize",
        true);
const Core::ParameterBool MtCnPruningNode::paramRemoveEpsSlots(
        "remove-eps-slots",
        "remove eps slots",
        false);

class MtNormalizedCnPruningNode : public MtCnPruningNode {
    typedef MtCnPruningNode Precursor;

private:
    ConstConfusionNetworkRef cn_;

    void get() {
        if (!cn_) {
            cn_ = requestCn(0);
            if (cn_) {
                if (rmEpsSlots_)
                    evgenyEpsSlots(cn_, epsSlotThreshold_);
            }
        }
    }

public:
    MtNormalizedCnPruningNode(const std::string& name, const Core::Configuration& config)
            : Precursor(name, config) {}
    virtual ~MtNormalizedCnPruningNode() {}

    virtual ConstLatticeRef sendLattice(Port to) {
        verify(to == 1);
        get();
        return cn2lattice(cn_);
    }

    virtual ConstConfusionNetworkRef sendCn(Port to) {
        verify(to == 0);
        get();
        return cn_;
    }

    virtual void sync() {
        cn_.reset();
    }
};
NodeRef createMtNormalizedCnPruningNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new MtNormalizedCnPruningNode(name, config));
}
// -------------------------------------------------------------------------

}  // namespace Flf
