#include <Core/Parameter.hh>
#include <Flf/Best.hh>
#include <Flf/Copy.hh>
#include <Flf/FlfCore/Basic.hh>
#include <Flf/FwdBwd.hh>
#include <Flf/Prune.hh>
#include <Flf/TimeAlignment.hh>
#include <Flf/TimeframeConfusionNetworkBuilder.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Project.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Sssp.hh>
#include <Fsa/Static.hh>

#include "MapDecoder.hh"

namespace Flf {

// -------------------------------------------------------------------------
class DecoderBase;
typedef Core::Ref<const DecoderBase> ConstDecoderBaseRef;
class DecoderBase : public Core::ReferenceCounted {
public:
    static const Core::ParameterBool  paramViterbi;
    static const Core::ParameterFloat paramAlpha;

private:
    bool                     viterbi_;
    f32                      alpha_;
    ConstSemiringRef         decodeSemiring_;
    TimeAlignmentBuilderRef  timeBoundaryBuilder_;
    mutable ConstSemiringRef lastSemiring_;
    mutable ConstSemiringRef newSemiring_;

public:
    DecoderBase(const Core::Configuration& config) {
        viterbi_             = paramViterbi(config);
        alpha_               = paramAlpha(config);
        decodeSemiring_      = Semiring::create(Fsa::SemiringTypeTropical, 1, ScoreList(1, 1.0), KeyList(1, "score"));
        timeBoundaryBuilder_ = TimeAlignmentBuilder::create(Core::Configuration(config, "time-boundaries"));
    }
    virtual ~DecoderBase() {}

    void dump(std::ostream& os) const {
        os << "Basic decoder setup:" << std::endl
           << "  viterbi=" << (viterbi_ ? "yes" : "no") << std::endl
           << "  alpha=" << alpha_ << std::endl;
        timeBoundaryBuilder_->dump(os);
    }

    bool isViterbi() const {
        return viterbi_;
    }

    f32 alpha() const {
        return alpha_;
    }

    Fsa::ConstAutomatonRef project(ConstLatticeRef l, Fsa::SemiringType semiringType) const {
        verify((semiringType == Fsa::SemiringTypeLog) || (semiringType == Fsa::SemiringTypeTropical));
        if (!lastSemiring_ || (lastSemiring_.get() != l->semiring().get())) {
            lastSemiring_ = l->semiring();
            newSemiring_  = (semiringType == Fsa::SemiringTypeLog) ? toLogSemiring(lastSemiring_, alpha_) : toTropicalSemiring(lastSemiring_);
        }
        l = changeSemiring(l, newSemiring_);
        return Fsa::staticCopy(Fsa::projectInput(toFsa(l)));
    }

    Fsa::ConstAutomatonRef determinize(Fsa::ConstAutomatonRef f) const {
        f = Fsa::staticCopy(Fsa::removeEpsilons(f));
        f = Fsa::staticCopy(Fsa::determinize(f));
        return f;
    }

    ConstLatticeRef best(Fsa::ConstAutomatonRef f) const {
        ConstLatticeRef l = fromFsa(f, decodeSemiring_, 0);
        l                 = copy(l);
        return Flf::best(l, ProjectingBellmanFord);
    }

    ConstLatticeRef timeBoundaries(ConstLatticeRef b, ConstLatticeRef t, ConstPosteriorCnRef fCn = ConstPosteriorCnRef()) const {
        return timeBoundaryBuilder_->align(b, t, fCn);
    }

    static ConstDecoderBaseRef create(const Core::Configuration& config) {
        return ConstDecoderBaseRef(new DecoderBase(config));
    }
};
const Core::ParameterBool DecoderBase::paramViterbi(
        "viterbi",
        "use Viterbi approximation",
        false);
const Core::ParameterFloat DecoderBase::paramAlpha(
        "alpha",
        "scale dimensions for posterior calculation",
        0.0);
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class MapDecoderNode : public FilterNode {
    typedef FilterNode Precursor;

private:
    ConstDecoderBaseRef decoderBase_;

protected:
    virtual ConstLatticeRef filter(ConstLatticeRef l) {
        Fsa::ConstAutomatonRef f = decoderBase_->project(l, (decoderBase_->isViterbi() ? Fsa::SemiringTypeTropical : Fsa::SemiringTypeLog));
        if (!decoderBase_->isViterbi())
            f = decoderBase_->determinize(f);
        ConstLatticeRef b = decoderBase_->best(f);
        return decoderBase_->timeBoundaries(b, l);
    }

public:
    MapDecoderNode(const std::string& name, const Core::Configuration& config)
            : Precursor(name, config) {}
    virtual ~MapDecoderNode() {}
    virtual void init(const std::vector<std::string>& arguments) {
        decoderBase_ = DecoderBase::create(config);
        Core::Component::Message msg(log());
        decoderBase_->dump(msg);
    }
};
NodeRef createMapDecoderNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new MapDecoderNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class IntersectionMapDecoderNode : public Node {
private:
    u32                 n_;
    FwdBwdBuilderRef    fbBuilder_;
    ConstDecoderBaseRef decoderBase_;
    ConstLatticeRef     result_;

private:
    ConstLatticeRef decode() {
        if (!result_) {
            ConstLatticeRefList             lats(n_);
            Core::Ref<Fsa::StaticAutomaton> first, intersection;
            for (u32 i = 0; i < n_; ++i) {
                lats[i]                  = requestLattice(i);
                Fsa::ConstAutomatonRef f = decoderBase_->project(lats[i], (decoderBase_->isViterbi() ? Fsa::SemiringTypeTropical : Fsa::SemiringTypeLog));
                f                        = decoderBase_->determinize(f);
                if (first) {
                    if (intersection) {
                        intersection = Fsa::staticCopy(Fsa::composeMatching(intersection, f));
                        Fsa::trimInPlace(intersection);
                        if (intersection && (intersection->initialStateId() == Fsa::InvalidStateId))
                            intersection.reset();
                    }
                }
                else
                    first = intersection = Fsa::staticCopy(f);
            }
            ConstLatticeRef b;
            if (first) {
                if (!intersection) {
                    warning("intersection result is empty; fall back to first system");
                    b = decoderBase_->best(first);
                }
                else
                    b = decoderBase_->best(intersection);
                std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult = fbBuilder_->build(lats);
                ConstPosteriorCnRef                        fCn      = buildFramePosteriorCn(fbResult.first, fbResult.second);
                result_                                             = decoderBase_->timeBoundaries(b, fbResult.first, fCn);
            }
        }
        return result_;
    }

public:
    IntersectionMapDecoderNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config),
              n_(0) {}
    virtual ~IntersectionMapDecoderNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        for (n_ = 0; connected(n_); ++n_)
            ;
        if (n_ == 0)
            criticalError("At least one incoming lattice at port 0 required.");
        Core::Component::Message msg = log();
        if (n_ > 1)
            msg << "Combine " << n_ << " lattices.\n\n";
        decoderBase_ = DecoderBase::create(config);
        decoderBase_->dump(msg);
        const Core::Configuration fcnConfig(config, "fcn");
        const Core::Configuration fcnFbConfig(fcnConfig, "fb");
        fbBuilder_ = FwdBwdBuilder::create(fcnFbConfig);
    }

    virtual void finalize() {}

    virtual ConstLatticeRef sendLattice(Port to) {
        verify(to == 0);
        return decode();
    }

    virtual void sync() {
        result_.reset();
    }
};
NodeRef createIntersectionMapDecoderNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new IntersectionMapDecoderNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class UnionMapDeocderNode : public Node {
public:
    static const Core::ParameterFloat paramWeight;

private:
    u32                 n_;
    std::vector<f32>    weights_;
    FwdBwdPrunerRef     fbPruner_;
    FwdBwdBuilderRef    fbBuilder_;
    ConstDecoderBaseRef decoderBase_;
    ConstLatticeRef     result_;

private:
    ConstLatticeRef decode() {
        if (!result_) {
            ConstLatticeRefList                  lats(n_);
            Core::Vector<Fsa::ConstAutomatonRef> fsas(n_);
            Core::Vector<Fsa::Weight>            initialWeights(n_);
            for (u32 i = 0; i < n_; ++i) {
                lats[i] = fbPruner_->prune(requestLattice(i), true);
                fsas[i] = decoderBase_->project(lats[i], (decoderBase_->isViterbi() ? Fsa::SemiringTypeTropical : Fsa::SemiringTypeLog));
                if (!decoderBase_->isViterbi())
                    fsas[i] = decoderBase_->determinize(fsas[i]);
                Fsa::Weight w(0);
                Fsa::posterior64(Fsa::changeSemiring(fsas[i], Fsa::LogSemiring), w);
                initialWeights[i] = Fsa::Weight(f32(w) + weights_[i]);
            }
            Fsa::ConstAutomatonRef u = Fsa::staticCopy(Fsa::unite(fsas, initialWeights));
            if (!decoderBase_->isViterbi()) {
                dbg("determinize union");

                u = decoderBase_->determinize(u);
            }
            ConstLatticeRef                            b        = decoderBase_->best(u);
            std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult = fbBuilder_->build(lats);
            ConstPosteriorCnRef                        fCn      = buildFramePosteriorCn(fbResult.first, fbResult.second);
            result_                                             = decoderBase_->timeBoundaries(b, fbResult.first, fCn);
        }
        return result_;
    }

public:
    UnionMapDeocderNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config),
              n_(0) {}
    virtual ~UnionMapDeocderNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        for (n_ = 0; connected(n_); ++n_)
            ;
        if (n_ == 0)
            criticalError("At least one incoming lattice at port 0 required.");
        Core::Component::Message msg = log();
        if (n_ > 1)
            msg << "Combine " << n_ << " lattices:\n";
        weights_.resize(n_);
        f32 norm = 0.0;
        for (u32 i = 0; i < n_; ++i) {
            const Core::Configuration latConfig(config, Core::form("lattice-%d", i));
            norm += weights_[i] = paramWeight(latConfig, 1.0);
        }
        verify(norm != 0.0);
        norm = 1.0 / norm;
        for (u32 i = 0; i < n_; ++i) {
            f32& w = weights_[i];
            w *= norm;
            msg << Core::form("%4d. lattice, weight=%.2f", i, w) << "\n";
            w = -::log(w);
        }
        decoderBase_ = DecoderBase::create(config);
        decoderBase_->dump(msg);
        fbPruner_ = FwdBwdPruner::create(select("prune"));
        const Core::Configuration fcnConfig(config, "fcn");
        const Core::Configuration fcnFbConfig(fcnConfig, "fb");
        fbBuilder_ = FwdBwdBuilder::create(fcnFbConfig);
    }

    virtual void finalize() {}

    virtual ConstLatticeRef sendLattice(Port to) {
        verify(to == 0);
        return decode();
    }

    virtual void sync() {
        result_.reset();
    }
};
const Core::ParameterFloat UnionMapDeocderNode::paramWeight(
        "weight",
        "lattice weight",
        1.0);
NodeRef createUnionMapDecoderNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new UnionMapDeocderNode(name, config));
}
// -------------------------------------------------------------------------

}  // namespace Flf
