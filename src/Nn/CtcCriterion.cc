/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "CtcCriterion.hh"
#include <Am/Module.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Types.hh>
#include <Core/Utility.hh>
#include <Flow/ArchiveWriter.hh>
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Output.hh>
#include <Fsa/Project.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Semiring.hh>
#include <Fsa/Semiring64.hh>
#include <Fsa/Sssp.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/Module.hh>
#include <Nn/FeedForwardTrainer.hh>
#include <Nn/Prior.hh>
#include <Nn/Types.hh>
#include <Search/Aligner.hh>
#include <Speech/AllophoneStateGraphBuilder.hh>
#include <Speech/Feature.hh>
#include <Speech/Module.hh>
#include <memory>
#include <vector>
#include "CtcTimeAlignedAutomaton.hh"

static const Core::ParameterBool paramInputInLogSpace(
        "input-in-log-space",
        "Whether the input of the criterion, i.e. the output of the NN "
        "is in +log-space or not. If you applied the softmax, then it is not.",
        false);

static const Core::ParameterBool paramUseSearchAligner(
        "use-search-aligner",
        "Use Search::Aligner to build the time-aligned automaton, "
        "i.e. Search::Aligner::getAlignmentFsa(). "
        "Otherwise, we have our own custom implementation, TimeAlignedAutomaton.",
        true);

static const Core::ParameterFloat paramMinAcousticPruningThreshold(
        "min-acoustic-pruning",
        "minimal threshold for pruning of state hypotheses (used by TimeAlignedAutomaton)",
        50, Core::Type<f32>::delta);

static const Core::ParameterFloat paramMaxAcousticPruningThreshold(
        "max-acoustic-pruning",
        "maximal threshold for pruning of state hypotheses (used by TimeAlignedAutomaton)",
        Core::Type<f32>::max, 0.0);

static const Core::ParameterBool paramUseDirectAlignmentExtraction(
        "use-direct-alignment-extraction",
        "Use TimeAlignedAutomaton::getAlignment(). "
        "This will automatically calculate the posterior scores (fwd-bwd). "
        "Otherwise, Fsa::posterior64 would have been used.",
        false);  // false by default to not change old behavior. however, to use this might make more sense.

static const Core::ParameterFloat paramStatePosteriorScale(
        "state-posterior-scale", "", 1.0);

static const Core::ParameterFloat paramStatePosteriorLogBackoff(
        "state-posterior-log-backoff",
        "State posterior probability will always be >= backoff. "
        "This is in -log space, thus is this will be the upper limit.",
        /* big number */ 1.0f / Core::Type<f32>::epsilon);

static const Core::ParameterBool paramPosteriorUseSearchAligner(
        "posterior-use-search-aligner", "", false);

static const Core::ParameterBool paramPosteriorTotalNormalize(
        "posterior-total-normalize", "", false);

static const Core::ParameterFloat paramPosteriorArcLogThreshold(
        "posterior-arc-log-threshold", "", Core::Type<f64>::max);

static const Core::ParameterFloat paramPosteriorScale(
        "posterior-scale", "", 1.0);

static const Core::ParameterInt paramPosteriorNBestLimit(
        "posterior-nbest-limit", "", Core::Type<Core::ParameterInt::Value>::max);

static const Core::ParameterBool paramUseCrossEntropyAsLoss(
        "use-cross-entropy-as-loss",
        "The CTC loss is L_ctc = -ln \\sum_a \\prod_t ... p(...|x_t). "
        "We calculate p(a_t=a|x,w) along the lines. "
        "We can use that to create the Cross-Entropy instead, i.e. "
        "L_ce = - \\sum_t,a p(a_t=a|x,w) ln y_t,a "
        "where y_t,a is the NN output. "
        "This will make the loss compareable to normal Cross-Entropy, "
        "and also should be a more stable/normalized value across several segments "
        "(when normalized with the segment length).",
        false);

// fixed mixture set
// (mostly useful for debugging)

static const Core::ParameterBool paramUseFixedMixtureSet(
        "use-fixed-mixture-set",
        "use FeatureScorer for given mixture set instead of using "
        "state posterior probabilities from the model itself",
        false);

static const Core::ParameterString paramFixedMixtureSetSelector(
        "fixed-mixture-set-selector", "config selector", "mixture-set");

static const Core::ParameterString paramFixedMixtureSetFeatureExtractionSelector(
        "fixed-mixture-set-feature-extraction-selector", "config selector", "feature-extraction");

static const Core::ParameterString paramFixedMixtureSetExtractAlignmentsPortName(
        "fixed-mixture-set-extract-alignments", "port name for alignments", "");

// debug:

static const Core::ParameterString paramDumpReferenceProbs(
        "dump-reference-probs", "cache file name", "");

static const Core::ParameterString paramDumpViterbiAlignments(
        "dump-viterbi-alignments", "cache file name", "");

static const Core::ParameterBool paramDebugDumps(
        "debug-dumps", "true/false", false);

static const Core::ParameterBool paramLogTimeStatistics(
        "log-time-statistics", "log time stats", false);

// Also see CombinedExactSegmentwiseMeTrainer,
// the code is somewhat relevant here because we iterate over
// multiple alignments there.
// That code is used in  AcousticModelTrainer, which is more
// for GMM training and not for NN training.

namespace Fsa {

class RemoveInvalidArcsAutomaton : public ModifyAutomaton {
private:
    class HasInvalidArcWeight {
    private:
        const f32 threshold_;

    public:
        HasInvalidArcWeight(f32 threshold = 1000)
                : threshold_(threshold) {}
        bool operator()(const Arc& a) const {
            return (f32(a.weight()) > threshold_);
        }
    };

public:
    RemoveInvalidArcsAutomaton(ConstAutomatonRef fsa)
            : ModifyAutomaton(fsa) {}
    virtual ~RemoveInvalidArcsAutomaton() {}

    virtual std::string describe() const {
        return "remove-invalid-arcs(" + fsa_->describe() + ")";
    }
    virtual void modifyState(State* sp) const {
        HasInvalidArcWeight pred;
        sp->remove(pred);
    }
};

static ConstAutomatonRef _removeInvalidArcs(ConstAutomatonRef fsa) {
    RemoveInvalidArcsAutomaton* rf =
            new RemoveInvalidArcsAutomaton(fsa);
    return ConstAutomatonRef(rf);
}

}  //namespace Fsa

template<int N>
struct TimeStats {
    bool             active;
    Core::XmlWriter& channel;
    const char*      name;
    const char*      names[N];
    timeval          times[N];
    int              n;
    TimeStats(bool active_, Core::XmlWriter& channel_, const char* name_)
            : active(active_), channel(channel_), name(name_), n(0) {
        checkpoint("start");
    }
    void checkpoint(const char* name) {
        if (!active)
            return;
        require_lt(n, N);
        names[n] = name;
        TIMER_START(times[n]);
        ++n;
    }
    ~TimeStats() {
        if (!active)
            return;
        checkpoint("final");
        channel << Core::XmlOpen("time-statistics") + Core::XmlAttribute("name", name);
        double total  = Core::timeDiff(times[0], times[n - 1]);
        double totalS = std::max(total, Core::Type<double>::epsilon);
        for (int i = 1; i < n; ++i) {
            double td  = Core::timeDiff(times[i - 1], times[i]);
            double rel = td / totalS;
            channel << Core::XmlEmpty(names[i - 1]) + Core::XmlAttribute("time", td) + Core::XmlAttribute("relative", rel);
        }
        channel << Core::XmlEmpty("total") + Core::XmlAttribute("time", total);
        channel << Core::XmlClose("time-statistics");
    }
};

namespace Nn {

template<typename FloatT>
struct CalcTypes;

template<>
struct CalcTypes<f32> {
    typedef Fsa::Semiring         Semiring;
    typedef Fsa::ConstSemiringRef ConstSemiringRef;
    static ConstSemiringRef       logSemiring() {
        return Fsa::LogSemiring;
    }  // -log space
};

template<>
struct CalcTypes<f64> {
    typedef Fsa::Semiring64         Semiring;
    typedef Fsa::ConstSemiring64Ref ConstSemiringRef;
    static ConstSemiringRef         logSemiring() {
        return Fsa::LogSemiring64;
    }  // -log space
};

typedef Speech::TimeframeIndex TimeIndex;

template<typename FloatT>
struct ClassProbsExtractor {
    typedef typename Types<FloatT>::NnMatrix             NnMatrix;
    typedef typename CalcTypes<FloatT>::Semiring         Semiring;
    typedef typename Semiring::Weight                    Weight;
    typedef typename CalcTypes<FloatT>::ConstSemiringRef ConstSemiringRef;

    u32                nClasses_, nTimeFrames_;
    ConstSemiringRef   semiring_;
    Am::AcousticModel& acousticModel_;
    FloatT             logZero_;
    FloatT             logThreshold_;
    NnMatrix&          classes_;

    ClassProbsExtractor(u32                nCl,
                        u32                T,
                        ConstSemiringRef   semiring,
                        Am::AcousticModel& m,
                        FloatT             threshold,
                        NnMatrix&          cl)
            : nClasses_(nCl), nTimeFrames_(T), semiring_(semiring),  // should be LogSemiring, i.e. -log space
              acousticModel_(m),
              logZero_((FloatT)semiring_->zero()),  // should be Type<f32>::max (big positive num)
              logThreshold_(threshold),
              classes_(cl) {}

    ~ClassProbsExtractor() {
        // Always at destruction of this object,
        // we expect that the classes_ matrix is in computation mode.
        if (!classes_.isComputing())
            classes_.initComputation(false);  // no need to sync because we probably failed
    }

    void initZero() {
        classes_.resize(nClasses_, nTimeFrames_);
        classes_.finishComputation(false);

        for (FloatT& v : classes_)
            v = logZero_;
    }

    void add(TimeIndex timeIdx, Fsa::LabelId allophoneStateId, FloatT weight) {
        Am::AcousticModel::EmissionIndex state = acousticModel_.emissionIndex(allophoneStateId);

        // Check for absolut limits.
        if (weight > logZero_ || std::isinf(weight) || Math::isnan(weight))
            weight = logZero_;

        // Check for really big number. This acts as a kind of smoothing.
        if (weight > logThreshold_)
            weight = logThreshold_;

        if (classes_.at(state, timeIdx) >= logZero_)
            classes_.at(state, timeIdx) = weight;
        else
            classes_.at(state, timeIdx) = semiring_->collect(
                    Weight(classes_.at(state, timeIdx)), Weight(weight));
    }

    bool extractViaFsa(Fsa::ConstAutomatonRef f) {
        // See Search::AlignmentExtractor for comparison.
        struct ClassProbsExtractorFsa : Fsa::DfsState {
            ClassProbsExtractor& base_;
            TimeIndex            time_;
            TimeIndex            maxTime_;

            ClassProbsExtractorFsa(
                    Fsa::ConstAutomatonRef f,
                    ClassProbsExtractor&   b)
                    : Fsa::DfsState(Fsa::normalize(f)),
                      base_(b),
                      time_(0),
                      maxTime_(0) {}

            void exploreArc(Fsa::ConstStateRef /*from*/, const Fsa::Arc& a) {
                if (a.input() != Fsa::Epsilon) {
                    base_.add(time_, a.input(), (FloatT)a.weight());
                    maxTime_ = std::max(maxTime_, time_);
                    ++time_;
                }
            }
            virtual void exploreTreeArc(Fsa::ConstStateRef from, const Fsa::Arc& a) {
                exploreArc(from, a);
            }
            virtual void exploreNonTreeArc(Fsa::ConstStateRef from, const Fsa::Arc& a) {
                verify(color(a.target()) == Black);  // verify, that the fsa contains no loops
                exploreArc(from, a);
            }
            virtual void finishArc(Fsa::ConstStateRef from, const Fsa::Arc& a) {
                if (a.input() != Fsa::Epsilon)
                    --time_;
            }

            void extract() {
                time_ = 0;
                recursiveDfs();  // depth-first search
            }
        };

        ClassProbsExtractorFsa extractor(f, *this);
        extractor.extract();

        if (extractor.maxTime_ != nTimeFrames_ - 1)
            return false;
        return true;
    }

    bool extractViaTimeAlignedAutomaton(TimeAlignedAutomaton<FloatT>& fsa) {
        fsa.extractAlignmentMatrix(classes_, nClasses_, false);
        return true;
    }
};

template<typename FloatT>
u32 CtcCriterion<FloatT>::getCurSegmentTimeLen() const {
    return stateLogPosteriors_.nColumns();
}

template<typename FloatT>
u32 CtcCriterion<FloatT>::nEmissions() const {
    return acousticModel_->nEmissions();
}

template<typename FloatT>
void CtcCriterion<FloatT>::getStateScorers_MixtureSet(std::vector<Mm::FeatureScorer::Scorer>& scorers) {
    // Use fixed mixture set for the FeatureScorer.

    verify(fixedMixtureSetFeatureExtractionDataSource_);

    // We need to set the Corpus parameters on the data source.
    // Normally, the CorpusVisitor would do this. However, we want to
    // keep our code independent from any underlying CorpusVisitor
    // and we just know about the Bliss::SpeechSegment here.
    // First, clear previous parameters, then set the current ones.
    Speech::clearSegmentParametersOnDataSource(
            fixedMixtureSetFeatureExtractionDataSource_, Precursor::segment_);
    Speech::setSegmentParametersOnDataSource(
            fixedMixtureSetFeatureExtractionDataSource_, Precursor::segment_);

    if (!fixedMixtureSetExtractAlignmentsPortName_.empty()) {
        // See AligningFeatureExtractor::initializeAlignment() for reference.
        Flow::PortId alignmentPortId =
                fixedMixtureSetFeatureExtractionDataSource_->getOutput(
                        fixedMixtureSetExtractAlignmentsPortName_);
        Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>> alignmentRef;
        if (!fixedMixtureSetFeatureExtractionDataSource_->getData(alignmentPortId, alignmentRef)) {
            Core::Component::error("Failed to extract alignment from fixed mixture set via Flow.");
        }
        // We don't actually use it. If you want to dump it, you could
        // use a Flow cache node.
        // XXX: We could also dump it explicitely here.
    }

    // Now we are prepared to grab the features from the data source,
    // and we can create the FeatureScorers.
    // See AlignmentNode::work() and FeatureExtractor::processSegment() for reference.

    // reset feature scorer for usage with embedded flow files
    fixedMixtureSetFeatureScorer_->reset();
    bool                       firstFeature = true;
    Core::Ref<Speech::Feature> feature;
    scorers.reserve(getCurSegmentTimeLen());
    while (fixedMixtureSetFeatureExtractionDataSource_->getData(feature)) {
        // Check feature dependencies for first feature.
        if (firstFeature) {
            Mm::FeatureDescription description(*this, *feature);

            // see AcousticModel::isCompatible()
            Core::DependencySet dependencies;
            fixedMixtureSetFeatureScorer_->getDependencies(dependencies);

            Core::DependencySet featureDependencies;
            description.getDependencies(featureDependencies);

            if (!dependencies.satisfies(featureDependencies)) {
                Core::Component::warning("Feature mismatch between fixed-mixture-set and feature extraction.")
                        << "\n fixed-mixture-set feature deps:\n"
                        << featureDependencies
                        << "\n given features:\n"
                        << dependencies;
            }

            firstFeature = false;
        }
        scorers.push_back(fixedMixtureSetFeatureScorer_->getScorer(feature));
    }
    // finalize embedded network if applicable i.e. EOS
    if (firstFeature)
        fixedMixtureSetFeatureScorer_->finalize();

    require_eq(scorers.size(), getCurSegmentTimeLen());
}

template<typename FloatT>
FloatT CtcCriterion<FloatT>::getStateScore(u32 timeIdx, u32 emissionIdx) {
    // Return in -log space.
    // Note: Normally, we should also apply the ClassLabelWrapper on emissionIdx.
    //   This is not implemented here, so it only works if you don't have any "disregared" states.
    //   See e.g. TrainerFeatureScorer::getScore for an example usage of ClassLabelWrapper.

    FloatT prob = 0;
    if (statePosteriorScale_ != 0) {
        prob = stateLogPosteriors_.at(emissionIdx, timeIdx);  // in -log space
        if (prob > statePosteriorLogBackoff_)
            prob = statePosteriorLogBackoff_;
        prob *= statePosteriorScale_;
    }

    FloatT prior = 0;
    if (statePriors_->scale() != 0)
        // priors are in +log space
        prior = -statePriors_->at(emissionIdx) * statePriors_->scale();  // in -log space

    FloatT res = prob - prior;
    return res;
}

// simply wrap the state probability matrix to the Mm::FeatureScorer::Scorer API
template<typename FloatT>
void CtcCriterion<FloatT>::getStateScorers(std::vector<Mm::FeatureScorer::Scorer>& scorers) {
    if (fixedMixtureSetFeatureScorer_) {
        getStateScorers_MixtureSet(scorers);
        return;
    }

    // Our own FeatureScorer, based on the NN outputs and the priors.

    struct FeatureScorer : Mm::FeatureScorer {
        struct ContextScorer : Mm::FeatureScorer::ContextScorer {
            TimeIndex             timeFrameIdx_;
            CtcCriterion<FloatT>* parent_;
            ContextScorer(TimeIndex t, CtcCriterion<FloatT>* parent)
                    : timeFrameIdx_(t), parent_(parent) {}
            virtual Mm::EmissionIndex nEmissions() const {
                return parent_->nEmissions();
            }
            virtual Mm::Score score(Mm::EmissionIndex emissionIdx) const {
                // Return in -log space.
                return parent_->getStateScore(timeFrameIdx_, emissionIdx);
            }
        };
    };
    typedef typename FeatureScorer::ContextScorer Scorer;

    TimeIndex T = getCurSegmentTimeLen();
    scorers.resize(T);
    for (TimeIndex t = 0; t < T; ++t)
        scorers[t] = Mm::FeatureScorer::Scorer(new Scorer(t, this));
}

template<typename FloatT>
Core::Ref<const Fsa::Automaton> CtcCriterion<FloatT>::getHypothesesAllophoneStateFsa() {
    Bliss::SpeechSegment& segment = *Precursor::segment_;

    std::string orth = segment.orth();

    if (orth.empty())
        Core::Component::error("speech segment without transcription");

    // FSA through all possible allophone states.
    // (See CombinedExactSegmentwiseMeTrainer.)
    Fsa::ConstAutomatonRef hypothesesAllophoneStateFsa = Fsa::removeDisambiguationSymbols(Fsa::projectInput(allophoneStateGraphBuilder_->buildTransducer(orth)));
    require_eq(acousticModel_->allophoneStateAlphabet(), hypothesesAllophoneStateFsa->getInputAlphabet());

    return hypothesesAllophoneStateFsa;
}

template<typename FloatT>
Core::Ref<const Fsa::Automaton> CtcCriterion<FloatT>::getTimeAlignedFsa_SearchAligner() {
    TimeStats<10> timeStats(logTimeStatistics_, this->clog(), "getTimeAlignedFsa_SearchAligner");
    // FSA through all possible allophone states.
    timeStats.checkpoint("getHypothesesAllophoneStateFsa");
    Fsa::ConstAutomatonRef hypothesesAllophoneStateFsa = getHypothesesAllophoneStateFsa();

    // The current aligner code does not support eps arcs.
    timeStats.checkpoint("removeEpsilons");
    hypothesesAllophoneStateFsa = Fsa::removeEpsilons(hypothesesAllophoneStateFsa);

    // Remove very unprobable arcs (still only the allophone state FSA, no acoustic scores).
    timeStats.checkpoint("_removeInvalidArcs");
    hypothesesAllophoneStateFsa = Fsa::_removeInvalidArcs(hypothesesAllophoneStateFsa);

    // Will probably speed-up the aligner a bit.
    timeStats.checkpoint("staticCopy");
    hypothesesAllophoneStateFsa = Fsa::staticCopy(hypothesesAllophoneStateFsa);

    // We can now use Search::Aligner::getAlignmentFsa() to get a FSA
    // via a Baum-Welch aligner, and Fsa::posterior64() to get its posterior FSA.
    // See SegmentwiseAlignmentGenerator.

    timeStats.checkpoint("aligner-reset");
    aligner_->setModel(hypothesesAllophoneStateFsa, acousticModel_);
    aligner_->restart();

    // For every time frame, a Scorer.
    // These are the vectors p(a|x_t) / p(a), in -log space.
    timeStats.checkpoint("getStateScorers");
    std::vector<Mm::FeatureScorer::Scorer> scorers;
    getStateScorers(scorers);
    timeStats.checkpoint("aligner-feed");
    aligner_->feed(scorers);

    if (!aligner_->reachedFinalState()) {
        Core::Component::warning("aligner did not reached final state")
                << ", final score: " << aligner_->alignmentScore()
                << ", segment:" << Precursor::segment_->name();
        // ignore
        return Fsa::ConstAutomatonRef();
    }

    // We get the automaton where the arcs are the allophone states and
    // its weights are the transition scores via hypothesesAllophoneStateFsa
    // combined with the emission scores via stateProbsCpu.
    // The scores are in -log space.
    timeStats.checkpoint("aligner-getAlignmentFsa");
    Fsa::ConstAutomatonRef alignmentFsa = aligner_->getAlignmentFsa();
    return alignmentFsa;
}

template<typename FloatT>
Core::Ref<TimeAlignedAutomaton<FloatT>> CtcCriterion<FloatT>::getTimeAlignedFsa_custom() {
    // FSA through all possible allophone states.
    Fsa::ConstAutomatonRef hypothesesAllophoneStateFsa = getHypothesesAllophoneStateFsa();

    // TimeAlignedAutomaton does not support eps arcs.
    hypothesesAllophoneStateFsa = Fsa::removeEpsilons(hypothesesAllophoneStateFsa);

    // Remove very unprobable arcs (still only the allophone state FSA, no acoustic scores).
    hypothesesAllophoneStateFsa = Fsa::_removeInvalidArcs(hypothesesAllophoneStateFsa);

    // Will probably speed-up the aligner a bit.
    auto staticHypothesesAllophoneStateFsa = Fsa::staticCopy(hypothesesAllophoneStateFsa);

    auto timeAlignedFsaOrig = Core::ref(new TimeAlignedAutomaton<FloatT>(
            dynamic_cast<BatchStateScoreIntf<FloatT>*>(this), acousticModel_, staticHypothesesAllophoneStateFsa));
    timeAlignedFsaOrig->fullSearchAutoIncrease(minAcousticPruningThreshold_, maxAcousticPruningThreshold_);
    timeAlignedFsaOrig->dumpCount(Core::Component::log("time-aligned FSA: "));
    if (timeAlignedFsaOrig->initialStateId() == Fsa::InvalidStateId)
        timeAlignedFsaOrig.reset();
    return timeAlignedFsaOrig;
}

template<typename FloatT>
Core::Ref<const Fsa::Automaton> CtcCriterion<FloatT>::getTimeAlignedFsa() {
    if (useSearchAligner_)
        return getTimeAlignedFsa_SearchAligner();

    auto timeAlignedFsaOrig = getTimeAlignedFsa_custom();
    if (!timeAlignedFsaOrig)
        return Fsa::ConstAutomatonRef();

    // The states returend by TimeAlignedAutomaton will be invalid ones the automaton is freed.
    // So we keep it alive until the end of this function.
    Fsa::ConstAutomatonRef timeAlignedFsa = timeAlignedFsaOrig;

    // We will calculate the posterior FSA based on this, and
    // its algorithm creates a state potential vector based on the state ids.
    timeAlignedFsa = Fsa::normalize(timeAlignedFsa);

    // staticCopy might speed it up a bit again.
    // Also important because the original automaton will go out of scope.
    timeAlignedFsa = Fsa::staticCopy(timeAlignedFsa);
    return timeAlignedFsa;
}

template<typename FloatT>
Core::Ref<const Fsa::Automaton> CtcCriterion<FloatT>::getPosteriorFsa() {
    TimeStats<5> timeStats(logTimeStatistics_, this->clog(), "getPosteriorFsa");

    // We get the automaton where the arcs are the allophone states and
    // its weights are the transition scores via hypothesesAllophoneStateFsa
    // combined with the emission scores via stateProbsCpu.
    // The scores are in -log space.
    timeStats.checkpoint("getTimeAlignedFsa");
    Fsa::ConstAutomatonRef alignmentFsa = getTimeAlignedFsa();
    if (!alignmentFsa)
        return Fsa::ConstAutomatonRef();
    if (alignmentFsa->initialStateId() == Fsa::InvalidStateId)
        return Fsa::ConstAutomatonRef();

    // The posterior automaton represents the accumulated scores
    // calculated via a forward-backward algorithm through the automaton.
    // The posterior automaton has the state prob errors on its arcs in -log space.
    // (Also see Aligner::getAlignmentPosteriorFsa() and MmiSegmentwiseNnTrainer<T>::getNumeratorPosterior() as reference.)
    timeStats.checkpoint("getAlignmentPosteriorFsa");
    Fsa::ConstAutomatonRef alignmentPosteriorFsa;
    if (useSearchAligner_ && posteriorUseSearchAligner_)
        alignmentPosteriorFsa = aligner_->getAlignmentPosteriorFsa(alignmentFsa).first;
    else {
        // Note: This requires that it uses the LogSemiring.
        // If the alignment-fsa is via the search-aligner, this is not the case (see Aligner::SearchSpace::getAlignmentFsaViterbi).
        // Aligner::getAlignmentPosteriorFsa will do the correct thing then.
        Fsa::Weight _alignmentPosteriorFsaTotal = alignmentFsa->semiring()->one();
        alignmentPosteriorFsa                   = Fsa::posterior64(alignmentFsa, _alignmentPosteriorFsaTotal, posteriorTotalNormalize_);
    }

    return alignmentPosteriorFsa;
}

template<typename FloatT>
void CtcCriterion<FloatT>::dumpViterbiAlignments() {
    // extract viterbi alignment...

    // We must have set the allophone state automaton (= model) before.
    verify(aligner_->getModel());

    // Switch aligner to Viterbi mode.
    Search::Aligner::Mode oldMode = aligner_->getMode();
    aligner_->selectMode(Search::Aligner::modeViterbi);

    std::vector<Mm::FeatureScorer::Scorer> scorers;
    getStateScorers(scorers);

    aligner_->restart();
    aligner_->feed(scorers);

    if (!aligner_->reachedFinalState()) {
        Core::Component::warning("Viterbi aligner did not reached final state");
    }
    else {
        // See AlignmentNode::work().
        Flow::ArchiveWriter<Speech::Alignment> writer(dumpViterbiAlignmentsArchive_.get());
        aligner_->getAlignment(writer.data_->data());

        writer.write(Precursor::segment_->fullName());
    }

    aligner_->selectMode(oldMode);
}

static int debug_iter = 0;

template<typename FloatT>
Core::Ref<Am::AcousticModel> CtcCriterion<FloatT>::getAcousticModel() {
    return acousticModel_;
}

template<typename FloatT>
bool CtcCriterion<FloatT>::getAlignment(Speech::Alignment& out,
                                        NnMatrix&          logPosteriors,
                                        const std::string& orthography,
                                        FloatT             minProbGT,
                                        FloatT             gamma) {
    stateLogPosteriors_.resize(logPosteriors.nRows(), logPosteriors.nColumns());
    stateLogPosteriors_.initComputation(false);
    stateLogPosteriors_.copy(logPosteriors);
    stateLogPosteriors_.scale(-1);  // -log space
    stateLogPosteriors_.finishComputation(true);

    Bliss::Corpus        dummyCorpus;
    Bliss::Recording     dummyRecording(&dummyCorpus);
    Bliss::SpeechSegment speechSegment(&dummyRecording);  // must be in scope until end when used
    speechSegment.setOrth(orthography);
    Precursor::segment_ = &speechSegment;

    u32       nClasses = stateLogPosteriors_.nRows();
    TimeIndex T        = stateLogPosteriors_.nColumns();
    require_gt(T, 0);
    require_eq(acousticModel_->nEmissions(), nClasses);
    require_eq(statePriors_->size(), nClasses);

    if (!useSearchAligner_ && useDirectAlignmentExtraction_) {
        Core::Ref<TimeAlignedAutomaton<FloatT>> timeAlignedFsa = getTimeAlignedFsa_custom();
        if (!timeAlignedFsa)
            return false;
        timeAlignedFsa->extractAlignment(out, minProbGT, gamma);
    }
    else {
        Fsa::ConstAutomatonRef alignmentPosteriorFsa = getPosteriorFsa();
        Precursor::segment_                          = NULL;
        if (!alignmentPosteriorFsa)
            return false;

        Search::extractAlignment(out, alignmentPosteriorFsa, minProbGT, gamma);
        if (out.empty())
            return false;
    }
    out.setAlphabet(acousticModel_->allophoneStateAlphabet());
    return true;
}

template<typename FloatT>
bool CtcCriterion<FloatT>::calcStateProbErrors(FloatT&   error,  // the error (objective function value) + the reference prob
                                               NnMatrix& referenceProb) {
    typedef typename CalcTypes<FloatT>::Semiring Semiring;
    typedef typename CalcTypes<FloatT>::ConstSemiringRef ConstSemiringRef;
    TimeStats<20> timeStats(logTimeStatistics_, this->clog(), "calcStateProbErrors");

    u32       nClasses = stateLogPosteriors_.nRows();
    TimeIndex T        = stateLogPosteriors_.nColumns();
    require_eq(acousticModel_->nEmissions(), nClasses);
    require_eq(statePriors_->size(), nClasses);

    ConstSemiringRef logSemiring = CalcTypes<FloatT>::logSemiring();
    ClassProbsExtractor<FloatT> classProbsExtractor(nClasses,
                                                    T,
                                                    logSemiring,
                                                    *acousticModel_,
                                                    posteriorArcLogThreshold_,
                                                    referenceProb);
    timeStats.checkpoint("referenceProb-initZero");
    classProbsExtractor.initZero();

    if (!useSearchAligner_ && useDirectAlignmentExtraction_) {
        timeStats.checkpoint("getTimeAlignedFsa_custom");
        Core::Ref<TimeAlignedAutomaton<FloatT>> timeAlignedFsa = getTimeAlignedFsa_custom();
        if (!timeAlignedFsa) {
            Core::Component::warning("No alignment found.");
            return false;
        }
        timeStats.checkpoint("classProbsExtractor-extractViaTimeAlignedAutomaton");
        if (!classProbsExtractor.extractViaTimeAlignedAutomaton(*timeAlignedFsa)) {
            Core::Component::warning("Could not extract via alignment.");
            return false;
        }
    }
    else {
        timeStats.checkpoint("getPosteriorFsa");
        Fsa::ConstAutomatonRef alignmentPosteriorFsa = getPosteriorFsa();
        if (!alignmentPosteriorFsa)
            return false;

        // Extract
        // P'_{t,a} := \sum_{\overline{a},a_t = a}
        //    \prod_\tau p(a_\tau|a_{\tau-1}, \overline{w}) \cdot \frac{p(a_\tau|x_\tau)}{p(a_\tau)} .
        // alignmentPosteriorFsa values are in -log space, so we use its log-semiring to collect the values.
        timeStats.checkpoint("referenceProb-extract");
        if (!classProbsExtractor.extractViaFsa(alignmentPosteriorFsa)) {
            Core::Component::warning("Did not get probs for all time frames.");
            return false;
        }
    }
    // We now have P'_{t,a} in referenceProb in -log-space.

    timeStats.checkpoint("dumpViterbiAlignments");
    if (dumpViterbiAlignmentsArchive_)
        dumpViterbiAlignments();

    if (doDebugDumps_)
        referenceProb.printToFile("data/dump-matrix-p-" + std::to_string(debug_iter));

    if (!useCrossEntropyAsLoss_) {
        timeStats.checkpoint("calc-loss");
        // P = \sum_a P'_{1,a}.
        // Calculated and result in -log space.
        static const int t         = 0;
        auto*            collector = logSemiring->getCollector(logSemiring->zero());
        for (u32 a = 0; a < referenceProb.nRows(); ++a) {
            FloatT prob = referenceProb.at(a, t);
            if (prob < (FloatT)logSemiring->zero() && !std::isinf(prob))
                collector->feed(typename Semiring::Weight(prob));
        }
        error = collector->get();  // L = -ln P.
        delete collector;

        if (std::isinf(error) || error > (FloatT)logSemiring->zero())
            error = logSemiring->zero();
    }

    if (statePriors_->learningRate() > 0) {
        timeStats.checkpoint("calc-state-priors-update");
        // XXX: Could be calculated on the GPU.

        // P'' = \sum_{t} P'_{t,a}, in -log-space
        NnVector p(nClasses);
        p.setToZero();
        for (u32 a = 0; a < nClasses; ++a) {
            auto* collector = logSemiring->getCollector(logSemiring->zero());
            for (u32 t = 0; t < T; ++t)
                collector->feed(typename Semiring::Weight(referenceProb.at(a, t)));  // in -log-space
            p.at(a) = collector->get();
            delete collector;
        }

        // Transfer to GPU, and transfer into std space.
        p.initComputation(true);
        p.scale(-1);  // transfer to +log space.
        p.exp();      // transfer to std space.

        FloatT errFactor = -1.0 / std::exp(-error);  // -1/P, in std space.

        statePriors_->trainSoftmax(p, errFactor);

        if (doDebugDumps_) {
            statePriors_->write("data/dump-prior-params-" + std::to_string(debug_iter));
        }
    }

    if (posteriorNBestLimit_ > 0 && posteriorNBestLimit_ < (u32)Core::Type<Core::ParameterInt::Value>::max) {
        timeStats.checkpoint("posteriorNBestLimit");
        // Simple, straight-forward, not-optimized, CPU-based implementation.
        // Note that we are in -log-space, thus the best is the lowest number (0).
        for (u32 t = 0; t < T; ++t) {
            // Find N best elements.
            std::set<FloatT> nBestNums;
            for (u32 a = 0; a < nClasses; ++a) {
                FloatT prob = referenceProb.at(a, t);

                if (nBestNums.size() < posteriorNBestLimit_) {
                    nBestNums.insert(prob);
                    continue;
                }

                // Lower than the biggest stored element.
                auto biggestNBestPtr = --nBestNums.end();
                if (prob < *biggestNBestPtr) {
                    // Remove and insert new prob.
                    nBestNums.erase(biggestNBestPtr);
                    nBestNums.insert(prob);
                }
            }

            // Reset all reference probabilities behind the limit.
            verify(!nBestNums.empty());
            FloatT limit = *nBestNums.rbegin();
            for (u32 a = 0; a < nClasses; ++a) {
                FloatT& prob = referenceProb.at(a, t);
                if (prob > limit)
                    prob = Core::Type<FloatT>::max;
            }
        }
    }

    // Copy over to GPU memory.
    timeStats.checkpoint("referenceProb-sync-to-gpu");
    referenceProb.initComputation(true);

    // Transfer to +log-space, and apply posterior scale.
    timeStats.checkpoint("referenceProb-scale");
    referenceProb.scale(-1 * posteriorScale_);

    // We want to transfer std space (exp) and to mean-normalize every column.
    // The element-wise softmax exactly does this.
    timeStats.checkpoint("referenceProb-softmax");
    referenceProb.softmax();

    if (useCrossEntropyAsLoss_) {
        timeStats.checkpoint("calc-loss");
        // L = - \sum_{t,a}  P'_{t,a} \cdot \log y_{t,a}.
        // stateLogPosteriors_ is in -log space, thus -\log y = stateLogPosteriors_.
        stateLogPosteriors_.initComputation(false /* already up-to-date */);
        error = referenceProb.dot(stateLogPosteriors_);
    }

    if (dumpReferenceProbsArchive_) {
        timeStats.checkpoint("dumpReferenceProbs");
        referenceProb.finishComputation(true);

        Flow::ArchiveWriter<Math::Matrix<FloatT>> writer(dumpReferenceProbsArchive_.get());

        referenceProb.convert(writer.data_->data());
        writer.write(Precursor::segment_->fullName());

        referenceProb.initComputation(false);
    }

    Core::Component::log() << "P = " << std::exp(-error)
                           << ", loss L = " << error
                           << ", frames = " << T
                           << ", normalized loss = " << (error / T);
    if (doDebugDumps_)
        Core::Component::log() << "iter: " << debug_iter;

    // Not exactly sure where this can be introduced. (Maybe the softmax?)
    // However, if it did happen, discard this segment - it would destroy our model in training.
    if (std::isinf(error) || Math::isnan(error)) {
        Core::Component::warning("Error-value is invalid.");
        return false;
    }

    // Maybe, in one time frame, there was no active emission state.
    // This would result in the softmax returning nans.
    timeStats.checkpoint("referenceProb-l1norm");
    FloatT refProbNorm = referenceProb.l1norm();
    if (std::isinf(refProbNorm) || Math::isnan(error)) {
        Core::Component::warning("Reference prob norm is invalid.");
        return false;
    }

    return true;
}

template<typename FloatT>
void CtcCriterion<FloatT>::initLexicon() {
    lexicon_ = Bliss::Lexicon::create(Core::Component::select("lexicon"));
    if (!lexicon_)
        Core::Component::criticalError("failed to initialize the lexicon");
}

template<typename FloatT>
void CtcCriterion<FloatT>::initAcousticModel() {
    // The acoustic model is only to define the state model, i.e. to
    // create the allophone state graph builder.
    // Thus, it does not need the state probabilities.
    // We calculate the state probabilities ourself (see FeatureScorer in processBuffer).
    acousticModel_ = Am::Module::instance().createAcousticModel(Core::Component::select("acoustic-model"),
                                                                lexicon_,
                                                                Am::AcousticModel::noEmissions);
    if (!acousticModel_)
        Core::Component::criticalError("failed to initialize the acoustic model");
}

template<typename FloatT>
void CtcCriterion<FloatT>::initAllophoneStateGraphBuilder() {
    // This gets the acoustic model, but it actually only uses the lexion + createTransducerBuilder,
    // which itself uses the lexion + HMM topology and related things.
    // It does not use the mixtureSet/featureScorer.
    // This is needed to build up DFA through all possible allophone states.
    allophoneStateGraphBuilder_ = std::make_shared<Speech::AllophoneStateGraphBuilder>(Core::Component::select("allophone-state-graph-builder"),
                                                                                       lexicon_,
                                                                                       acousticModel_);
    // AllophoneStateGraphBuilder will load all transducers lazily when it first needs them.
    // For better timing statistics, just load them now.
    // To do that, just build a orthography now.
    std::string dummyOrth;
    auto        lemmasRange = lexicon_->lemmas();
    for (auto lemma_iter = lemmasRange.first; lemma_iter != lemmasRange.second; ++lemma_iter) {
        const Bliss::Lemma* lemma(*lemma_iter);
        if (lemma->nPronunciations() == 0)
            continue;
        dummyOrth = lemma->preferredOrthographicForm().str();
        if (dummyOrth.empty())
            this->warning("Empty orthography for lemma '%s'.", lemma->name().str());
        break;
    }
    if (!dummyOrth.empty())
        allophoneStateGraphBuilder_->buildTransducer(dummyOrth + " ");
    else
        this->warning("Did not found any pronunciation in lexicon.");
}

template<typename FloatT>
void CtcCriterion<FloatT>::initSearchAligner() {
    // We might not need to create it if !useSearchAligner_.
    // However, we also might use it in some debug code.
    // So for now, always create.
    aligner_ = std::make_shared<Search::Aligner>(Core::Component::select("ctc-aligner"));
    if (useSearchAligner_ && aligner_->getMode() != Search::Aligner::modeBaumWelch)
        Core::Component::log("CTC aligner is not in Baum-Welch mode but in Viterbi mode");
    if (useSearchAligner_ && useDirectAlignmentExtraction_)
        Core::Component::error("CTC: use-search-aligner=true and use-direct-alignment-extraction=true don't work together");
}

template<typename FloatT>
void CtcCriterion<FloatT>::initStatePriors() {
    statePriors_ = std::make_shared<Prior<FloatT>>(Core::Component::select("priors"));
    if (statePriors_->fileName().empty())
        statePriors_->initUniform(acousticModel_->nEmissions());
    else {
        // XXX: It is a bit unfortunate that the prior filename
        // is used for both loading and saving.
        // We have not implemented yet to save the CTC prior, but
        // we should have two separate config options for the load filename
        // and save filename.
        Core::Component::log("state priors: ") << statePriors_->fileName();
        if (!statePriors_->read()) {
            // A warning, until we have figured out a solution.
            Core::Component::warning("could not read priors, init with uniform");
            statePriors_->initUniform(acousticModel_->nEmissions());
        }
        else {
            require_eq(acousticModel_->nEmissions(), statePriors_->size());
        }
    }
}

template<typename FloatT>
void CtcCriterion<FloatT>::initFixedMixtureSet() {
    if (statePosteriorScale_ != 1)
        Core::Component::warning("The state-posterior-scale %f will be ignored with fixed-mixture-set.",
                                 statePosteriorScale_);
    // As well as the state priors and any scaling in there, but no check for that here
    // as it only complicates things.

    std::string mixtureSetSelector        = paramFixedMixtureSetSelector(Precursor::config);
    std::string featureExtractionSelector = paramFixedMixtureSetFeatureExtractionSelector(Precursor::config);
    Core::Component::log("CTC: using fixed mixture set, selector '%s', feature extraction selector '%s'",
                         mixtureSetSelector.c_str(), featureExtractionSelector.c_str());

    Core::Ref<Mm::AbstractMixtureSet> mixtureSet = Mm::Module::instance().readAbstractMixtureSet(Core::Component::select(mixtureSetSelector));
    if (!mixtureSet)
        Core::Component::criticalError("failed to initialize the mixture set");

    fixedMixtureSetFeatureScorer_ = Mm::Module::instance().createScaledFeatureScorer(Core::Component::select(mixtureSetSelector), mixtureSet);
    if (!fixedMixtureSetFeatureScorer_)
        Core::Component::criticalError("failed to initialize the mixture set feature scorer");

    fixedMixtureSetFeatureExtractionDataSource_ = Core::Ref<Speech::DataSource>(Speech::Module::instance().createDataSource(Core::Component::select(featureExtractionSelector), true));
    if (!fixedMixtureSetFeatureExtractionDataSource_)
        Core::Component::criticalError("failed to initialize the mixture set feature extraction");
    // The main data source will drive the progress indicator -
    // this is another separate data source which should not infer.
    fixedMixtureSetFeatureExtractionDataSource_->setProgressIndication(false);

    fixedMixtureSetExtractAlignmentsPortName_ = paramFixedMixtureSetExtractAlignmentsPortName(Precursor::config);
}

template<typename FloatT>
void CtcCriterion<FloatT>::initDebug() {
    std::string dumpViterbiFilename = paramDumpViterbiAlignments(Precursor::config);
    if (!dumpViterbiFilename.empty())
        dumpViterbiAlignmentsArchive_ = std::shared_ptr<Core::Archive>(Core::Archive::create(Core::Component::select(paramDumpViterbiAlignments.name()),
                                                                                             dumpViterbiFilename,
                                                                                             Core::Archive::AccessModeWrite));

    std::string dumpReferenceProbs = paramDumpReferenceProbs(Precursor::config);
    if (!dumpReferenceProbs.empty())
        dumpReferenceProbsArchive_ = std::shared_ptr<Core::Archive>(Core::Archive::create(Core::Component::select(paramDumpReferenceProbs.name()),
                                                                                          dumpReferenceProbs,
                                                                                          Core::Archive::AccessModeWrite));
}

template<typename FloatT>
CtcCriterion<FloatT>::CtcCriterion(const Core::Configuration& config)
        : Precursor(config),
          useSearchAligner_(paramUseSearchAligner(config)),
          minAcousticPruningThreshold_(paramMinAcousticPruningThreshold(Core::Component::select("ctc-aligner"))),
          maxAcousticPruningThreshold_(paramMaxAcousticPruningThreshold(Core::Component::select("ctc-aligner"))),
          useDirectAlignmentExtraction_(paramUseDirectAlignmentExtraction(config)),
          statePosteriorScale_(paramStatePosteriorScale(config)),
          statePosteriorLogBackoff_(paramStatePosteriorLogBackoff(config)),
          posteriorUseSearchAligner_(paramPosteriorUseSearchAligner(config)),
          posteriorTotalNormalize_(paramPosteriorTotalNormalize(config)),
          posteriorArcLogThreshold_(paramPosteriorArcLogThreshold(config)),
          posteriorScale_(paramPosteriorScale(config)),
          posteriorNBestLimit_(paramPosteriorNBestLimit(config)),
          doDebugDumps_(paramDebugDumps(config)),
          logTimeStatistics_(paramLogTimeStatistics(config)),
          useCrossEntropyAsLoss_(paramUseCrossEntropyAsLoss(config)),
          inputInLogSpace_(paramInputInLogSpace(config)) {
    TimeStats<10> timeStats(logTimeStatistics_, this->clog(), "initialization");
    timeStats.checkpoint("initLexicon");
    initLexicon();
    timeStats.checkpoint("initAcousticModel");
    initAcousticModel();
    timeStats.checkpoint("initAllophoneStateGraphBuilder");
    initAllophoneStateGraphBuilder();
    timeStats.checkpoint("initSearchAligner");
    initSearchAligner();
    timeStats.checkpoint("initStatePriors");
    initStatePriors();
    timeStats.checkpoint("initFixedMixtureSet");
    if ((bool)paramUseFixedMixtureSet(config))
        initFixedMixtureSet();
    timeStats.checkpoint("initDebug");
    initDebug();
}

template<typename FloatT>
CtcCriterion<FloatT>::~CtcCriterion() {
}

template<typename FloatT>
void CtcCriterion<FloatT>::inputSpeechSegment(Bliss::SpeechSegment& segment, NnMatrix& nnOutput, NnVector* weights) {
    TimeStats<10> timeStats(logTimeStatistics_, this->clog(), "inputSpeechSegment");

    discardCurrentInput_ = true;
    verify(nnOutput.isComputing());

    if (weights)
        Core::Component::error("CtcCriterion::inputSpeechSegment not yet implemented with weights");

    typedef typename Math::FastMatrix<FloatT> FastMatrix;
    Precursor::segment_ = &segment;
    require_eq(acousticModel_->nEmissions(), nnOutput.nRows());
    Precursor::input(nnOutput, weights);

    // The nnOutput contains the state posterior probabilities,
    // i.e. p(a|x_t), where a is an allophone state,
    // for all time-frames (columns).

    ++debug_iter;
    if (doDebugDumps_) {
        nnOutput.finishComputation(true);
        nnOutput.printToFile("data/dump-nn-output-" + std::to_string(debug_iter));
        nnOutput.initComputation(false);
    }

    // Copy over. nnOutput is in std space or +log space (depending on inputInLogSpace_ option), and in GPU mode.
    timeStats.checkpoint("stateLogPosteriors-copy");
    stateLogPosteriors_.resize(nnOutput.nRows(), nnOutput.nColumns());
    stateLogPosteriors_.initComputation(false);
    stateLogPosteriors_.copy(nnOutput);

    // Note: We could also let the output layer not apply softmax and
    // thus avoid the log(exp(x)) operation.
    // We even can substract the bias directly from it.
    // See BatchFeatureScorer which does that.
    timeStats.checkpoint("stateLogPosteriors-log");
    if (!inputInLogSpace_)
        stateLogPosteriors_.log();  // +log space
    timeStats.checkpoint("stateLogPosteriors-scale");
    stateLogPosteriors_.scale(-1);  // -log space

    // All of the CTC criterion calculation is currently done on the CPU.
    // Because of the alignment-code, the automata stuff, etc., I think
    // it's not that easy to implement it for the GPU.

    // Get into CPU mode, but both GPU memory and CPU memory are up-to-date,
    // and we will not modify it further.
    timeStats.checkpoint("stateLogPosteriors-sync");
    stateLogPosteriors_.finishComputation(true);

    // Simple check. Can happen if we have destroyed the matrix weights earlier.
    // Can also happen if you have the wrong input-in-log-space option.
    require(!Math::isnan(stateLogPosteriors_.at(0, 0)));

    // For the given segment transcription, it builds the class probabilities per frame.
    timeStats.checkpoint("calcStateProbErrors");
    discardCurrentInput_                     = !calcStateProbErrors(Precursor::objectiveFunction_,
                                                Precursor::errorSignal_  // we keep here the reference prob, \hat{y}
    );
    Precursor::needRecalc_objectiveFunction_ = false;
    Precursor::needRecalc_errorSignal_       = false;
}

template<typename FloatT>
void CtcCriterion<FloatT>::getObjectiveFunction(FloatT& value) {
    if (!discardCurrentInput_)
        value = Precursor::objectiveFunction_;
    else
        // fallback if this is called anyway
        value = Core::Type<FloatT>::max;
}

template<typename FloatT>
void CtcCriterion<FloatT>::getErrorSignal(NnMatrix& errorSignal) {
    if (!discardCurrentInput_) {
        if (inputInLogSpace_) {  // y - \hat{y}
            // If we got posteriors in log-space, we interpret it here like we applied softmax on it
            // (which we did not - we just took them as they are because we need the log-posteriors anyway)
            // thus the error signal is like natural pairing with softmax.
            errorSignal.copy(*Precursor::nnOutput_);
            errorSignal.softmax();
            errorSignal.add(Precursor::errorSignal_, (FloatT)-1);  // this is the reference prob, \hat{y}
        }
        else {                                          // -\hat{y} / y
            errorSignal.copy(Precursor::errorSignal_);  // this is the reference prob, \hat{y}
            errorSignal.elementwiseDivision(*Precursor::nnOutput_);
            errorSignal.scale(-1);
        }
    }
    else
        // fallback if this is called anyway
        errorSignal.setToZero();
}

template<typename FloatT>
void CtcCriterion<FloatT>::getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer) {
    if (!discardCurrentInput_ && !inputInLogSpace_) {
        switch (lastLayer.getLayerType()) {
            case NeuralNetworkLayer<FloatT>::linearAndSoftmaxLayer:
            case NeuralNetworkLayer<FloatT>::softmaxLayer:
                // y - \hat{y}
                errorSignal.copy(*Precursor::nnOutput_);
                errorSignal.add(Precursor::errorSignal_, (FloatT)-1);  // this is the reference prob, \hat{y}
                return;

            default: {
                static bool warningOnce = false;
                if (!warningOnce) {
                    warningOnce = true;
                    Core::Component::warning() << "using CtcCriterion natural pairing with unsupported last NN layer; "
                                               << "using default implementation instead";
                }
            }
        }
    }

    // fallback
    Precursor::getErrorSignal_naturalPairing(errorSignal, lastLayer);
}

template<typename FloatT>
typename CtcCriterion<FloatT>::NnMatrix* CtcCriterion<FloatT>::getPseudoTargets() {
    return &this->errorSignal_;
}

// explicit template instantiation
template class CtcCriterion<f32>;
template class CtcCriterion<f64>;

}  // namespace Nn
