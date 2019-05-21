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
#include "LatticeRescorerNodes.hh"
#include <Bliss/Orthography.hh>
#include <Flf/FlfCore/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Dfs.hh>
#include <Fsa/Project.hh>
#include <Lattice/Merge.hh>
#include "AdvancedAccuracyFsaBuilder.hh"
#include "LatticeRescorerAutomaton.hh"

using namespace Speech;

/** NumeratorFromDenominatorNode
 */
const Core::ParameterString NumeratorFromDenominatorNode::paramSegmentOrth(
        "orthography",
        "segment orthography to determine correct hypotheses");

NumeratorFromDenominatorNode::NumeratorFromDenominatorNode(
        const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          segmentOrth_(paramSegmentOrth(config)),
          orthToLemma_(0) {}

NumeratorFromDenominatorNode::~NumeratorFromDenominatorNode() {
    delete orthToLemma_;
}

bool NumeratorFromDenominatorNode::setParameter(const std::string& name, const std::string& value) {
    if (paramSegmentOrth.match(name)) {
        segmentOrth_ = paramSegmentOrth(value);
        return true;
    }
    return Precursor::setParameter(name, value);
}

bool NumeratorFromDenominatorNode::work(Flow::PortId p) {
    if (!Precursor::work(p)) {
        return false;
    }

    Flow::DataPtr<Flow::DataAdaptor<Flf::ConstLatticeRef>> in;
    if (!getData(1, in)) {
        error("could not read port lattice");
        return putData(0, in.get());
    }

    Flf::ConstLatticeRef                     l           = in->data();
    Lattice::ConstWordLatticeRef             denominator = toWordLattice(l);
    Lattice::ConstWordLatticeRef             numerator   = Lattice::extractNumerator(segmentOrth_,
                                                                       denominator,
                                                                       orthToLemma_,
                                                                       lemmaPronToLemma_,
                                                                       lemmaToLemmaConfusion_);
    Flow::DataAdaptor<Flf::ConstLatticeRef>* out         = new Flow::DataAdaptor<Flf::ConstLatticeRef>();
    out->data()                                          = fromWordLattice(numerator);
    require(!getData(1, in));
    return putData(0, out) && putData(0, in.get());
}

void NumeratorFromDenominatorNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);

    Bliss::LexiconRef lexicon = modelCombination->lexicon();
    verify(!orthToLemma_);
    orthToLemma_      = new Bliss::OrthographicParser(select("orthographic-parser"), lexicon);
    lemmaPronToLemma_ = lexicon->createLemmaPronunciationToLemmaTransducer();

    Fsa::ConstAutomatonRef lemmaToEval = lexicon->createLemmaToEvaluationTokenTransducer();
    lemmaToLemmaConfusion_             = Fsa::composeMatching(lemmaToEval, Fsa::invert(lemmaToEval));
    lemmaToLemmaConfusion_             = Fsa::cache(lemmaToLemmaConfusion_);

    needInit_ = false;
}

/** LatticeRescorerNode
 */
LatticeRescorerNode::LatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {
    addInputs(1);
    addOutputs(1);
}

bool LatticeRescorerNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes);
    getInputAttributes(1, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<Flf::ConstLatticeRef>::type())) {
        return false;
    }
    return Precursor::configure() && putOutputAttributes(0, attributes);
}

/** DistanceLatticeRescorerNode
 */
DistanceLatticeRescorerNode::DistanceLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

/** ApproximateDistanceLatticeRescorerNode
 */
ApproximateDistanceLatticeRescorerNode::ApproximateDistanceLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {
    addInputs(2);
}

bool ApproximateDistanceLatticeRescorerNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());

    getInputAttributes(2, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<AlignmentGeneratorRef>::type())) {
        return false;
    }

    getInputAttributes(3, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<Flf::ConstLatticeRef>::type())) {
        return false;
    }

    return Precursor::configure();
}

bool ApproximateDistanceLatticeRescorerNode::work(Flow::PortId p) {
    if (!Precursor::work(p)) {
        return false;
    }

    Flow::DataPtr<Flow::DataAdaptor<Flf::ConstLatticeRef>> inHyp;
    if (!getData(1, inHyp)) {
        error("could not read port hypotheses");
        return putData(0, inHyp.get());
    }

    Flow::DataPtr<Flow::DataAdaptor<AlignmentGeneratorRef>> inAli;
    if (!getData(2, inAli)) {
        error("could not read port alignments");
    }
    alignmentGenerator_ = inAli->data();

    Flow::DataPtr<Flow::DataAdaptor<Flf::ConstLatticeRef>> inRef;
    if (!getData(3, inRef)) {
        error("could not read port references");
        return putData(0, inRef.get());
    }

    Lattice::ConstWordLatticeRef ref  = toWordLattice(inRef->data());
    Lattice::ConstWordLatticeRef hyp  = toWordLattice(inHyp->data());
    Fsa::ConstAutomatonRef       dist = getDistanceFsa(ref, hyp);
    Flf::ConstLatticeRef         l    = inHyp->data();
    l                                 = Flf::fromFsa(dist, Flf::ConstSemiringRef(new Flf::TropicalSemiring(1)), 0);
    l->setBoundaries(inHyp->data()->getBoundaries());
    Flow::DataAdaptor<Flf::ConstLatticeRef>* out = new Flow::DataAdaptor<Flf::ConstLatticeRef>();
    out->data()                                  = l;
    putData(0, out);

    require(!getData(1, inHyp));
    require(!getData(2, inAli));
    require(!getData(3, inRef));
    return putData(0, inHyp.get());
}

/** ApproximatePhoneAccuracyLatticeRescorerNode
 */
ApproximatePhoneAccuracyLatticeRescorerNode::ApproximatePhoneAccuracyLatticeRescorerNode(
        const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          builder_(0) {}

ApproximatePhoneAccuracyLatticeRescorerNode::~ApproximatePhoneAccuracyLatticeRescorerNode() {
    delete builder_;
}

void ApproximatePhoneAccuracyLatticeRescorerNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);
    verify(!builder_);
    builder_ = new ApproximatePhoneAccuracyLatticeBuilder(
            select("approximate-phone-accuracy-lattice-builder"), modelCombination->lexicon());
    needInit_ = false;
}

Fsa::ConstAutomatonRef ApproximatePhoneAccuracyLatticeRescorerNode::getDistanceFsa(
        Lattice::ConstWordLatticeRef ref, Lattice::ConstWordLatticeRef hyp) {
    verify(builder_);
    return builder_->createFunctor(segmentId(), ref, hyp, alignmentGenerator_).build();
}

/** FramePhoneAccuracyLatticeRescorerNode
 */
FramePhoneAccuracyLatticeRescorerNode::FramePhoneAccuracyLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          builder_(0) {}

FramePhoneAccuracyLatticeRescorerNode::~FramePhoneAccuracyLatticeRescorerNode() {
    delete builder_;
}

void FramePhoneAccuracyLatticeRescorerNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);
    verify(!builder_);
    builder_  = new FramePhoneAccuracyLatticeBuilder(select("frame-phone-accuracy-lattice-builder"), modelCombination->lexicon());
    needInit_ = false;
}

Fsa::ConstAutomatonRef FramePhoneAccuracyLatticeRescorerNode::getDistanceFsa(
        Lattice::ConstWordLatticeRef ref, Lattice::ConstWordLatticeRef hyp) {
    verify(builder_);
    return builder_->createFunctor(segmentId(), ref, hyp, alignmentGenerator_).build();
}

/** SoftFramePhoneAccuracyLatticeRescorerNode
 */
SoftFramePhoneAccuracyLatticeRescorerNode::SoftFramePhoneAccuracyLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          builder_(0) {}

SoftFramePhoneAccuracyLatticeRescorerNode::~SoftFramePhoneAccuracyLatticeRescorerNode() {
    delete builder_;
}

void SoftFramePhoneAccuracyLatticeRescorerNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);
    verify(!builder_);
    builder_  = new SoftFramePhoneAccuracyLatticeBuilder(select("soft-frame-phone-accuracy-lattice-builder"), modelCombination->lexicon());
    needInit_ = false;
}

Fsa::ConstAutomatonRef SoftFramePhoneAccuracyLatticeRescorerNode::getDistanceFsa(Lattice::ConstWordLatticeRef ref, Lattice::ConstWordLatticeRef hyp) {
    criticalError("not yet implemented");
    return Fsa::ConstAutomatonRef();
}

Fsa::ConstAutomatonRef SoftFramePhoneAccuracyLatticeRescorerNode::getDistanceFsa(const Alignment* ref, Lattice::ConstWordLatticeRef hyp) {
    verify(builder_);
    return builder_->createFunctor(segmentId(), ref, hyp, alignmentGenerator_).build();
}

bool SoftFramePhoneAccuracyLatticeRescorerNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());

    getInputAttributes(2, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<AlignmentGeneratorRef>::type())) {
        return false;
    }

    getInputAttributes(3, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<Alignment>::type())) {
        return false;
    }

    return DistanceLatticeRescorerNode::configure();
}

bool SoftFramePhoneAccuracyLatticeRescorerNode::work(Flow::PortId p) {
    if (!DistanceLatticeRescorerNode::work(p)) {
        return false;
    }

    Flow::DataPtr<Flow::DataAdaptor<Flf::ConstLatticeRef>> inHyp;
    if (!getData(1, inHyp)) {
        error("could not read port hypotheses");
        return putData(0, inHyp.get());
    }

    Flow::DataPtr<Flow::DataAdaptor<AlignmentGeneratorRef>> inAli;
    if (!getData(2, inAli)) {
        error("could not read port alignments");
        return false;
    }
    alignmentGenerator_ = inAli->data();
    if (!alignmentGenerator_) {
        error("ailgnment-generator is empty");
        return false;
    }

    Flow::DataPtr<Flow::DataAdaptor<Alignment>> inRef;
    if (!getData(3, inRef)) {
        error("could not read port references");
        return putData(0, inHyp.get());
    }

    const Alignment&             ref  = inRef->data();
    Lattice::ConstWordLatticeRef hyp  = toWordLattice(inHyp->data());
    Fsa::ConstAutomatonRef       dist = getDistanceFsa(&ref, hyp);
    Flf::ConstLatticeRef         l    = inHyp->data();
    l                                 = Flf::fromFsa(dist, Flf::ConstSemiringRef(new Flf::TropicalSemiring(1)), 0);
    l->setBoundaries(inHyp->data()->getBoundaries());
    Flow::DataAdaptor<Flf::ConstLatticeRef>* out = new Flow::DataAdaptor<Flf::ConstLatticeRef>();
    out->data()                                  = l;
    putData(0, out);

    require(!getData(1, inHyp));
    require(!getData(2, inAli));
    require(!getData(3, inRef));
    return putData(0, inHyp.get());
}

/** WeightedFramePhoneAccuracyLatticeRescorerNode
 */
WeightedFramePhoneAccuracyLatticeRescorerNode::WeightedFramePhoneAccuracyLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          builder_(0) {}

WeightedFramePhoneAccuracyLatticeRescorerNode::~WeightedFramePhoneAccuracyLatticeRescorerNode() {
    delete builder_;
}

void WeightedFramePhoneAccuracyLatticeRescorerNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);
    verify(!builder_);
    builder_  = new WeightedFramePhoneAccuracyLatticeBuilder(select("weighted-frame-phone-accuracy-lattice-builder"), modelCombination->lexicon());
    needInit_ = false;
}

Fsa::ConstAutomatonRef WeightedFramePhoneAccuracyLatticeRescorerNode::getDistanceFsa(Lattice::ConstWordLatticeRef ref, Lattice::ConstWordLatticeRef hyp) {
    verify(builder_);
    return builder_->createFunctor(segmentId(), ref, hyp, alignmentGenerator_).build();
}

/** AcousticLatticeRescorerNode
 */
const Core::Choice AcousticLatticeRescorerNode::choiceRescoreMode(
        "alignment", rescoreModeAlignment,
        "combined", rescoreModeCombined,
        "em", rescoreModeEm,
        "tdp", rescoreModeTdp,
        Core::Choice::endMark());

const Core::ParameterChoice AcousticLatticeRescorerNode::paramRescoreMode(
        "rescore-mode", &choiceRescoreMode, "operation to perfom", rescoreModeAlignment);

AcousticLatticeRescorerNode::AcousticLatticeRescorerNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          rescoreMode_((RescoreMode)paramRescoreMode(c)) {}

bool AcousticLatticeRescorerNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());

    getInputAttributes(2, *attributes);
    if (!configureDatatype(attributes, Flow::DataAdaptor<AlignmentGeneratorRef>::type())) {
        return false;
    }

    return Precursor::configure();
}

bool AcousticLatticeRescorerNode::work(Flow::PortId p) {
    if (!Precursor::work(p)) {
        return false;
    }

    Flow::DataPtr<Flow::DataAdaptor<Flf::ConstLatticeRef>> inLat;
    if (!getData(1, inLat)) {
        error("could not read port hypotheses");
        return putData(0, inLat.get());
    }

    Flow::DataPtr<Flow::DataAdaptor<AlignmentGeneratorRef>> inAli;
    if (!getData(2, inAli)) {
        error("could not read port alignments");
    }
    alignmentGenerator_ = inAli->data();

    Lattice::ConstWordLatticeRef lat  = toWordLattice(inLat->data());
    Fsa::ConstAutomatonRef       resc = getRescoredFsa(lat);

    std::vector<Fsa::ConstAutomatonRef> fsas = Flf::toFsaVector(inLat->data());
    require(!fsas.empty());
    fsas[0] = resc;

    Flf::ConstLatticeRef l = Flf::fromFsaVector(fsas, inLat->data()->semiring());
    l->setBoundaries(inLat->data()->getBoundaries());

    Flow::DataAdaptor<Flf::ConstLatticeRef>* out = new Flow::DataAdaptor<Flf::ConstLatticeRef>();
    out->data()                                  = l;
    putData(0, out);

    require(!getData(1, inLat));
    require(!getData(2, inAli));
    return putData(0, inLat.get());
}

void AcousticLatticeRescorerNode::initialize(ModelCombinationRef modelCombination) {
    Precursor::initialize(modelCombination);
    acousticModel_ = modelCombination->acousticModel();
    needInit_      = false;
}

Fsa::ConstAutomatonRef AcousticLatticeRescorerNode::getRescoredFsa(Lattice::ConstWordLatticeRef lattice) {
    Fsa::ConstAutomatonRef fsa;
    switch (rescoreMode_) {
        case rescoreModeAlignment:
            fsa = Fsa::ConstAutomatonRef(new AlignmentLatticeRescorerAutomaton(lattice, alignmentGenerator_));
            break;
        case rescoreModeCombined:
            fsa = Fsa::ConstAutomatonRef(new CombinedAcousticLatticeRescorerAutomaton(lattice,
                                                                                      alignmentGenerator_,
                                                                                      acousticModel_,
                                                                                      alignmentGenerator_->features(), alignmentGenerator_->allophoneStateGraphBuilder()));
            break;
        case rescoreModeEm:
            fsa = Fsa::ConstAutomatonRef(new EmissionLatticeRescorerAutomaton(lattice,
                                                                              alignmentGenerator_,
                                                                              alignmentGenerator_->features(),
                                                                              acousticModel_));
            break;
        case rescoreModeTdp:
            fsa = Fsa::ConstAutomatonRef(new TdpLatticeRescorerAutomaton(lattice,
                                                                         alignmentGenerator_,
                                                                         alignmentGenerator_->allophoneStateGraphBuilder(),
                                                                         acousticModel_));
            break;
        default:
            criticalError("Unknown mode in AcousticLatticeRescorerNode.");
    }
    return fsa;
}

/** AlignmentAcousticLatticeRescorerNode */
AlignmentAcousticLatticeRescorerNode::AlignmentAcousticLatticeRescorerNode(
        const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

Fsa::ConstAutomatonRef AlignmentAcousticLatticeRescorerNode::getRescoredFsa(
        Lattice::ConstWordLatticeRef lattice) {
    return Fsa::ConstAutomatonRef(
            new AlignmentLatticeRescorerAutomaton(lattice, alignmentGenerator_));
}
