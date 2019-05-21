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
#include "AdvancedAccuracyFsaBuilder.hh"
#include <Bliss/Evaluation.hh>
#include <Bliss/Lexicon.hh>
#include <Bliss/Orthography.hh>
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Project.hh>
#include <Lattice/Archive.hh>
#include <Lattice/Arithmetic.hh>
#include <Lattice/Basic.hh>
#include <Lattice/Merge.hh>
#include <Lattice/Posterior.hh>
#include <Lattice/SmoothedAccuracy.hh>
#include <Lattice/TimeframeError.hh>

using namespace Speech;

/**
 * LevenshteinNBestListBuilder
 */
class LevenshteinNBestList : public Fsa::SlaveAutomaton {
private:
    Bliss::Evaluator* evaluator_;

public:
    LevenshteinNBestList(Fsa::ConstAutomatonRef, Bliss::Evaluator*);
    virtual ~LevenshteinNBestList() {}

    virtual std::string describe() const {
        return Core::form("levenshtein-list(%s)", fsa_->describe().c_str());
    }
    virtual Fsa::ConstStateRef getState(Fsa::StateId s) const;
};

LevenshteinNBestList::LevenshteinNBestList(Fsa::ConstAutomatonRef list, Bliss::Evaluator* evaluator)
        : Fsa::SlaveAutomaton(Fsa::multiply(list, Fsa::Weight(0.0))),
          evaluator_(evaluator) {}

Fsa::ConstStateRef LevenshteinNBestList::getState(Fsa::StateId s) const {
    if (s == fsa_->initialStateId()) {
        Fsa::State* hypotheses = new Fsa::State(*fsa_->getState(s));
        for (Fsa::State::iterator hyp = hypotheses->begin();
             hyp != hypotheses->end(); ++hyp) {
            Fsa::ConstAutomatonRef hypothesis =
                    Fsa::projectInput(Fsa::partial(fsa_, hyp->target()));
            f32 distance = evaluator_->evaluate(
                    hypothesis, hypothesis->describe());
            hyp->weight_ = Fsa::Weight(distance);
        }
        return Fsa::ConstStateRef(hypotheses);
    }
    return fsa_->getState(s);
}

LevenshteinNBestListBuilder::LevenshteinNBestListBuilder(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          evaluator_(0) {
    verify(!evaluator_);
    evaluator_ = new Bliss::Evaluator(select("evaluation"), lexicon);
}

LevenshteinNBestListBuilder::~LevenshteinNBestListBuilder() {
    delete evaluator_;
}

Fsa::ConstAutomatonRef LevenshteinNBestListBuilder::build(Fsa::ConstAutomatonRef list) {
    if (list) {
        return Fsa::ConstAutomatonRef(new LevenshteinNBestList(list, evaluator_));
    }
    return Fsa::ConstAutomatonRef();
}

LevenshteinNBestListBuilder::Functor LevenshteinNBestListBuilder::createFunctor(const std::string& id, const std::string& orth, Fsa::ConstAutomatonRef list) {
    evaluator_->setReferenceTranscription(orth);
    return Functor(*this, id, list);
}

/**
 * OrthographyApproximatePhoneAccuracyMaskLatticeBuilder
 */
OrthographyApproximatePhoneAccuracyMaskLatticeBuilder::OrthographyApproximatePhoneAccuracyMaskLatticeBuilder(const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon),
          confidenceArchive_(new ConfidenceArchive(select("confidence-archive"))),
          confidences_(new Confidences(select("confidences"))) {
    tokenType_ = phoneType;
    initializeShortPauses(lexicon);
}

OrthographyApproximatePhoneAccuracyMaskLatticeBuilder::~OrthographyApproximatePhoneAccuracyMaskLatticeBuilder() {
    delete confidenceArchive_;
    delete confidences_;
}

OrthographyApproximatePhoneAccuracyMaskLatticeBuilder::Functor
        OrthographyApproximatePhoneAccuracyMaskLatticeBuilder::createFunctor(
                const std::string&                           id,
                const std::string&                           orth,
                Lattice::ConstWordLatticeRef                 lattice,
                Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    alignmentGenerator_ = alignmentGenerator;

    verify(confidenceArchive_ and confidences_);
    confidenceArchive_->get(*confidences_, id);

    return Precursor::createFunctor(id, orth, lattice);
}

Fsa::ConstAutomatonRef OrthographyApproximatePhoneAccuracyMaskLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        require(confidences_->isValid());
        Lattice::ConstWordLatticeRef result = Lattice::getApproximatePhoneAccuracyMask(lattice,
                                                                                       reference_,
                                                                                       *confidences_,
                                                                                       shortPauses_,
                                                                                       alignmentGenerator_);
        return result->mainPart();
    }
    else {
        warning("Approximate phone accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * ArchiveFrameStateAccuracyLatticeBuilder
 */
ArchiveFrameStateAccuracyLatticeBuilder::ArchiveFrameStateAccuracyLatticeBuilder(const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon) {
    tokenType_ = stateType;
}

ArchiveFrameStateAccuracyLatticeBuilder::Functor
        ArchiveFrameStateAccuracyLatticeBuilder::createFunctor(
                const std::string&                           id,
                const std::string&                           segmentId,
                Lattice::ConstWordLatticeRef                 lattice,
                Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    alignmentGenerator_ = alignmentGenerator;
    return Precursor::createFunctor(id, segmentId, lattice);
}

Fsa::ConstAutomatonRef ArchiveFrameStateAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result =
                Lattice::getFrameStateAccuracy(
                        lattice, reference_, shortPauses_, alignmentGenerator_);
        return result->mainPart();
    }
    else {
        warning("Frame state accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * OrthographyFrameStateAccuracyLatticeBuilder
 */
OrthographyFrameStateAccuracyLatticeBuilder::OrthographyFrameStateAccuracyLatticeBuilder(const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon),
          lexicon_(lexicon) {
    tokenType_ = stateType;
    // see @function createFunctor
    // initializeShortPauses(lexicon);
}

OrthographyFrameStateAccuracyLatticeBuilder::Functor
        OrthographyFrameStateAccuracyLatticeBuilder::createFunctor(
                const std::string&                           id,
                const std::string&                           orth,
                Lattice::ConstWordLatticeRef                 lattice,
                Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    alignmentGenerator_ = alignmentGenerator;
    if (shortPauses_.begin() == shortPauses_.end()) {
        shortPauses_.insert(Fsa::InvalidLabelId);
        std::vector<std::string> shortPausesLemmata = paramShortPausesLemmata(config);
        if (!shortPausesLemmata.empty()) {
            if (shortPausesLemmata.size() == 1) {
                std::string silence(shortPausesLemmata.front());
                Core::normalizeWhitespace(silence);
                log("Append short pause lemma \"") << silence << "\"";
                const Bliss::Lemma* lemma = lexicon_->lemma(silence);
                if (lemma == lexicon_->specialLemma("silence")) {
                    const Fsa::LabelId silAlloStateId = alignmentGenerator_->acousticModel()->silenceAllophoneStateIndex();
                    shortPauses_.insert(silAlloStateId);
                }
                else {
                    error("Lemma must be the special lemma \"silence\"");
                }
            }
            else {
                error("Lemma can be only the special lemma \"silence\"");
            }
        }
    }
    return Precursor::createFunctor(id, orth, lattice);
}

Fsa::ConstAutomatonRef OrthographyFrameStateAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result = Lattice::getFrameStateAccuracy(lattice,
                                                                             reference_,
                                                                             shortPauses_,
                                                                             alignmentGenerator_);
        return result->mainPart();
    }
    else {
        warning("Frame state accuracies cannot be calculated because of empty reference.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * OrthographySmoothedFrameStateAccuracyLatticeBuilder
*/
OrthographySmoothedFrameStateAccuracyLatticeBuilder::OrthographySmoothedFrameStateAccuracyLatticeBuilder(
        const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon),
          smoothing_(Lattice::SmoothingFunction::createSmoothingFunction(select("smoothing-function"))) {
    tokenType_ = stateType;
    initializeShortPauses(lexicon);

    if (!smoothing_) {
        criticalError("smoothing function could not be instantiated");
    }
}

OrthographySmoothedFrameStateAccuracyLatticeBuilder::~OrthographySmoothedFrameStateAccuracyLatticeBuilder() {
    smoothing_->dumpStatistics(clog());
}

OrthographySmoothedFrameStateAccuracyLatticeBuilder::Functor
        OrthographySmoothedFrameStateAccuracyLatticeBuilder::createFunctor(
                const std::string&                           id,
                const std::string&                           orth,
                Lattice::ConstWordLatticeRef                 lattice,
                Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    alignmentGenerator_ = alignmentGenerator;
    /*
     * It is assumed that @param lattice contains the
     * total scores. We would like to calculate
     * f'(E[\chi_{spk,t}])E[\chi_{spk,t}], cf.
     * "accuracy" lattice in gradient of unified
     * training criterion. So, the arc posteriors
     * are calculated before passing the lattice
     * for initialization.
     */
    Lattice::ConstWordLatticeRef post = Lattice::expm(
            Lattice::changeSemiring(
                    Lattice::posterior(
                            Lattice::changeSemiring(
                                    lattice,
                                    Fsa::LogSemiring)),
                    lattice->mainPart()->semiring()));
    return Precursor::createFunctor(id, orth, post);
}

Fsa::ConstAutomatonRef OrthographySmoothedFrameStateAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        verify(smoothing_);
        Lattice::ConstWordLatticeRef result =
                Lattice::getSmoothedFrameStateAccuracy(
                        lattice, reference_, alignmentGenerator_, *smoothing_);
        return result->mainPart();
    }
    else {
        warning("Smoothed frame state accuracies cannot be calculated because of empty reference.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * OrthographyFrameWordAccuracyLatticeBuilder
 */
const Core::ParameterFloat OrthographyFrameWordAccuracyLatticeBuilder::paramNormalization(
        "normalization-scale",
        "normalization scale for computing timeframe accuracy",
        1, 0, 1);

OrthographyFrameWordAccuracyLatticeBuilder::OrthographyFrameWordAccuracyLatticeBuilder(const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon),
          normalization_(paramNormalization(config)) {
    tokenType_ = (TokenType)paramTokenType(config);
    if (tokenType_ != lemmaPronunciationType && tokenType_ != lemmaType) {
        criticalError("Invalid token type");
    }
    initializeShortPauses(lexicon);
}

Fsa::ConstAutomatonRef OrthographyFrameWordAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result = Lattice::getWordTimeframeAccuracy(lattice,
                                                                                reference_,
                                                                                shortPauses_,
                                                                                tokenType_ == lemmaType,
                                                                                normalization_);
        return result->mainPart();
    }
    else {
        warning("Word timeframe accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * OrthographyFramePhoneAccuracyLatticeBuilder
 */
OrthographyFramePhoneAccuracyLatticeBuilder::OrthographyFramePhoneAccuracyLatticeBuilder(const Core::Configuration& c, Core::Ref<const Bliss::Lexicon> lexicon)
        : Precursor(c, lexicon) {
    tokenType_ = stateType;
}

OrthographyFramePhoneAccuracyLatticeBuilder::Functor
        OrthographyFramePhoneAccuracyLatticeBuilder::createFunctor(
                const std::string&                           id,
                const std::string&                           orth,
                Lattice::ConstWordLatticeRef                 lattice,
                Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    alignmentGenerator_ = alignmentGenerator;
    return Precursor::createFunctor(id, orth, lattice);
}

Fsa::ConstAutomatonRef OrthographyFramePhoneAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result = Lattice::getFramePhoneAccuracy(lattice, reference_, shortPauses_, alignmentGenerator_, normalization_);
        return result->mainPart();
    }
    else {
        warning("Frame state accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * FramePhoneAccuracyLatticeBuilder
 */
const Core::ParameterFloat FramePhoneAccuracyLatticeBuilder::paramNormalization(
        "normalization-scale",
        "normalization scale for computing frame phone accuracy",
        0, 0, 1);

FramePhoneAccuracyLatticeBuilder::FramePhoneAccuracyLatticeBuilder(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon),
          normalization_(paramNormalization(config)) {
    tokenType_ = phoneType;
    initializeShortPauses(lexicon);
}

FramePhoneAccuracyLatticeBuilder::Functor
        FramePhoneAccuracyLatticeBuilder::createFunctor(
                const std::string&           id,
                Lattice::ConstWordLatticeRef reference,
                Lattice::ConstWordLatticeRef lattice,
                AlignmentGeneratorRef        alignmentGenerator) {
    setReference(reference);
    alignmentGenerator_ = alignmentGenerator;
    return Functor(*this, id, lattice);
}

Fsa::ConstAutomatonRef FramePhoneAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result = Lattice::getFramePhoneAccuracy(lattice,
                                                                             reference_,
                                                                             shortPauses_,
                                                                             alignmentGenerator_,
                                                                             normalization_);
        return result->mainPart();
    }
    else {
        warning("Frame phone accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * SoftFramePhoneAccuracyLatticeBuilder
 */
SoftFramePhoneAccuracyLatticeBuilder::SoftFramePhoneAccuracyLatticeBuilder(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon) {
    tokenType_ = phoneType;
    initializeShortPauses(lexicon);
}

void SoftFramePhoneAccuracyLatticeBuilder::setReference(const Alignment* forcedAlignment) {
    reference_.reset();
    forcedAlignment_ = 0;
    if (forcedAlignment) {
        forcedAlignment_ = forcedAlignment;
    }
}

SoftFramePhoneAccuracyLatticeBuilder::Functor
        SoftFramePhoneAccuracyLatticeBuilder::createFunctor(
                const std::string&           id,
                Lattice::ConstWordLatticeRef reference,
                Lattice::ConstWordLatticeRef lattice,
                AlignmentGeneratorRef        alignmentGenerator) {
    forcedAlignment_ = 0;
    Precursor::setReference(reference);
    alignmentGenerator_ = alignmentGenerator;
    return Functor(*this, id, lattice);
}

SoftFramePhoneAccuracyLatticeBuilder::Functor
        SoftFramePhoneAccuracyLatticeBuilder::createFunctor(
                const std::string&           id,
                const Alignment*             forcedAlignment,
                Lattice::ConstWordLatticeRef lattice,
                AlignmentGeneratorRef        alignmentGenerator) {
    setReference(forcedAlignment);
    alignmentGenerator_ = alignmentGenerator;
    return Functor(*this, id, lattice);
}

Fsa::ConstAutomatonRef SoftFramePhoneAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        return Lattice::getSoftFramePhoneAccuracy(lattice,
                                                  reference_,
                                                  shortPauses_,
                                                  alignmentGenerator_)
                ->mainPart();
    }
    else if (forcedAlignment_) {
        return Lattice::getSoftFramePhoneAccuracy(lattice,
                                                  *forcedAlignment_,
                                                  shortPauses_,
                                                  alignmentGenerator_)
                ->mainPart();
    }
    else {
        warning("Soft frame phone accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}

/**
 * WeightedFramePhoneAccuracyLatticeBuilder
 */
const Core::ParameterFloat WeightedFramePhoneAccuracyLatticeBuilder::paramBeta(
        "beta",
        "parameter to control smoothness of sigmoid function",
        1, 0);

const Core::ParameterFloat WeightedFramePhoneAccuracyLatticeBuilder::paramMargin(
        "margin",
        "parameter to control margin, i.e., offset of sigmoid function",
        0);

WeightedFramePhoneAccuracyLatticeBuilder::WeightedFramePhoneAccuracyLatticeBuilder(
        const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon),
          beta_(paramBeta(config)),
          margin_(paramMargin(config)) {
    tokenType_ = phoneType;
    initializeShortPauses(lexicon);
}

WeightedFramePhoneAccuracyLatticeBuilder::Functor
        WeightedFramePhoneAccuracyLatticeBuilder::createFunctor(
                const std::string&           id,
                Lattice::ConstWordLatticeRef reference,
                Lattice::ConstWordLatticeRef lattice,
                AlignmentGeneratorRef        alignmentGenerator) {
    setReference(reference);
    alignmentGenerator_ = alignmentGenerator;
    return Functor(*this, id, lattice);
}

Fsa::ConstAutomatonRef WeightedFramePhoneAccuracyLatticeBuilder::build(Lattice::ConstWordLatticeRef lattice) {
    if (reference_) {
        Lattice::ConstWordLatticeRef result = Lattice::getWeightedFramePhoneAccuracy(lattice,
                                                                                     reference_,
                                                                                     shortPauses_,
                                                                                     alignmentGenerator_,
                                                                                     beta_,
                                                                                     margin_);
        return result->mainPart();
    }
    else {
        warning("Weighted frame phone accuracies cannot be calculated because reference is empty.");
        return Fsa::ConstAutomatonRef();
    }
}
