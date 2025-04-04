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
 */
#include "AdvancedLatticeSetProcessor.hh"
#include <Bliss/Evaluation.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Project.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Lattice/Arithmetic.hh>
#include <Lattice/Basic.hh>
#include <Lattice/Cache.hh>
#include <Lattice/Compose.hh>
#include <Lattice/RemoveEpsilons.hh>
#include <Lattice/Static.hh>
#include <Lattice/Utilities.hh>
#include "DataExtractor.hh"
#include "PhonemeSequenceAlignmentGenerator.hh"

using namespace Speech;

/**
 * ChangeSemiringLatticeProcessorNode
 */
Core::Choice ChangeSemiringLatticeProcessorNode::choiceSemiringType(
        "unknown", Fsa::SemiringTypeUnknown,
        "log", Fsa::SemiringTypeLog,
        "tropical", Fsa::SemiringTypeTropical,
        "tropical-integer", Fsa::SemiringTypeTropicalInteger,
        "count", Fsa::SemiringTypeCount,
        "probability", Fsa::SemiringTypeProbability,
        Core::Choice::endMark());

Core::ParameterChoice ChangeSemiringLatticeProcessorNode::paramSemiringType(
        "semiring-type",
        &choiceSemiringType,
        "type of semiring",
        Fsa::SemiringTypeUnknown);

ChangeSemiringLatticeProcessorNode::ChangeSemiringLatticeProcessorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          semiring_(Fsa::getSemiring((Fsa::SemiringType)paramSemiringType(c))) {
    if (semiring_ == Fsa::UnknownSemiring)
        error("Parameter 'semiring-type' needs to be set");
}

void ChangeSemiringLatticeProcessorNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    Precursor::processWordLattice(Lattice::changeSemiring(lattice, semiring_), s);
}

/**
 * MultiplyLatticeProcessorNode
 */
const Core::ParameterFloat MultiplyLatticeProcessorNode::paramFactor(
        "factor",
        "multiply all scores with this factor",
        1);

const Core::ParameterFloatVector MultiplyLatticeProcessorNode::paramFactors(
        "factors",
        "multiply scores componentwise with this factors");

MultiplyLatticeProcessorNode::MultiplyLatticeProcessorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          factor_(Fsa::Weight(paramFactor(config))) {
    std::vector<f64> factors = paramFactors(config);
    for (u32 i = 0; i < factors.size(); ++i) {
        factors_.push_back(Fsa::Weight(f32(factors[i])));
    }
}

void MultiplyLatticeProcessorNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    Lattice::ConstWordLatticeRef result = lattice;
    if (factors_.empty()) {
        if (f32(factor_) != 1) {
            result = Lattice::multiply(lattice, factor_);
        }
    }
    else {
        if (factors_.size() != lattice->nParts()) {
            criticalError("mismatch in number of factors and number of lattice parts");
        }
        result = Lattice::multiply(lattice, factors_);
    }
    Precursor::processWordLattice(result, s);
}

/**
 * ExtendBestPathLatticeProcessorNode
 */
ExtendBestPathLatticeProcessorNode::ExtendBestPathLatticeProcessorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void ExtendBestPathLatticeProcessorNode::processWordLattice(Lattice::ConstWordLatticeRef lattice,
                                                            Bliss::SpeechSegment*        s) {
    Fsa::Weight minimum(-f32(Fsa::bestscore(lattice->mainPart())));
    Precursor::processWordLattice(Lattice::extendFinal(lattice, minimum), s);
}

/**
 * MapToNonCoarticulationLatticeProcessorNode
 */
MapToNonCoarticulationLatticeProcessorNode::MapToNonCoarticulationLatticeProcessorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void MapToNonCoarticulationLatticeProcessorNode::processWordLattice(Lattice::ConstWordLatticeRef l, Bliss::SpeechSegment* s) {
    Core::Ref<Lattice::WordLattice>      result(new Lattice::WordLattice(*l));
    Core::Ref<Lattice::WordBoundaries>   r(new Lattice::WordBoundaries(*l->wordBoundaries()));
    const Lattice::WordBoundary::Transit nonCoarticulation;
    for (u32 i = 0; i < r->size(); ++i) {
        (*r)[i].setTransit(nonCoarticulation);
    }
    result->setWordBoundaries(r);
    return Precursor::processWordLattice(result, s);
}

/**
 * TokenMappingLatticeProcessorNode
 */
TokenMappingLatticeProcessorNode::TokenMappingLatticeProcessorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void TokenMappingLatticeProcessorNode::initialize(
        Bliss::LexiconRef lexicon) {
    Precursor::initialize(lexicon);

    require(lexicon);
    lexicon_          = lexicon;
    lemmaPronToLemma_ = Fsa::cache(Fsa::multiply(lexicon->createLemmaPronunciationToLemmaTransducer(),
                                                 Fsa::Weight(f32(0))));
}

/**
 * LemmaPronunciationToEvaluationToken
 */
LemmaPronunciationToEvaluationToken::LemmaPronunciationToEvaluationToken(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void LemmaPronunciationToEvaluationToken::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    if (lattice->mainPart()->getInputAlphabet() != lexicon_->lemmaPronunciationAlphabet()) {
        criticalError("Input alphabet must be the lemma pronuncation alphabet.");
    }

    Fsa::ConstAutomatonRef evalFsa = Fsa::cache(
            Fsa::projectOutput(
                    Fsa::composeMatching(
                            Fsa::composeMatching(lattice->mainPart(),
                                                 Fsa::multiply(lemmaPronToLemma_, Fsa::Weight(0.0))),
                            Fsa::multiply(lemmaToEval_, Fsa::Weight(0.0)))));
    Core::Ref<Lattice::WordLattice> eval(new Lattice::WordLattice);
    eval->setFsa(evalFsa, lattice->mainName());

    Precursor::processWordLattice(eval, s);
}

void LemmaPronunciationToEvaluationToken::initialize(
        Bliss::LexiconRef lexicon) {
    Precursor::initialize(lexicon);

    lemmaToEval_ = lexicon->createLemmaToEvaluationTokenTransducer();
}

/**
 * LemmaPronunciationToSyntacticToken
 */
LemmaPronunciationToSyntacticToken::LemmaPronunciationToSyntacticToken(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void LemmaPronunciationToSyntacticToken::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    if (lattice->mainPart()->getInputAlphabet() != lexicon_->lemmaPronunciationAlphabet()) {
        criticalError("Input alphabet must be the lemma pronuncation alphabet.");
    }

    Fsa::ConstAutomatonRef lemmaPronToSynt = Fsa::composeMatching(Fsa::composeMatching(lattice->mainPart(),
                                                                                       lemmaPronToLemma_),
                                                                  lemmaToSynt_);

    Core::Ref<Lattice::WordLattice> synt(new Lattice::WordLattice);
    synt->setFsa(
            Fsa::cache(
                    Fsa::removeEpsilons(
                            Fsa::cache(
                                    Fsa::projectOutput(
                                            lemmaPronToSynt)))),
            lattice->mainName());

    Precursor::processWordLattice(synt, s);
}

void LemmaPronunciationToSyntacticToken::initialize(Bliss::LexiconRef lexicon) {
    Precursor::initialize(lexicon);

    lemmaToSynt_ = Fsa::cache(Fsa::multiply(lexicon->createLemmaToSyntacticTokenTransducer(), Fsa::Weight(f32(0))));
}

/**
 * DumpWordBoundariesNode
 */
DumpWordBoundariesNode::DumpWordBoundariesNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

void DumpWordBoundariesNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    if (lattice && lattice->wordBoundaries()) {
        Lattice::dumpWordBoundaries(lattice->wordBoundaries(), clog());
    }
    Precursor::processWordLattice(lattice, s);
}

/**
 * MinimumMaximumWeightNode
 */
const Core::ParameterFloat MinimumMaximumWeightNode::paramMinimumErrorLevel(
        "error-level", "if minimum is below this value, error is generated",
        Core::Type<f32>::min);

const Core::ParameterFloat MinimumMaximumWeightNode::paramMaximumErrorLevel(
        "error-level", "if maximum exceeds this value, error is generated",
        Core::Type<f32>::max);

MinimumMaximumWeightNode::MinimumMaximumWeightNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          errorLevel_(paramMinimumErrorLevel(config), paramMaximumErrorLevel(config)),
          minMax_(Core::Type<f32>::max, Core::Type<f32>::min) {}

void MinimumMaximumWeightNode::accumulate(const std::pair<Fsa::Weight, Fsa::Weight>& minMax) {
    minMax_.first = minMax_.first < f32(minMax.first) ? minMax_.first : f32(minMax.first);
    if (f32(minMax.first) < errorLevel_.first) {
        error("Minimum exceeded the error level %f\n", errorLevel_.first) << f32(minMax.first);
    }
    minMax_.second = minMax_.second > f32(minMax.second) ? minMax_.second : f32(minMax.second);
    if (f32(minMax.second) < errorLevel_.second) {
        error("Maximum exceeded the error level %f\n", errorLevel_.second) << f32(minMax.second);
    }
}

void MinimumMaximumWeightNode::leaveCorpus(Bliss::Corpus* corpus) {
    if (corpus->level() == 0) {
        if (f32(minMax_.first) < errorLevel_.first) {
            error("Minimum exceeded the error level %f\n", errorLevel_.first) << f32(minMax_.first);
        }
        if (f32(minMax_.second) < errorLevel_.second) {
            error("Maximum exceeded the error level %f\n", errorLevel_.second) << f32(minMax_.second);
        }
    }
}

void MinimumMaximumWeightNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    if (lattice && lattice->mainPart()) {
        accumulate(Lattice::minMaxWeights(lattice));
    }
    Precursor::processWordLattice(lattice, s);
}

/**
 * ExpmNode
 */
ExpmNode::ExpmNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

ExpmNode::~ExpmNode() {}

void ExpmNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    if (lattice) {
        Precursor::processWordLattice(Lattice::expm(lattice), segment);
    }
    else {
        error("skip segment because lattice is empty");
    }
}

/**
 * epsilon removal
 */
EpsilonRemoval::EpsilonRemoval(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c) {}

EpsilonRemoval::~EpsilonRemoval() {}

void EpsilonRemoval::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    if (lattice) {
        Precursor::processWordLattice(Lattice::removeEpsilons(lattice), segment);
    }
    else {
        error("skip segment because lattice is empty");
    }
}
