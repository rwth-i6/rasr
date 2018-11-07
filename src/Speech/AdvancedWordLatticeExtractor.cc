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
#include "AdvancedWordLatticeExtractor.hh"
#include <Bliss/Lexicon.hh>
#include <Bliss/Orthography.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Linear.hh>
#include <Fsa/Project.hh>
#include <Lattice/Basic.hh>
#include <Lattice/Merge.hh>
#include <Lattice/Rational.hh>
#include <Lattice/Static.hh>
#include <Lm/Module.hh>
#include "AdvancedLatticeExtractor.hh"
#include "DataExtractor.hh"

using namespace Speech;

/**
 * TimeConditionedLatticeSetProcessor
 */
TimeConditionedLatticeSetProcessor::TimeConditionedLatticeSetProcessor(
    const Core::Configuration &c)
    :
    Core::Component(c),
    Precursor(c)
{}

TimeConditionedLatticeSetProcessor::~TimeConditionedLatticeSetProcessor()
{}

void TimeConditionedLatticeSetProcessor::processWordLattice(
    Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment *s)
{
    Lattice::ConstWordLatticeRef timeConditioned =
        Lattice::timeConditionedWordLattice(lattice);
    Precursor::processWordLattice(timeConditioned, s);
}

void TimeConditionedLatticeSetProcessor::setWordLatticeDescription(
    const Lattice::WordLatticeDescription &description)
{
    for (size_t i = 0; i < description.nStreams(); ++ i) {
        bool isAcousticFsa = description[i].verifyValue(
            Lattice::WordLatticeDescription::nameModel,
            std::string(Lattice::WordLattice::acousticFsa));
        bool isWordLevel = description[i].verifyValue(
            Lattice::WordLatticeDescription::nameModel,
            std::string(Lattice::WordLattice::acousticFsa));
        if (isAcousticFsa and isWordLevel) {
            return;
        }
    }
    criticalError("Input lattice does not have an acoustic fsa at word level");
}

/**
 * NumeratorLatticeGenerator
 */
NumeratorLatticeGenerator::NumeratorLatticeGenerator(const Core::Configuration &c) :
    Core::Component(c),
    Precursor(c),
    recognizer_(0),
    orthToLemma_(0)
{}

NumeratorLatticeGenerator::~NumeratorLatticeGenerator()
{
    delete recognizer_;
    delete orthToLemma_;
}

void NumeratorLatticeGenerator::signOn(CorpusVisitor &corpusVisitor)
{
    verify(segmentwiseFeatureExtractor_);
    segmentwiseFeatureExtractor_->signOn(corpusVisitor);
    segmentwiseFeatureExtractor_->respondToDelayedErrors();
    Precursor::signOn(corpusVisitor);
}

void NumeratorLatticeGenerator::leaveSpeechSegment(Bliss::SpeechSegment *s)
{
    verify(recognizer_);
    verify(orthToLemma_);
    Core::Ref<Lattice::WordLattice> orth(new Lattice::WordLattice);
    orth->setFsa(orthToLemma_->createLemmaAcceptor(s->orth()), Lattice::WordLattice::acousticFsa);
    Lattice::ConstWordLatticeRef lattice = recognizer_->extract(orth, s);
    if (lattice and lattice->nParts() == 1) {
        processWordLattice(lattice, s);
    } else {
        log("Skip this segment because numerator lattice could not be generated.");
    }
    Precursor::leaveSpeechSegment(s);
}

void NumeratorLatticeGenerator::initialize(Bliss::LexiconRef lexicon)
{
    Precursor::initialize(lexicon);
    verify(!recognizer_);
    recognizer_ = new RecognizerWithConstrainedLanguageModel(
        select("constrained-reocgnizer"), lexicon);
    segmentwiseFeatureExtractor_ = Core::ref(
        new SegmentwiseFeatureExtractor(
            select("segmentwise-feature-extraction")));
    recognizer_->setSegmentwiseFeatureExtractor(segmentwiseFeatureExtractor_);

    verify(!orthToLemma_);
    orthToLemma_ = new Bliss::OrthographicParser(select("orthographic-parser"), lexicon);
}
