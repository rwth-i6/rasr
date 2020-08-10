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
#ifndef _SPEECH_ADVANCED_WORD_LATTICE_EXTRACTOR_HH
#define _SPEECH_ADVANCED_WORD_LATTICE_EXTRACTOR_HH

#include "WordLatticeExtractor.hh"

namespace Speech {
class RecognizerWithConstrainedLanguageModel;
class SegmentwiseFeatureExtractor;
}  // namespace Speech

namespace Speech {

/**
 *  WordLatticeWithoutRedundantSilencesAndNoises
 */
class WordLatticeWithoutRedundantSilencesAndNoises : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

private:
    static const Core::ParameterBool paramShouldPruneSoftly;

private:
    Bliss::LexiconRef lexicon_;
    bool              shouldPruneSoftly_;

public:
    WordLatticeWithoutRedundantSilencesAndNoises(const Core::Configuration&);
    virtual ~WordLatticeWithoutRedundantSilencesAndNoises();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
    virtual void initialize(Bliss::LexiconRef);
};

/**
 *  Time-conditioned
 */
class TimeConditionedLatticeSetProcessor : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    TimeConditionedLatticeSetProcessor(const Core::Configuration&);
    virtual ~TimeConditionedLatticeSetProcessor();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
    virtual void setWordLatticeDescription(const Lattice::WordLatticeDescription&);
};

/**
 *  NumeratorLatticeGenerator
 */
class NumeratorLatticeGenerator : public LatticeSetProcessorRoot {
    typedef LatticeSetProcessorRoot Precursor;

private:
    RecognizerWithConstrainedLanguageModel* recognizer_;
    Bliss::OrthographicParser*              orthToLemma_;
    Core::Ref<SegmentwiseFeatureExtractor>  segmentwiseFeatureExtractor_;

public:
    NumeratorLatticeGenerator(const Core::Configuration&);
    virtual ~NumeratorLatticeGenerator();

    virtual void signOn(CorpusVisitor&);
    virtual void leaveSpeechSegment(Bliss::SpeechSegment*);
    virtual void initialize(Bliss::LexiconRef);
};

}  // namespace Speech

#endif  // _SPEECH_ADVANCED_WORD_LATTICE_EXTRACTOR_HH
