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
#ifndef _SPEECH_NBEST_LIST_EXTRACTOR_HH
#define _SPEECH_NBEST_LIST_EXTRACTOR_HH

#include <Core/Parameter.hh>
#include <Lattice/Best.hh>
#include "LatticeSetProcessor.hh"

namespace Speech {

class NBestListExtractor : public Core::Component, Lattice::NBestListExtractor {
private:
    static const Core::ParameterInt   paramNumberOfHypotheses;
    static const Core::ParameterFloat paramMinPruningThreshold;
    static const Core::ParameterFloat paramMaxPruningThreshold;
    static const Core::ParameterFloat paramPruningIncrement;
    static const Core::ParameterBool  paramWorkOnOutput;
    static const Core::ParameterBool  paramLatticeIsDeterministic;
    static const Core::ParameterBool  paramHasFailArcs;
    static const Core::ParameterBool  paramNormalize;

public:
    NBestListExtractor(const Core::Configuration&);
    virtual ~NBestListExtractor();

    void initialize(Bliss::LexiconRef lexicon) {
        Lattice::NBestListExtractor::initialize(lexicon);
    }
    Lattice::ConstWordLatticeRef getNBestList(Lattice::ConstWordLatticeRef l) {
        return Lattice::NBestListExtractor::getNBestList(l);
    }
};

class NBestListLatticeProcessor : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

private:
    mutable Core::XmlChannel statisticsChannel_;
    NBestListExtractor       extractor_;

public:
    NBestListLatticeProcessor(const Core::Configuration&);
    virtual ~NBestListLatticeProcessor();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
    virtual void initialize(Bliss::LexiconRef);
};

}  // namespace Speech

#endif  // _SPEECH_NBEST_LIST_EXTRACTOR_HH
