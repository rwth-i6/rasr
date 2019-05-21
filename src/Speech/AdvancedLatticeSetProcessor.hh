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
#ifndef _SPEECH_ADVANCED_LATTICE_SET_PROCESSOR_HH
#define _SPEECH_ADVANCED_LATTICE_SET_PROCESSOR_HH

#include "LatticeSetProcessor.hh"

namespace Speech {

class ChangeSemiringLatticeProcessorNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

private:
    static Core::Choice          choiceSemiringType;
    static Core::ParameterChoice paramSemiringType;

private:
    Fsa::ConstSemiringRef semiring_;

public:
    ChangeSemiringLatticeProcessorNode(const Core::Configuration&);
    virtual ~ChangeSemiringLatticeProcessorNode() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

class MultiplyLatticeProcessorNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

private:
    /**
     *  behaviour:
     *      -componentwise multiplication, if @param paramFactors is configured
     *      -vector multiplication with scalar @param paramFactor, otherwise
     */
    static const Core::ParameterFloat       paramFactor;
    static const Core::ParameterFloatVector paramFactors;

private:
    Fsa::Weight              factor_;
    std::vector<Fsa::Weight> factors_;

public:
    MultiplyLatticeProcessorNode(const Core::Configuration&);
    virtual ~MultiplyLatticeProcessorNode() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

class ExtendBestPathLatticeProcessorNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    ExtendBestPathLatticeProcessorNode(const Core::Configuration&);
    virtual ~ExtendBestPathLatticeProcessorNode() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

class MapToNonCoarticulationLatticeProcessorNode : public LatticeSetProcessor {
private:
    typedef LatticeSetProcessor Precursor;

public:
    MapToNonCoarticulationLatticeProcessorNode(const Core::Configuration&);
    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

class TokenMappingLatticeProcessorNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

protected:
    Bliss::LexiconRef      lexicon_;
    Fsa::ConstAutomatonRef lemmaPronToLemma_;

public:
    TokenMappingLatticeProcessorNode(const Core::Configuration&);
    virtual ~TokenMappingLatticeProcessorNode() {}

    virtual void initialize(Bliss::LexiconRef);
};

class LemmaPronunciationToEvaluationToken : public TokenMappingLatticeProcessorNode {
    typedef TokenMappingLatticeProcessorNode Precursor;

protected:
    Fsa::ConstAutomatonRef lemmaToEval_;

public:
    LemmaPronunciationToEvaluationToken(const Core::Configuration&);
    virtual ~LemmaPronunciationToEvaluationToken() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
    virtual void initialize(Bliss::LexiconRef);
};

class LemmaPronunciationToSyntacticToken : public TokenMappingLatticeProcessorNode {
    typedef TokenMappingLatticeProcessorNode Precursor;

protected:
    Fsa::ConstAutomatonRef lemmaToSynt_;

public:
    LemmaPronunciationToSyntacticToken(const Core::Configuration&);
    virtual ~LemmaPronunciationToSyntacticToken() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
    virtual void initialize(Bliss::LexiconRef);
};

class DumpWordBoundariesNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    DumpWordBoundariesNode(const Core::Configuration&);
    virtual ~DumpWordBoundariesNode() {}

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

class MinimumMaximumWeightNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

private:
    static const Core::ParameterFloat paramMinimumErrorLevel;
    static const Core::ParameterFloat paramMaximumErrorLevel;

private:
    std::pair<f32, f32> errorLevel_;
    std::pair<f32, f32> minMax_;

private:
    void accumulate(const std::pair<Fsa::Weight, Fsa::Weight>& minMax);

public:
    MinimumMaximumWeightNode(const Core::Configuration&);
    virtual ~MinimumMaximumWeightNode() {}

    virtual void leaveCorpus(Bliss::Corpus*);
    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

/**
 * Expm node
 */
class ExpmNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    ExpmNode(const Core::Configuration&);
    virtual ~ExpmNode();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

/**
 * epsilon removal
 */
class EpsilonRemoval : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    EpsilonRemoval(const Core::Configuration&);
    virtual ~EpsilonRemoval();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};

/**
 * determinize
 */
class DeterminizeNode : public LatticeSetProcessor {
    typedef LatticeSetProcessor Precursor;

public:
    DeterminizeNode(const Core::Configuration&);
    virtual ~DeterminizeNode();

    virtual void processWordLattice(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment*);
};
}  // namespace Speech

#endif  // _SPEECH_ADVANCED_LATTICE_SET_PROCESSOR_HH
