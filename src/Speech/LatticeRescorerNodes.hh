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
#ifndef _SPEECH_LATTICE_RESCORER_NODES_HH
#define _SPEECH_LATTICE_RESCORER_NODES_HH

#include "DataExtractor.hh"
#include "LatticeNodes.hh"
#include "PhonemeSequenceAlignmentGenerator.hh"
#include "SegmentNode.hh"

namespace Bliss {
class OrthographicParser;
}

namespace Speech {

class ApproximatePhoneAccuracyLatticeBuilder;
class FramePhoneAccuracyLatticeBuilder;
class SoftFramePhoneAccuracyLatticeBuilder;
class WeightedFramePhoneAccuracyLatticeBuilder;

/** LatticeRescorerNode */
class LatticeRescorerNode : public SegmentNode {
    typedef SegmentNode Precursor;

public:
    LatticeRescorerNode(const Core::Configuration&);
    virtual Flow::PortId getInput(const std::string& name) {
        return name == "model-combination" ? 0 : 1;
    }
    virtual bool configure();
};

/** NumeratorFromDenominatorNode */
class NumeratorFromDenominatorNode : public LatticeRescorerNode {
    typedef LatticeRescorerNode Precursor;

private:
    static const Core::ParameterString paramSegmentOrth;

private:
    std::string                segmentOrth_;
    Bliss::OrthographicParser* orthToLemma_;
    Fsa::ConstAutomatonRef     lemmaPronToLemma_;
    Fsa::ConstAutomatonRef     lemmaToLemmaConfusion_;

protected:
    virtual void initialize(ModelCombinationRef);

public:
    static std::string filterName() {
        return "lattice-numerator-from-denominator";
    }
    NumeratorFromDenominatorNode(const Core::Configuration&);
    virtual ~NumeratorFromDenominatorNode();
    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool work(Flow::PortId);
};

/** LatticeDistanceRescorerNode */
class DistanceLatticeRescorerNode : public LatticeRescorerNode {
    typedef LatticeRescorerNode Precursor;

public:
    DistanceLatticeRescorerNode(const Core::Configuration&);
    virtual ~DistanceLatticeRescorerNode() {}
};

/** ApproximateLatticeRescorerNode */
class ApproximateDistanceLatticeRescorerNode : public DistanceLatticeRescorerNode {
    typedef DistanceLatticeRescorerNode Precursor;

protected:
    AlignmentGeneratorRef alignmentGenerator_;

protected:
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Lattice::ConstWordLatticeRef) {
        return Fsa::ConstAutomatonRef();
    }

public:
    ApproximateDistanceLatticeRescorerNode(const Core::Configuration&);
    virtual Flow::PortId getInput(const std::string& name) {
        if (name == "model-combination")
            return 0;
        else if (name == "lattice")
            return 1;
        else if (name == "alignment-generator")
            return 2;
        else if (name == "reference")
            return 3;
        else
            return Flow::IllegalPortId;
    }
    virtual bool configure();
    virtual bool work(Flow::PortId);
};

/** ApproximatePhoneAccuracyLatticeRescorerNode */
class ApproximatePhoneAccuracyLatticeRescorerNode : public ApproximateDistanceLatticeRescorerNode {
    typedef ApproximateDistanceLatticeRescorerNode Precursor;

private:
    ApproximatePhoneAccuracyLatticeBuilder* builder_;

protected:
    virtual void                   initialize(ModelCombinationRef);
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Lattice::ConstWordLatticeRef);

public:
    static std::string filterName() {
        return "lattice-approximate-phone-accuracy";
    }
    ApproximatePhoneAccuracyLatticeRescorerNode(const Core::Configuration&);
    virtual ~ApproximatePhoneAccuracyLatticeRescorerNode();
};

/** FramePhoneAccuracyLatticeRescorerNode */
class FramePhoneAccuracyLatticeRescorerNode : public ApproximateDistanceLatticeRescorerNode {
    typedef ApproximateDistanceLatticeRescorerNode Precursor;

private:
    FramePhoneAccuracyLatticeBuilder* builder_;

protected:
    virtual void                   initialize(ModelCombinationRef);
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Lattice::ConstWordLatticeRef);

public:
    static std::string filterName() {
        return "lattice-frame-phone-accuracy";
    }
    FramePhoneAccuracyLatticeRescorerNode(const Core::Configuration&);
    virtual ~FramePhoneAccuracyLatticeRescorerNode();
};

/** SoftFramePhoneAccuracyLatticeRescorerNode */
class SoftFramePhoneAccuracyLatticeRescorerNode : public ApproximateDistanceLatticeRescorerNode {
    typedef ApproximateDistanceLatticeRescorerNode Precursor;

private:
    SoftFramePhoneAccuracyLatticeBuilder* builder_;

protected:
    AlignmentGeneratorRef alignmentGenerator_;

protected:
    virtual void                   initialize(ModelCombinationRef);
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Lattice::ConstWordLatticeRef);
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            const Alignment*, Lattice::ConstWordLatticeRef);

public:
    static std::string filterName() {
        return "lattice-soft-frame-phone-accuracy";
    }
    SoftFramePhoneAccuracyLatticeRescorerNode(const Core::Configuration&);
    virtual ~SoftFramePhoneAccuracyLatticeRescorerNode();

    virtual bool configure();
    virtual bool work(Flow::PortId);
};

/** WeightedFramePhoneAccuracyLatticeRescorerNode */
class WeightedFramePhoneAccuracyLatticeRescorerNode : public ApproximateDistanceLatticeRescorerNode {
    typedef ApproximateDistanceLatticeRescorerNode Precursor;

private:
    WeightedFramePhoneAccuracyLatticeBuilder* builder_;

protected:
    virtual void                   initialize(ModelCombinationRef);
    virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Lattice::ConstWordLatticeRef);

public:
    static std::string filterName() {
        return "lattice-weighted-frame-phone-accuracy";
    }
    WeightedFramePhoneAccuracyLatticeRescorerNode(const Core::Configuration&);
    virtual ~WeightedFramePhoneAccuracyLatticeRescorerNode();
};

/** AcousticLatticeRescorerNode */
class AcousticLatticeRescorerNode : public LatticeRescorerNode {
    typedef LatticeRescorerNode                          Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

protected:
    Core::Ref<Am::AcousticModel> acousticModel_;
    AlignmentGeneratorRef        alignmentGenerator_;

protected:
    virtual void                   initialize(ModelCombinationRef);
    virtual Fsa::ConstAutomatonRef getRescoredFsa(Lattice::ConstWordLatticeRef);

private:
    static const Core::Choice          choiceRescoreMode;
    static const Core::ParameterChoice paramRescoreMode;

    enum RescoreMode {
        rescoreModeAlignment,
        rescoreModeCombined,
        rescoreModeEm,
        rescoreModeTdp
    };

    const RescoreMode rescoreMode_;

public:
    AcousticLatticeRescorerNode(const Core::Configuration&);

    virtual Flow::PortId getInput(const std::string& name) {
        if (name == "model-combination")
            return 0;
        else if (name == "lattice")
            return 1;
        else if (name == "alignment-generator")
            return 2;
        else
            return Flow::IllegalPortId;
    }

    virtual bool                       configure();
    virtual bool                       work(Flow::PortId);
    Core::Ref<const Am::AcousticModel> acousticModel() const {
        return acousticModel_;
    }
    static std::string filterName() {
        return "lattice-acoustic-arc-rescoring";
    }
};

/** AlignmentAcousticLatticeRescorerNode
 * In contrast to CombinedAcousticLatticeRescorer, this
 * rescorer uses the scores from the alignment,
 * which makes rescoring more efficient.
 * However, it is less general than CombinedAcousticLatticeRescorer
 * because the acoustic model for the alignment and scoring are the same.
 */
class AlignmentAcousticLatticeRescorerNode : public AcousticLatticeRescorerNode {
    typedef AcousticLatticeRescorerNode Precursor;

protected:
    virtual Fsa::ConstAutomatonRef getRescoredFsa(Lattice::ConstWordLatticeRef);

public:
    static std::string filterName() {
        return "lattice-alignment-acoustic";
    }
    AlignmentAcousticLatticeRescorerNode(const Core::Configuration&);
};

}  // namespace Speech

#endif  // _SPEECH_LATTICE_RESCORER_NODES_HH
