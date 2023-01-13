#ifndef _FLF_ACOUSTIC_ALIGNMENT_HH
#define _FLF_ACOUSTIC_ALIGNMENT_HH

#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Core/ReferenceCounting.hh>
#include <Speech/Alignment.hh>

#include <Flf/FlfCore/Lattice.hh>
#include <Flf/FlfCore/LatticeInternal.hh>
#include <Flf/FwdBwd.hh>
#include <Flf/Lexicon.hh>
#include <Flf/Map.hh>
#include <Flf/Network.hh>
#include <Flf/RescoreInternal.hh>

namespace Flf {

class AcousticPhonemeSequenceAligner;

/*
 * class WeightedAlignment : public Speech::Alignment, public Core::ReferenceCounted {};
 * typedef Core::Ref<const WeightedAlignment> ConstWeightedAlignmentRef;
 */

struct StateAlignment {
    const Bliss::LemmaPronunciation* lemmaPron;
    const Speech::Alignment*         alignment;
    Speech::Score                    emissionScore;
    Speech::Score                    transitionScore;
    StateAlignment();
};

struct SubWord {
    Fsa::LabelId       label;
    Time               duration;
    Bliss::Phoneme::Id leftContext, rightContext;
    SubWord(Fsa::LabelId label, Time duration)
            : label(label), duration(duration), leftContext(Bliss::Phoneme::term), rightContext(Bliss::Phoneme::term) {}
    SubWord(Fsa::LabelId label, Time duration, Bliss::Phoneme::Id leftContext, Bliss::Phoneme::Id rightContext)
            : label(label), duration(duration), leftContext(leftContext), rightContext(rightContext) {}
};

class SubWordAlignment : public std::vector<SubWord>, public Core::ReferenceCounted {
public:
    typedef std::vector<SubWord> Precursor;

private:
    Fsa::LabelId label_;

public:
    SubWordAlignment()
            : Precursor(), label_(Fsa::InvalidLabelId) {}
    SubWordAlignment(Fsa::LabelId label)
            : Precursor(), label_(label) {}
    SubWordAlignment(Fsa::LabelId label, const SubWord& sw)
            : Precursor(1, sw), label_(label) {}
    void setLabel(Fsa::LabelId label) {
        label_ = label;
    }
    Fsa::LabelId label() const {
        return label_;
    }
    Time duration() const;
};
typedef Core::Ref<const SubWordAlignment> ConstSubWordAlignmentRef;

/*
 * Produce arc-wise alignments for a lattice
 */
class LatticeAlignment : public Core::ReferenceCounted {
private:
    ConstLatticeRef                 l_;
    bool                            isLemma_;
    AcousticPhonemeSequenceAligner* aligner_;
    LabelMapList                    subwordMaps_;
    u32                             size_;

    Lexicon::ConstLemmaPronunciationPtrList nonWordLemmaProns_;

protected:
    Time duration(const State* sp, State::const_iterator a) const;

public:
    LatticeAlignment(ConstLatticeRef l, AcousticPhonemeSequenceAligner* aligner, const LabelMapList& subwordMaps);
    virtual ~LatticeAlignment();

    void setNonWordLemmaPronunciations(const Lexicon::ConstLemmaPronunciationPtrList& nonWordLemmaProns);

    const AcousticPhonemeSequenceAligner* aligner() const;
    u32                                   size() const {
        return size_;
    }

    StateAlignment stateAlignment(const State* sp, State::const_iterator a) const;

    ConstSubWordAlignmentRef phonemeAlignment(const State* sp, State::const_iterator a) const;
    ConstSubWordAlignmentRef subwordAlignment(const State* sp, State::const_iterator a, u32 i = 0) const;

    ConstLatticeRef phonemeLattice() const;
    ConstLatticeRef subwordLattice(u32 i = 0) const;

    // AllophoneState frame posterior CN -> weighted alignment
    ConstPosteriorCnRef framePosteriorCn(ConstFwdBwdRef) const;
    ConstPosteriorCnRef phonemeFramePosteriorCn(ConstFwdBwdRef) const;
    ConstPosteriorCnRef subwordFramePosteriorCn(ConstFwdBwdRef, u32 i = 0) const;

    // Lemma frame posterior CN; scores are phoneme/subword posteriors
    // ATTENTION: resulting frame posterior CN is not normalized
    ConstPosteriorCnRef phonemeScoreLemmaPosteriorCn(ConstFwdBwdRef) const;
    // ConstPosteriorCnRef phonemeScoreLemmaPosteriorCn(ConstFwdBwdRef, u32 i = 0) const;
};
typedef Core::Ref<const LatticeAlignment> ConstLatticeAlignmentRef;

class LatticeAlignmentBuilder;
typedef Core::Ref<LatticeAlignmentBuilder> LatticeAlignmentBuilderRef;
class LatticeAlignmentBuilder : public Core::Component, public Core::ReferenceCounted {
private:
    AcousticPhonemeSequenceAligner*         aligner_;
    LabelMapList                            subwordMaps_;
    Lexicon::ConstLemmaPronunciationPtrList nonWordLemmaProns_;

public:
    LatticeAlignmentBuilder(const Core::Configuration& config, AcousticPhonemeSequenceAligner* aligner, const LabelMapList& subwordMaps);
    virtual ~LatticeAlignmentBuilder();

    ConstLatticeAlignmentRef build(ConstLatticeRef l, const Bliss::SpeechSegment* segment);

    static LatticeAlignmentBuilderRef create(const Core::Configuration& config, const LabelMapList& subwordMaps = LabelMapList(), bool computeEmissionAndTransitionScore = false);
};

ConstLatticeRef extendByAcousticScore(ConstLatticeRef l, ConstLatticeAlignmentRef latticeAlignment, ScoreId id,
                                      Score scale = 1.0, Score maxScore = Semiring::Max, bool scoreEpsArcs = false,
                                      RescoreMode rescoreMode = RescoreModeClone);

/*
 * Nodes using LatticeAlignment
 */
NodeRef createArcAlignmentNode(const std::string& name, const Core::Configuration& config);

NodeRef createExtendByAcousticScoreNode(const std::string& name, const Core::Configuration& config);

NodeRef createAllophoneStatePosteriorCnNode(const std::string& name, const Core::Configuration& config);

NodeRef createPhonemePosteriorFeatureNode(const std::string& name, const Core::Configuration& config);

/*
 * Align an orthography, aka a sequence of words,
 * and produce a lattice from it
 */
class AcousticOrthographyAligner;

class OrthographyAlignment : public Core::ReferenceCounted {
private:
    AcousticOrthographyAligner* aligner_;

public:
    OrthographyAlignment(AcousticOrthographyAligner* aligner);
    virtual ~OrthographyAlignment();

    const Speech::Alignment* stateAlignment() const;
    ConstLatticeRef          lattice() const;
};
typedef Core::Ref<const OrthographyAlignment> ConstOrthographyAlignmentRef;

class OrthographyAlignmentBuilder;
typedef Core::Ref<OrthographyAlignmentBuilder> OrthographyAlignmentBuilderRef;
class OrthographyAlignmentBuilder : public Core::Component, public Core::ReferenceCounted {
private:
    AcousticOrthographyAligner* aligner_;

public:
    OrthographyAlignmentBuilder(const Core::Configuration& config, AcousticOrthographyAligner* aligner);
    virtual ~OrthographyAlignmentBuilder();

    ConstOrthographyAlignmentRef build(const Bliss::SpeechSegment* segment);

    static OrthographyAlignmentBuilderRef create(const Core::Configuration& config);
};

NodeRef createOrthographyAlignmentNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf

#endif  // _FLF_ACOUSTIC_ALIGNMENT_HH
