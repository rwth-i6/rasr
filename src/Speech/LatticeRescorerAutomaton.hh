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
#ifndef _SPEECH_LATTICE_RESCORER_AUTOMTA_HH
#define _SPEECH_LATTICE_RESCORER_AUTOMTA_HH

#include <Lattice/Lattice.hh>
#include "PhonemeSequenceAlignmentGenerator.hh"

namespace Speech {

/*
 * LatticeRescorerAutomaton: base class
 */
class LatticeRescorerAutomaton : public Lattice::ModifyWordLattice {
    typedef Lattice::ModifyWordLattice Precursor;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const = 0;

public:
    LatticeRescorerAutomaton(Lattice::ConstWordLatticeRef);
    virtual void modifyState(Fsa::State* sp) const;
};

/*
 * LatticeRescorerAutomaton: with cache
 */
class CachedLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
    typedef LatticeRescorerAutomaton Precursor;
    struct Key {
        std::string str;

        Key(Fsa::LabelId                 input,
            const Lattice::WordBoundary& wbl,
            const Lattice::WordBoundary& wbr) {
            str = Core::form("%d|%d|%d|%d|%d",
                             input,
                             wbl.time(),
                             wbr.time(),
                             wbl.transit().final,
                             wbr.transit().initial);
        }
    };
    typedef Core::StringHashMap<Fsa::Weight> Scores;

private:
    mutable Scores cache_;

public:
    CachedLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef);
    virtual void modifyState(Fsa::State* sp) const;
};

/*
 * EmissionLatticeRescorerAutomaton
 */
class EmissionLatticeRescorerAutomaton : public CachedLatticeRescorerAutomaton {
    typedef CachedLatticeRescorerAutomaton               Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

private:
    AlignmentGeneratorRef        alignmentGenerator_;
    ConstSegmentwiseFeaturesRef  features_;
    Core::Ref<Am::AcousticModel> acousticModel_;

protected:
    // returns emission score of arc a, outgoing from state s
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    EmissionLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef,
                                     AlignmentGeneratorRef, ConstSegmentwiseFeaturesRef,
                                     Core::Ref<Am::AcousticModel>);
    virtual std::string describe() const {
        return Core::form("emission-rescore(%s)", fsa_->describe().c_str());
    }
    // returns emission score of coarticulatedPronunciation
    // alignment is read from cache or generated on demand
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

/*
 * TdpLatticeRescorerAutomaton
 */
class TdpLatticeRescorerAutomaton : public CachedLatticeRescorerAutomaton {
    typedef CachedLatticeRescorerAutomaton               Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

private:
    AlignmentGeneratorRef                    alignmentGenerator_;
    AllophoneStateGraphBuilder*              allophoneStateGraphBuilder_;
    Core::Ref<const Am::AcousticModel>       acousticModel_;
    const Bliss::LemmaPronunciationAlphabet* alphabet_;

private:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    TdpLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, AlignmentGeneratorRef,
                                AllophoneStateGraphBuilder*,
                                Core::Ref<const Am::AcousticModel>);
    virtual std::string describe() const {
        return Core::form("tdp-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

/*
 * CombinedAcousticLatticeRescorerAutomaton
 */
class CombinedAcousticLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
    typedef LatticeRescorerAutomaton                     Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

protected:
    Core::Ref<const EmissionLatticeRescorerAutomaton> emissionRescorer_;
    Core::Ref<const TdpLatticeRescorerAutomaton>      tdpRescorer_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    CombinedAcousticLatticeRescorerAutomaton(
            Lattice::ConstWordLatticeRef, AlignmentGeneratorRef,
            Core::Ref<Am::AcousticModel>,
            ConstSegmentwiseFeaturesRef, AllophoneStateGraphBuilder*);
    virtual std::string describe() const {
        return Core::form("combined-acoustic-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

/*
 * CombinedAcousticSummedPronunciationLatticeRescorerAutomaton
 */
class CombinedAcousticSummedPronunciationLatticeRescorerAutomaton : public CombinedAcousticLatticeRescorerAutomaton {
    typedef CombinedAcousticLatticeRescorerAutomaton     Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    CombinedAcousticSummedPronunciationLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, AlignmentGeneratorRef,
                                                                Core::Ref<Am::AcousticModel>,
                                                                ConstSegmentwiseFeaturesRef, AllophoneStateGraphBuilder*);
    virtual std::string describe() const {
        return Core::form("combined-acoustic-summed-pronunciation-rescore(%s)", fsa_->describe().c_str());
    }
};

class AlignmentLatticeRescorerAutomaton : public CachedLatticeRescorerAutomaton {
    typedef CachedLatticeRescorerAutomaton               Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

private:
    AlignmentGeneratorRef alignmentGenerator_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    AlignmentLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, AlignmentGeneratorRef);
    virtual std::string describe() const {
        return Core::form("acoustic-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

/*
 * PronunciationLatticeRescorerAutomaton
 */
class PronunciationLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
private:
    const Bliss::LemmaPronunciationAlphabet* alphabet_;
    f32                                      pronunciationScale_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    PronunciationLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, f32);
    virtual std::string describe() const {
        return Core::form("pronunciation-rescore(%s)", fsa_->describe().c_str());
    }
};

/*
 * LmLatticeRescorerAutomaton
 */
class LmLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
    typedef Core::Ref<const Lm::ScaledLanguageModel> ConstLanguageModelRef;
    typedef Lm::History                              History;
    typedef Core::Vector<History>                    Histories;

private:
    ConstLanguageModelRef                    languageModel_;
    mutable Histories                        histories_;
    const Bliss::LemmaPronunciationAlphabet* alphabet_;
    f32                                      pronunciationScale_;

private:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    LmLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef,
                               Core::Ref<const Lm::ScaledLanguageModel>,
                               f32 pronunciationScale = 0);
    virtual std::string describe() const {
        return Core::form("lm-rescore(%s)", fsa_->describe().c_str());
    }
};

}  // namespace Speech

#endif  // _SPEECH_LATTICE_RESCORER_AUTOMTA_HH
