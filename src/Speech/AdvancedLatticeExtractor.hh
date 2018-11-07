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
#ifndef _SPEECH_ADVANCED_LATTICE_EXTRACTOR_HH
#define _SPEECH_ADVANCED_LATTICE_EXTRACTOR_HH

#include "LatticeExtractor.hh"

namespace Search {
    class WordConditionedTreeSearch;
}


namespace Speech {

    class OrthographyApproximatePhoneAccuracyMaskLatticeBuilder;
    class OrthographyFrameStateAccuracyLatticeBuilder;
    class ArchiveFrameStateAccuracyLatticeBuilder;
    class OrthographyFrameWordAccuracyLatticeBuilder;
    class OrthographyFramePhoneAccuracyLatticeBuilder;
    class OrthographySmoothedFrameStateAccuracyLatticeBuilder;
    class LevenshteinNBestListBuilder;

    /*
     * LatticeRescorer: emission
     */
    class EmissionLatticeRescorer : public virtual AcousticLatticeRescorer
    {
        typedef AcousticLatticeRescorer Precursor;
    private:
        static const Core::ParameterString paramPortName;
        static const Core::ParameterString paramSparsePortName;
    protected:
        Core::Ref<SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor_;
        Flow::PortId portId_;
        Flow::PortId sparsePortId_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        EmissionLatticeRescorer(const Core::Configuration &,
                                bool initialize = true);
        EmissionLatticeRescorer(const Core::Configuration &, Core::Ref<Am::AcousticModel>);
        virtual ~EmissionLatticeRescorer() {}

        void setSegmentwiseFeatureExtractor(Core::Ref<SegmentwiseFeatureExtractor>);
        void setFeatures(const std::vector<Core::Ref<Feature> > &);
    };

    /*
     * LatticeRescorer: tdp
     */
    class TdpLatticeRescorer : public virtual AcousticLatticeRescorer
    {
        typedef AcousticLatticeRescorer Precursor;
    protected:
        static const Core::ParameterStringVector paramSilencesAndNoises;
    protected:
        AllophoneStateGraphBuilder *allophoneStateGraphBuilder_;
    public:
        TdpLatticeRescorer(const Core::Configuration &,
                           bool initialize = true);
        virtual ~TdpLatticeRescorer();

        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    };

    /*
     * LatticeRescorer: combined acoustic model
     */
    class CombinedAcousticLatticeRescorer :
        public EmissionLatticeRescorer, public TdpLatticeRescorer
    {
        typedef AcousticLatticeRescorer Precursor;
    private:
        static const Core::ParameterBool paramShouldSumOverPronunciations;
    private:
        bool shouldSumOverPronunciations_;
    public:
        CombinedAcousticLatticeRescorer(const Core::Configuration &);
        CombinedAcousticLatticeRescorer(const Core::Configuration &, Core::Ref<Am::AcousticModel>);
        virtual ~CombinedAcousticLatticeRescorer() {}

        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    };

    /*
     * LatticeGenerator: pronunciation rescoring
     */
    class PronunciationLatticeRescorer : public LatticeRescorer
    {
        typedef LatticeRescorer Precursor;
    private:
        f32 pronunciationScale_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        PronunciationLatticeRescorer(const Core::Configuration &);
        virtual ~PronunciationLatticeRescorer() {}
    };

    /*
     * RestoreScoresLatticeRescorer
     */
    class RestoreScoresLatticeRescorer : public LatticeRescorer
    {
        typedef LatticeRescorer Precursor;
    private:
        static const Core::ParameterString paramFsaPrefix;
    private:
        Lattice::ArchiveReader *archiveReader_;
    private:
        /**
         *  Prefix distinguishing different lattices in one lattice archive.
         *  If paramFsaPrefix is not given, configuration name of this object
         *  is used.
         */
        std::string fsaPrefix_;
        Core::Ref<const Lm::ScaledLanguageModel> languageModel_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        RestoreScoresLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~RestoreScoresLatticeRescorer() {}
    };

    class OrthographyApproximatePhoneAccuracyMaskLatticeRescorer :
        public ApproximatePhoneAccuracyLatticeRescorer
    {
        typedef ApproximatePhoneAccuracyLatticeRescorer Precursor;
    private:
        OrthographyApproximatePhoneAccuracyMaskLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        OrthographyApproximatePhoneAccuracyMaskLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~OrthographyApproximatePhoneAccuracyMaskLatticeRescorer();
    };

    /*
     * LatticeRescorer: frame state accuracy
     */
    class FrameStateAccuracyLatticeRescorer :
        public ApproximateDistanceLatticeRescorer
    {
        typedef ApproximateDistanceLatticeRescorer Precursor;
        typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;
    protected:
        AlignmentGeneratorRef alignmentGenerator_;
    public:
        FrameStateAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~FrameStateAccuracyLatticeRescorer();

        void setAlignmentGenerator(AlignmentGeneratorRef alignmentGenerator) {
            alignmentGenerator_ = alignmentGenerator;
        }
    };

    class ArchiveFrameStateAccuracyLatticeRescorer :
        public FrameStateAccuracyLatticeRescorer
    {
        typedef FrameStateAccuracyLatticeRescorer Precursor;
    private:
        ArchiveFrameStateAccuracyLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        ArchiveFrameStateAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~ArchiveFrameStateAccuracyLatticeRescorer();
    };

    class OrthographyFrameStateAccuracyLatticeRescorer :
        public FrameStateAccuracyLatticeRescorer
    {
        typedef FrameStateAccuracyLatticeRescorer Precursor;
    private:
        OrthographyFrameStateAccuracyLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        OrthographyFrameStateAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~OrthographyFrameStateAccuracyLatticeRescorer();
    };

    /*
     * OrthographySmoothedFrameStateAccuracyLatticeRescorer
     * Used for state-based training criterion with smoothing
     * function f(x) of the type \sum_{t}f(E[\chi_{spk,t}]).
     * @return is the rescored lattice has arc weights
     * \sum_{t}f'(E[\chi_{spk,t}])\chi_{spk,t} and is used
     * in MinimumErrorSegmentwise[Gmm/Me]Trainer as accuracy
     * lattice to calculate the gradient,
     * Cov(@return, \nabla\log p).
     */
    class OrthographySmoothedFrameStateAccuracyLatticeRescorer :
        public FrameStateAccuracyLatticeRescorer
    {
        typedef FrameStateAccuracyLatticeRescorer Precursor;
    private:
        OrthographySmoothedFrameStateAccuracyLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        OrthographySmoothedFrameStateAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~OrthographySmoothedFrameStateAccuracyLatticeRescorer();
    };

    /*
     * LatticeRescorer: word accuracy
     */
    class WordAccuracyLatticeRescorer : public DistanceLatticeRescorer
    {
        typedef DistanceLatticeRescorer Precursor;
    private:
        Fsa::ConstAutomatonRef lemmaPronToLemma_;
        Fsa::ConstAutomatonRef lemmaToEval_;
        Bliss::OrthographicParser *orthToLemma_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        WordAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~WordAccuracyLatticeRescorer();
    };

    /*
     * LatticeRescorer: phoneme accuracy
     */
    class PhonemeAccuracyLatticeRescorer : public DistanceLatticeRescorer
    {
        typedef DistanceLatticeRescorer Precursor;
    private:
        Fsa::ConstAutomatonRef lemmaPronToPhoneme_;
        Fsa::ConstAutomatonRef lemmaToPhoneme_;
        Bliss::OrthographicParser *orthToLemma_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        PhonemeAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~PhonemeAccuracyLatticeRescorer();
    };

    /*
     * LatticeRescorer: levenshtein distance (on n-best lists)
     */
    class LevenshteinListRescorer : public DistanceLatticeRescorer
    {
        typedef DistanceLatticeRescorer Precursor;
    private:
        LevenshteinNBestListBuilder *builder_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        LevenshteinListRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~LevenshteinListRescorer();
    };

    class OrthographyFrameWordAccuracyLatticeRescorer :
        public ApproximateDistanceLatticeRescorer
    {
        typedef ApproximateDistanceLatticeRescorer Precursor;
    private:
        OrthographyFrameWordAccuracyLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        OrthographyFrameWordAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~OrthographyFrameWordAccuracyLatticeRescorer();
    };

    class OrthographyFramePhoneAccuracyLatticeRescorer :
        public FrameStateAccuracyLatticeRescorer
    {
        typedef FrameStateAccuracyLatticeRescorer Precursor;
    private:
        OrthographyFramePhoneAccuracyLatticeBuilder *builder_;
    protected:
        virtual Fsa::ConstAutomatonRef getDistanceFsa(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        OrthographyFramePhoneAccuracyLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~OrthographyFramePhoneAccuracyLatticeRescorer();
    };

    /*
     * LatticeRescorer: posterior
     */
    class PosteriorLatticeRescorer : public LatticeRescorer
    {
        typedef LatticeRescorer Precursor;
    public:
        enum PosteriorType {
            probability,
            expectation,
            combinedProbability
        };
        static Core::Choice choicePosteriorType;
        static Core::ParameterChoice paramPosteriorType;
    private:
        static const Core::ParameterInt paramTolerance;
        static const Core::ParameterBool paramPNormalized;
    protected:
        s32 tolerance_;
        bool pNormalized_;
        Mm::Sum accumulator_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
        void accumulate(f32 toAcc);
    public:
        PosteriorLatticeRescorer(const Core::Configuration &);
        virtual ~PosteriorLatticeRescorer();

        static LatticeRescorer* createPosteriorLatticeRescorer(
            const Core::Configuration &, Bliss::LexiconRef);
    };

    /*
     * PosteriorLatticeRescorer: expectation
     */
    class ExpectationPosteriorLatticeRescorer : public PosteriorLatticeRescorer
    {
        typedef PosteriorLatticeRescorer Precursor;
    private:
        static const Core::ParameterBool paramVNormalized;
    protected:
        bool vNormalized_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        ExpectationPosteriorLatticeRescorer(const Core::Configuration &);
        virtual ~ExpectationPosteriorLatticeRescorer() {}
    };

    /*
     * LatticeRescorer: combined probability
     */
    class CombinedPosteriorLatticeRescorer : public PosteriorLatticeRescorer
    {
        typedef PosteriorLatticeRescorer Precursor;
    private:
        Lattice::ArchiveReader *archiveToCombine_;
    protected:
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    public:
        CombinedPosteriorLatticeRescorer(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~CombinedPosteriorLatticeRescorer();
    };

    /*
     * RecognizerWithConstrainedLanguageModel
     * Applications: - full search acoustic rescoring
     *               - numerator lattice generation
     */
    class RecognizerWithConstrainedLanguageModel : public AcousticLatticeRescorerBase
    {
        typedef AcousticLatticeRescorerBase Precursor;
    private:
        static const Core::ParameterString paramPortName;
    protected:
        Core::Ref<SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor_;
        Flow::PortId portId_;
        Search::WordConditionedTreeSearch *recognizer_;
        Fsa::ConstAutomatonRef lemmaPronunciationToLemmaTransducer_;
        Fsa::ConstAutomatonRef lemmaToSyntacticTokenTransducer_;
    private:
        void setGrammar(Fsa::ConstAutomatonRef g);
        void feed();
    public:
        RecognizerWithConstrainedLanguageModel(const Core::Configuration &, Bliss::LexiconRef);
        virtual ~RecognizerWithConstrainedLanguageModel();

        virtual Lattice::ConstWordLatticeRef extract(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
        virtual Lattice::ConstWordLatticeRef work(
            Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *) { return Lattice::ConstWordLatticeRef(); };

        void setSegmentwiseFeatureExtractor(Core::Ref<SegmentwiseFeatureExtractor>);
    };

}

#endif // _SPEECH_ADVANCED_LATTICE_EXTRACTOR_HH
