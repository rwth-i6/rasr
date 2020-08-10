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
#ifndef _SPEECH_ADVANCED_ACCURACY_FSA_BUILDER_HH
#define _SPEECH_ADVANCED_ACCURACY_FSA_BUILDER_HH

#include "AccuracyFsaBuilder.hh"
#include "Confidences.hh"

namespace Lattice {
class SmoothingFunction;
}

namespace Speech {

class LevenshteinNBestListBuilder : public MetricFsaBuilder<Fsa::ConstAutomatonRef> {
    typedef MetricFsaBuilder<Fsa::ConstAutomatonRef> Precursor;

private:
    Bliss::Evaluator* evaluator_;

public:
    LevenshteinNBestListBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    ~LevenshteinNBestListBuilder();

    virtual Fsa::ConstAutomatonRef build(Fsa::ConstAutomatonRef);
    Functor                        createFunctor(const std::string&     id,
                                                 const std::string&     orth,
                                                 Fsa::ConstAutomatonRef list);
};

class OrthographyApproximatePhoneAccuracyMaskLatticeBuilder : public OrthographyTimeAlignmentBasedMetricLatticeBuilder {
    typedef OrthographyTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator_;
    ConfidenceArchive*                           confidenceArchive_;
    Confidences*                                 confidences_;

public:
    OrthographyApproximatePhoneAccuracyMaskLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~OrthographyApproximatePhoneAccuracyMaskLatticeBuilder();

    Functor                        createFunctor(const std::string&                           id,
                                                 const std::string&                           segmentId,
                                                 Lattice::ConstWordLatticeRef                 lattice,
                                                 Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

class OrthographyFrameStateAccuracyLatticeBuilder : public OrthographyTimeAlignmentBasedMetricLatticeBuilder {
    typedef OrthographyTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    Core::Ref<const Bliss::Lexicon>              lexicon_;
    Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator_;

public:
    OrthographyFrameStateAccuracyLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~OrthographyFrameStateAccuracyLatticeBuilder() {}

    Functor                        createFunctor(const std::string&                           id,
                                                 const std::string&                           segmentId,
                                                 Lattice::ConstWordLatticeRef                 lattice,
                                                 Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

class ArchiveFrameStateAccuracyLatticeBuilder : public ArchiveTimeAlignmentBasedMetricLatticeBuilder {
    typedef ArchiveTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator_;

public:
    ArchiveFrameStateAccuracyLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~ArchiveFrameStateAccuracyLatticeBuilder() {}

    Functor                        createFunctor(const std::string&                           id,
                                                 const std::string&                           segmentId,
                                                 Lattice::ConstWordLatticeRef                 lattice,
                                                 Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

class OrthographySmoothedFrameStateAccuracyLatticeBuilder : public OrthographyTimeAlignmentBasedMetricLatticeBuilder {
    typedef OrthographyTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator_;
    Lattice::SmoothingFunction*                  smoothing_;

public:
    OrthographySmoothedFrameStateAccuracyLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~OrthographySmoothedFrameStateAccuracyLatticeBuilder();

    Functor                        createFunctor(const std::string&                           id,
                                                 const std::string&                           segmentId,
                                                 Lattice::ConstWordLatticeRef                 lattice,
                                                 Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

/**
 * frame word accuracy
 */
class OrthographyFrameWordAccuracyLatticeBuilder : public OrthographyTimeAlignmentBasedMetricLatticeBuilder {
    typedef OrthographyTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    static const Core::ParameterFloat paramNormalization;

private:
    f32 normalization_;

public:
    OrthographyFrameWordAccuracyLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~OrthographyFrameWordAccuracyLatticeBuilder() {}

    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

/**
 * frame phone accuracy
 */
class OrthographyFramePhoneAccuracyLatticeBuilder : public OrthographyTimeAlignmentBasedMetricLatticeBuilder {
    typedef OrthographyTimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    static const Core::ParameterFloat paramNormalization;

private:
    f32                                          normalization_;
    Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator_;

public:
    OrthographyFramePhoneAccuracyLatticeBuilder(const Core::Configuration&, Core::Ref<const Bliss::Lexicon>);
    virtual ~OrthographyFramePhoneAccuracyLatticeBuilder() {}

    Functor                        createFunctor(const std::string&                           id,
                                                 const std::string&                           segmentId,
                                                 Lattice::ConstWordLatticeRef                 lattice,
                                                 Core::Ref<PhonemeSequenceAlignmentGenerator> alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

/**
 * frame phone accuracy
 */
class FramePhoneAccuracyLatticeBuilder : public TimeAlignmentBasedMetricLatticeBuilder {
    typedef TimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    static const Core::ParameterFloat paramNormalization;

private:
    f32                   normalization_;
    AlignmentGeneratorRef alignmentGenerator_;

public:
    FramePhoneAccuracyLatticeBuilder(const Core::Configuration&, Bliss::LexiconRef);

    Functor                        createFunctor(const std::string&           id,
                                                 Lattice::ConstWordLatticeRef reference,
                                                 Lattice::ConstWordLatticeRef lattice,
                                                 AlignmentGeneratorRef        alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

/**
 * soft frame phone accuracy
 */
class SoftFramePhoneAccuracyLatticeBuilder : public TimeAlignmentBasedMetricLatticeBuilder {
    typedef TimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    AlignmentGeneratorRef alignmentGenerator_;
    const Alignment*      forcedAlignment_;

protected:
    void setReference(const Alignment* forcedAlignment);

public:
    SoftFramePhoneAccuracyLatticeBuilder(const Core::Configuration&, Bliss::LexiconRef);

    Functor                        createFunctor(const std::string&           id,
                                                 Lattice::ConstWordLatticeRef reference,
                                                 Lattice::ConstWordLatticeRef lattice,
                                                 AlignmentGeneratorRef        alignmentGenerator);
    Functor                        createFunctor(const std::string&           id,
                                                 const Alignment*             forcedAlignment,
                                                 Lattice::ConstWordLatticeRef lattice,
                                                 AlignmentGeneratorRef        alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

/**
 * weighted frame phone accuracy
 */
class WeightedFramePhoneAccuracyLatticeBuilder : public TimeAlignmentBasedMetricLatticeBuilder {
    typedef TimeAlignmentBasedMetricLatticeBuilder Precursor;

private:
    static const Core::ParameterFloat paramBeta;
    static const Core::ParameterFloat paramMargin;

private:
    f32                   beta_;
    f32                   margin_;
    AlignmentGeneratorRef alignmentGenerator_;

public:
    WeightedFramePhoneAccuracyLatticeBuilder(const Core::Configuration&, Bliss::LexiconRef);

    Functor                        createFunctor(const std::string&           id,
                                                 Lattice::ConstWordLatticeRef reference,
                                                 Lattice::ConstWordLatticeRef lattice,
                                                 AlignmentGeneratorRef        alignmentGenerator);
    virtual Fsa::ConstAutomatonRef build(Lattice::ConstWordLatticeRef);
};

}  // namespace Speech

#endif  // _SPEECH_ADVANCED_ACCURACY_FSA_BUILDER_HH
