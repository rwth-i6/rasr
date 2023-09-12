/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_ONNXFEATURESCORER_HH
#define _ONNX_ONNXFEATURESCORER_HH

#include <deque>

#include <Core/Types.hh>
#include <Mm/Feature.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/MixtureSet.hh>
#include <Mm/Types.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/Prior.hh>
#include <Nn/Types.hh>

#include "IOSpecification.hh"
#include "Session.hh"

namespace Onnx {

class OnnxFeatureScorer : public Mm::FeatureScorer {
public:
    using Precursor = Mm::FeatureScorer;
    using Scorer    = Core::Ref<const ContextScorer>;

    static const Core::ParameterBool paramApplyLogOnOutput;
    static const Core::ParameterBool paramNegateOutput;

    OnnxFeatureScorer(const Core::Configuration& c, Core::Ref<const Mm::MixtureSet> mixtureSet);
    virtual ~OnnxFeatureScorer() = default;

    virtual Mm::EmissionIndex nMixtures() const;
    virtual void              getFeatureDescription(Mm::FeatureDescription& description) const;

    /**
     * Return a scorer for the current feature and append the
     * given feature to the buffer.
     * The current feature may not be the same as f
     * because of the feature buffering.
     * Requires bufferFilled() == true.
     */
    virtual FeatureScorer::Scorer getScorer(Core::Ref<const Mm::Feature> f) const;
    virtual FeatureScorer::Scorer getScorer(const Mm::FeatureVector& f) const;

    virtual Mm::Score getScore(Mm::EmissionIndex e, u32 position) const;

    /**
     * reset should be overloaded/defined in/for
     * featurescorer related to sign language recognition
     * especially the tracking part
     *
     */
    virtual void reset() const;

    /**
     * finalize should be overloaded/defined in classes using
     * embedded flow networks to sent final end of sequence token
     * if necessary
     */
    virtual void finalize() const;

    /**
     * Return true if the feature scorer buffers features.
     */
    virtual bool isBuffered() const;

    /**
     * Add a feature to the feature buffer.
     */
    virtual void addFeature(const Mm::FeatureVector& f) const;
    virtual void addFeature(Core::Ref<const Mm::Feature> f) const;

    /**
     * Return a scorer for the current feature without adding a
     * new feature to the buffer.
     * Should be called until bufferEmpty() == true.
     * Requires bufferEmpty() == false.
     * Implementation required if isBuffered() == true
     */
    virtual Mm::FeatureScorer::Scorer flush() const;

    /**
     * Return true if the feature buffer is full.
     */
    virtual bool bufferFilled() const;

    /**
     * Return true if the feature buffer is empty.
     */
    virtual bool bufferEmpty() const;

    /**
     * Return the number of buffered features required to
     * execute getScorer().
     * This will be MAX_INT for this class, because there is no limit.
     * Normally, you would just use bufferFilled()/bufferEmpy() instead.
     */
    virtual u32 bufferSize() const;

    /**
     * Like CachedNeuralNetworkFeatureScorer, used in SegmentwiseAlignmentGenerator.
     */
    virtual bool hasTimeIndexedCache() const;

    virtual FeatureScorer::Scorer getTimeIndexedScorer(u32 time) const;

protected:
    using Float = Mm::Score;
    class ContextScorer;

    const bool applyLogOnOutput_;
    const bool negateOutput_;

    Nn::Prior<Float>                       prior_;
    std::unique_ptr<Nn::ClassLabelWrapper> labelWrapper_;
    int64_t                                expectedFeatureDim_;
    int64_t                                expectedOutputDim_;

    mutable std::deque<Mm::FeatureVector>    inputBuffer_;
    mutable u32                              currentFeature_;
    mutable u32                              batchIteration_;
    mutable bool                             scoresComputed_;
    std::unique_ptr<Math::FastMatrix<Float>> scores_;

    // ONNX related members
    mutable Session                    session_;
    std::vector<Onnx::IOSpecification> ioSpec_;
    IOMapping                          mapping_;
    IOValidator                        validator_;

    void      addFeatureInternal(const Mm::FeatureVector& f) const;
    Value     createInputValue() const;
    Mm::Score getScoreFromOutput(Mm::EmissionIndex e, u32 position) const;
};

// inline implementations

inline OnnxFeatureScorer::FeatureScorer::Scorer OnnxFeatureScorer::getScorer(Core::Ref<const Mm::Feature> f) const {
    return getScorer(*f->mainStream());
}

inline bool OnnxFeatureScorer::isBuffered() const {
    return true;
}

inline void OnnxFeatureScorer::addFeature(Core::Ref<const Mm::Feature> f) const {
    addFeature(*f->mainStream());
}

inline bool OnnxFeatureScorer::bufferFilled() const {
    return inputBuffer_.size() >= bufferSize();
}

inline bool OnnxFeatureScorer::bufferEmpty() const {
    return currentFeature_ >= inputBuffer_.size();
}  // == cannot call flush() anymore

inline u32 OnnxFeatureScorer::bufferSize() const {
    return std::numeric_limits<u32>::max();
}

inline bool OnnxFeatureScorer::hasTimeIndexedCache() const {
    return true;
}

}  // namespace Onnx

#endif  // _ONNX_ONNXFEATURESCORER_HH
