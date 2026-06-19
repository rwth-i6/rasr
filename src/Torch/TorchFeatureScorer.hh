/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef _TORCH_FEATURESCORER_HH
#define _TORCH_FEATURESCORER_HH

#include <deque>

#include <Core/Types.hh>
#include <Mm/Feature.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/MixtureSet.hh>
#include <Mm/Types.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/Prior.hh>
#include <Nn/Types.hh>

#include "Model.hh"
#include "StateManager.hh"

namespace Torch {

/*
 * Feature scorer implementation based on an exported Torch model
 */
class TorchFeatureScorer : public Mm::FeatureScorer {
public:
    using Precursor = Mm::FeatureScorer;
    using Scorer    = Core::Ref<const ContextScorer>;

    static const Core::ParameterBool paramApplyLogOnOutput;
    static const Core::ParameterBool paramNegateOutput;
    static const Core::ParameterBool paramUseOutputAsIs;
    static const Core::ParameterInt  paramFeatureDimension;
    static const Core::ParameterInt  paramOutputDimension;

    TorchFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet);
    virtual ~TorchFeatureScorer() = default;

    virtual Mm::EmissionIndex nMixtures() const;
    virtual void              getFeatureDescription(Mm::FeatureDescription& description) const;

    // Appends the given feature to the internal buffer and returns a scorer for the current buffered position
    virtual FeatureScorer::Scorer getScorer(Core::Ref<const Mm::Feature> f) const;
    virtual FeatureScorer::Scorer getScorer(const Mm::FeatureVector& f) const;

    // Returns the score for one emission at one output frame
    virtual Mm::Score getScore(Mm::EmissionIndex e, u32 position) const;

    // Resets buffered runtime state for a new segment
    virtual void reset() const;

    virtual void finalize() const;

    virtual bool isBuffered() const;

    virtual void addFeature(const Mm::FeatureVector& f) const;
    virtual void addFeature(Core::Ref<const Mm::Feature> f) const;

    // Forces score computation for the currently buffered features
    virtual Mm::FeatureScorer::Scorer flush() const;

    // Returns true if the feature buffer is full
    virtual bool bufferFilled() const;
    // Returns true if the feature buffer is empty
    virtual bool bufferEmpty() const;
    // Return the number of buffered features required to execute getScorer()
    virtual u32 bufferSize() const;

    // Like CachedNeuralNetworkFeatureScorer, used in SegmentwiseAlignmentGenerator
    virtual bool hasTimeIndexedCache() const;

    virtual FeatureScorer::Scorer getTimeIndexedScorer(u32 time) const;

protected:
    using Float = Mm::Score;
    class ContextScorer;

    const bool applyLogOnOutput_;
    const bool negateOutput_;
    const bool useOutputAsIs_;

    // Optional prior handling and label mapping
    Nn::Prior<Float>                       prior_;
    std::unique_ptr<Nn::ClassLabelWrapper> labelWrapper_;
    int64_t                                expectedFeatureDim_;
    int64_t                                expectedOutputDim_;

    // Buffered input features and cached model scores for the current chunk
    mutable std::deque<Mm::FeatureVector>    inputBuffer_;
    mutable u32                              currentFeature_;
    mutable u32                              batchIteration_;
    mutable bool                             scoresComputed_;
    std::unique_ptr<Math::FastMatrix<Float>> scores_;

    // Torch model runtime and optional state handling
    Model                           torchModel_;
    std::unique_ptr<StateManager>   stateManager_;
    std::vector<TorchStateVariable> stateVariables_;

    void addFeatureInternal(const Mm::FeatureVector& f) const;
    void computeScoresInternal();

private:
    void initializeStatesFromModelSpec();
};

// inline implementations

inline TorchFeatureScorer::FeatureScorer::Scorer TorchFeatureScorer::getScorer(Core::Ref<const Mm::Feature> f) const {
    return getScorer(*f->mainStream());
}

inline bool TorchFeatureScorer::isBuffered() const {
    return true;
}

inline void TorchFeatureScorer::addFeature(Core::Ref<const Mm::Feature> f) const {
    addFeature(*f->mainStream());
}

inline bool TorchFeatureScorer::bufferFilled() const {
    return inputBuffer_.size() >= bufferSize();
}

inline bool TorchFeatureScorer::bufferEmpty() const {
    return scoresComputed_ && currentFeature_ >= scores_->nRows();
}

inline u32 TorchFeatureScorer::bufferSize() const {
    return std::numeric_limits<u32>::max();
}

inline bool TorchFeatureScorer::hasTimeIndexedCache() const {
    return true;
}

}  // namespace Torch

#endif  // _TORCH_FEATURESCORER_HH