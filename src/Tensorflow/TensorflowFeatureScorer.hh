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
#ifndef NN_TENSORFLOWFEATURESCORER_HH
#define NN_TENSORFLOWFEATURESCORER_HH

#include <Core/Types.hh>
#include <Mm/Feature.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/MixtureSet.hh>
#include <Mm/Types.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/Prior.hh>
#include <Nn/Types.hh>
#include <deque>

#include "GraphLoader.hh"
#include "Session.hh"
#include "StateManager.hh"
#include "Tensor.hh"
#include "TensorMap.hh"

namespace Tensorflow {

// We use a TF graph to calculate posterior scores and wrap those in a Mm::FeatureScorer.
// Otherwise, it's quite similar to the Nn::BatchFeatureScorer.
class TensorflowFeatureScorer : public Mm::FeatureScorer {
protected:
    using Float = Mm::Score;
    class ContextScorer;

    u32        nClasses_;
    const u32  inputDimension_;
    const bool useOutputAsIs_;
    const bool returnScoresInNegLog_;
    const bool asyncInitialization_;

    Nn::Prior<Float>                       prior_;
    mutable u32                            currentFeature_; /* pointer to current position in buffer */
    mutable std::deque<Mm::FeatureVector>  buffer_;
    mutable bool                           scoresComputed_;
    mutable u32                            batchIteration_;
    std::unique_ptr<Nn::ClassLabelWrapper> labelWrapper_;
    std::atomic<bool>                      tensorflowInitialized_;

    void         _addFeature(const Mm::FeatureVector& f) const;
    Tensor       _createInputTensor() const;
    virtual void _compute();
    void         initializeTensorflow();
    void         waitForInitialization() const;

    // tensorflow related members
    std::vector<std::string>                 output_tensor_names_;
    std::vector<std::string>                 target_tensor_names_;
    mutable Session                          session_;
    std::unique_ptr<GraphLoader>             loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;
    TensorInputMap                           tensor_input_map_;
    TensorOutputMap                          tensor_output_map_;
    std::unique_ptr<StateManager>            state_manager_;
    std::unique_ptr<Math::FastMatrix<Float>> scores_;

public:
    TensorflowFeatureScorer(const Core::Configuration& c, Core::Ref<const Mm::MixtureSet> mixtureSet);
    virtual ~TensorflowFeatureScorer() = default;

    virtual Mm::EmissionIndex nMixtures() const;
    virtual void              getFeatureDescription(Mm::FeatureDescription& description) const;

    typedef Core::Ref<const ContextScorer> Scorer;
    /**
     * Return a scorer for the current feature and append the
     * given feature to the buffer.
     * The current feature may not be the same as f
     * because of the feature buffering.
     * Requires bufferFilled() == true.
     */
    virtual FeatureScorer::Scorer getScorer(Core::Ref<const Mm::Feature> f) const {
        return getScorer(*f->mainStream());
    }
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
    virtual bool isBuffered() const {
        return true;
    }

    /**
     * Add a feature to the feature buffer.
     */
    virtual void addFeature(const Mm::FeatureVector& f) const;
    virtual void addFeature(Core::Ref<const Mm::Feature> f) const {
        addFeature(*f->mainStream());
    }

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
    virtual bool hasTimeIndexedCache() const {
        return true;
    }
    virtual FeatureScorer::Scorer getTimeIndexedScorer(u32 time) const;
};
}  // namespace Tensorflow

#endif  // NN_TENSORFLOWFEATURESCORER_HH
