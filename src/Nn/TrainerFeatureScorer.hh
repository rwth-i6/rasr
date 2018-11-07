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
#ifndef NN_TRAINERFEATURESCORER_HH
#define NN_TRAINERFEATURESCORER_HH

#include <Mm/FeatureScorer.hh>
#include <Mm/Types.hh>
#include <Mm/Feature.hh>
#include <Mm/MixtureSet.hh>
#include <Nn/Prior.hh>
#include <Nn/Types.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/NeuralNetworkTrainer.hh>
#include <Core/Types.hh>
#include <deque>

namespace Nn {

// We use a NN trainer to calculate posterior scores and wrap those in a Mm::FeatureScorer.
// Otherwise, it's quite similar to the Nn::BatchFeatureScorer.
class TrainerFeatureScorer : public Mm::FeatureScorer {
protected:
    typedef Mm::FeatureScorer Precursor;
    typedef Mm::Score Float;

    class ContextScorer;

    Prior<Float> prior_;
    mutable u32 currentFeature_;		/* pointer to current position in buffer */
    mutable std::deque<Mm::FeatureVector> buffer_;
    mutable bool scoresComputed_;
    bool returnScoresInNegLog_;
    u32 nClasses_;
    u32 inputDimension_;
    mutable u32 batchIteration_;
    ClassLabelWrapper* labelWrapper_;
    mutable NeuralNetworkTrainer<Float>* trainer_;

public:
    TrainerFeatureScorer(const Core::Configuration &c, Core::Ref<const Mm::MixtureSet> mixtureSet);
    virtual ~TrainerFeatureScorer();
private:
    void _addFeature(const Mm::FeatureVector &f) const;
public:
    virtual Mm::EmissionIndex nMixtures() const;
    virtual void getFeatureDescription(Mm::FeatureDescription &description) const;

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
    virtual FeatureScorer::Scorer getScorer(const Mm::FeatureVector &f) const;

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
    virtual void finalize() const {}

    /**
     * Return true if the feature scorer buffers features.
     */
    virtual bool isBuffered() const { return true; }

    /**
     * Add a feature to the feature buffer.
     */
    virtual void addFeature(const Mm::FeatureVector &f) const;
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
    virtual bool hasTimeIndexedCache() const { return true; }
    virtual FeatureScorer::Scorer getTimeIndexedScorer(u32 time) const;

};

}

#endif // TRAINERFEATURESCORER_HH
