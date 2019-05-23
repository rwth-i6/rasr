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
#include "TrainerFeatureScorer.hh"

/*
 * TrainerFeatureScorer uses a NeuralNetworkTrainer to extract feature scores (getClassLabelPosteriors()).
 * That way we can e.g. use the PythonTrainer to use CRNN so we can use an RNN to get the scores.
 *
 * For an RNN, esp a bidirectional one, we need to first get all the features before we can
 * forward it and get the scores. That is important for the design of this class.
 *
 * Also, this class supports buffering (isBuffered() == true), so both the
 * addFeature()/flush() but also the (traditionally non-buffered) getScorer()
 * has to be supported.
 * getScorer() is used e.g. in the aligner.
 * addFeature()/flush() is used by the recognizer.
 *
 */

namespace Nn {

static const Core::ParameterInt paramFeatureDimension(
        "feature-dimension", "feature = input dimension");

static const Core::ParameterInt paramOutputDimension(
        "trainer-feature-scorer-output-dimension", "if set, will ignore the number of mixtures", -1);

static const Core::ParameterBool paramReturnScoresInNegLog(
        "return-scores-in-neg-log", "return scores in -log space (default)", true);

TrainerFeatureScorer::TrainerFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          Mm::FeatureScorer(config),
          prior_(config),
          currentFeature_(0),
          scoresComputed_(false),
          returnScoresInNegLog_(paramReturnScoresInNegLog(config)),
          nClasses_(mixtureSet->nMixtures()),
          inputDimension_(paramFeatureDimension(config)),
          batchIteration_(0),
          labelWrapper_(0),
          trainer_(0) {
    log("initialize nn-trainer-feature-scorer with feature dimension %i", inputDimension_);
    require_gt(inputDimension_, 0);

    int outputDim = paramOutputDimension(config);
    if (outputDim >= 0) {
        log("nn-trainer-feature-scorer will ignore mixture-set number of classes %i but use %i instead", nClasses_, outputDim);
        nClasses_ = outputDim;
    }

    trainer_ = NeuralNetworkTrainer<Float>::createUnsupervisedTrainer(config);
    if (!trainer_)
        criticalError("failed to init trainer");
    if (!trainer_->hasClassLabelPosteriors())
        criticalError("cannot calculate posteriors with this trainer");

    labelWrapper_ = new ClassLabelWrapper(select("class-labels"), nClasses_);
    if (!labelWrapper_->isOneToOneMapping())
        error("no one-to-one correspondence between network outputs and classes!");

    require_eq(trainer_->getClassLabelPosteriorDimension(), labelWrapper_->nClassesToAccumulate());

    std::vector<u32> streamSizes(1 /* one single stream */, inputDimension_);
    trainer_->initializeTrainer(1000 /* dummy buffer size, will be resized automatically */, streamSizes);

    if (prior_.scale() != 0) {
        if (prior_.fileName() != "")
            prior_.read();
        else
            prior_.setFromMixtureSet(mixtureSet, *labelWrapper_);
        // The prior classes are the NN output classes.
        require_eq(labelWrapper_->nClassesToAccumulate(), prior_.size());
    }
}

TrainerFeatureScorer::~TrainerFeatureScorer() {
    delete labelWrapper_;
    delete trainer_;
}

/**
 * Stores the current feature and the number of buffered features.
 * All computations are done in BatchFeatureScorer.
 * This class is used only because it is required by the
 * FeatureScorer interface.
 */
class TrainerFeatureScorer::ContextScorer : public FeatureScorer::ContextScorer {
public:
    ContextScorer(const TrainerFeatureScorer* parent, u32 currentFeature, u32 batchIteration)
            : parent_(parent),
              currentFeature_(currentFeature),
              batchIteration_(batchIteration) {}
    virtual ~ContextScorer() {}
    virtual Mm::EmissionIndex nEmissions() const {
        return parent_->nMixtures();
    }
    virtual Mm::Score score(Mm::EmissionIndex e) const {
        require_eq(batchIteration_, parent_->batchIteration_);
        return parent_->getScore(e, currentFeature_);
    }

private:
    const TrainerFeatureScorer* parent_;
    u32                         currentFeature_;
    u32                         batchIteration_;
};

void TrainerFeatureScorer::_addFeature(const Mm::FeatureVector& f) const {
    require(!bufferFilled());
    require(!scoresComputed_);
    if (inputDimension_ != f.size()) {
        criticalError("feature-scorer was configured with input dimension %i but we got features with dimension %zu",
                      inputDimension_, f.size());
    }
    buffer_.push_back(f);
}

void TrainerFeatureScorer::addFeature(const Mm::FeatureVector& f) const {
    // Lazily call reset() when flush() went through all the buffer before.
    if (currentFeature_ > 0 && currentFeature_ >= buffer_.size())
        reset();
    _addFeature(f);
}

void TrainerFeatureScorer::reset() const {
    if (!buffer_.empty()) {
        trainer_->processBatch_finish();
        buffer_.clear();
        buffer_.shrink_to_fit();
    }
    scoresComputed_ = false;
    currentFeature_ = 0;
    batchIteration_++;
}

Mm::EmissionIndex TrainerFeatureScorer::nMixtures() const {
    require_gt(nClasses_, 0);
    return nClasses_;
}

void TrainerFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
    require_gt(inputDimension_, 0);
    description.mainStream().setValue(Mm::FeatureDescription::nameDimension, inputDimension_);
}

// See comment in header. The scorer is not for `f`.
Mm::FeatureScorer::Scorer TrainerFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    _addFeature(f);  // Don't reset() yet.
    return flush();
}

Mm::FeatureScorer::Scorer TrainerFeatureScorer::flush() const {
    require_lt(currentFeature_, buffer_.size());
    Scorer scorer(new ContextScorer(this, currentFeature_, batchIteration_));
    currentFeature_++;
    // We must not call reset() here because the calls to getScore() will be delayed.
    return scorer;
}

Mm::Score TrainerFeatureScorer::getScore(Mm::EmissionIndex e, u32 position) const {
    require_lt(position, buffer_.size());
    // process buffer if needed
    if (!scoresComputed_) {
        // We need to copy the buffer into a NN trainer compatible format.
        typedef Types<Float>::NnMatrix NnMatrix;
        std::vector<NnMatrix>          nnBuffer(1);  // always one input stream (at the moment. implement addFeature(Core::Ref<const Feature>)...)
        nnBuffer[0].resize(buffer_[0].size(), buffer_.size());
        for (u32 t = 0; t < buffer_.size(); ++t)
            for (u32 i = 0; i < buffer_[0].size(); ++i)
                nnBuffer[0].at(i, t) = buffer_[t].at(i);
        trainer_->processBatch_feedInput(nnBuffer, NULL, NULL);
        // scores and buffer must be readable/writable
        trainer_->getClassLabelPosteriors().finishComputation();
        // mark computed
        scoresComputed_ = true;
    }
    // return score in -log space
    Mm::Score score = 0.0;
    if (labelWrapper_->isClassToAccumulate(e)) {
        u32 idx = labelWrapper_->getOutputIndexFromClassIndex(e);
        // Get score in std space.
        score = trainer_->getClassLabelPosteriors().at(idx, position);
        if (returnScoresInNegLog_) {
            score = -Core::log(score);  // transfer to -log space
            if (prior_.scale() != 0)
                score -= -prior_.at(idx) * prior_.scale();  // priors are in +log space. substract them
        }
        else
            // not sure if that make sense otherwise
            require(prior_.scale() == 0);
    }
    else {
        if (returnScoresInNegLog_)
            score = Core::Type<Mm::Score>::max;  // = score 0 in std space
        else
            score = 0;
    }
    return score;
}

Mm::FeatureScorer::Scorer TrainerFeatureScorer::getTimeIndexedScorer(u32 time) const {
    require_lt(time, buffer_.size());
    Scorer scorer(new ContextScorer(this, time, batchIteration_));
    return scorer;
}

// Must never be full. We want to support segments of any len, and we want to
// get all features in advance before we calculate the scores (to support bi-RNNs).
bool TrainerFeatureScorer::bufferFilled() const {
    return false;
}  // == cannot call addFeature() anymore
bool TrainerFeatureScorer::bufferEmpty() const {
    return currentFeature_ >= buffer_.size();
}  // == cannot call flush() anymore
u32 TrainerFeatureScorer::bufferSize() const {
    return u32(-1);
}

}  // namespace Nn
