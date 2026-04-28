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
#include "TensorflowFeatureScorer.hh"
#include <thread>

#include "Module.hh"
/*
 * TensorflowFeatureScorer is based on Nn/TrainerFeatureScorer.cc.
 * Instead of instantiating a trainer, it feeds features through a TF graph
 */
namespace Tensorflow {

static const Core::ParameterInt paramFeatureDimension(
        "feature-dimension", "feature = input dimension");

static const Core::ParameterInt paramOutputDimension(
        "trainer-feature-scorer-output-dimension", "if set, will ignore the number of mixtures", -1);

static const Core::ParameterBool paramUseOutputAsIs(
        "use-output-as-is", "return the output of the neural network without modification (except prior)", false);

static const Core::ParameterBool paramReturnScoresInNegLog(
        "return-scores-in-neg-log", "return scores in -log space (default)", true);

static const Core::ParameterBool paramAsyncInitalization(
        "async-initialization", "initializa tensorflow asynchronously", false);

TensorflowFeatureScorer::TensorflowFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          Mm::FeatureScorer(config),
          nClasses_(mixtureSet->nMixtures()),
          inputDimension_(paramFeatureDimension(config)),
          useOutputAsIs_(paramUseOutputAsIs(config)),
          returnScoresInNegLog_(paramReturnScoresInNegLog(config)),
          asyncInitialization_(paramAsyncInitalization(config)),
          prior_(config),
          currentFeature_(0),
          scoresComputed_(false),
          batchIteration_(0),
          session_(select("session")),
          loader_(Module::instance().createGraphLoader(select("loader"))),
          graph_(loader_->load_graph()),
          tensor_input_map_(select("input-map")),
          tensor_output_map_(select("output-map")),
          state_manager_() {
    if (asyncInitialization_) {
        std::thread t(&TensorflowFeatureScorer::initializeTensorflow, this);
        t.detach();
    }
    else {
        initializeTensorflow();
    }
    // no need to resize as Tensor::get() will resize on demand
    scores_ = std::unique_ptr<Math::FastMatrix<Float>>(new Math::FastMatrix<Float>);

    log("initialize tf-feature-scorer with feature dimension %i", inputDimension_);
    require_gt(inputDimension_, 0);

    int outputDim = paramOutputDimension(config);
    if (outputDim >= 0) {
        log("tf-feature-scorer will ignore mixture-set number of classes %i but use %i instead", nClasses_, outputDim);
        nClasses_ = outputDim;
    }

    labelWrapper_ = std::unique_ptr<Nn::ClassLabelWrapper>(new Nn::ClassLabelWrapper(select("class-labels"), nClasses_));
    if (!labelWrapper_->isOneToOneMapping())
        error("no one-to-one correspondence between network outputs and classes!");

    if (prior_.scale() != 0) {
        if (prior_.fileName() != "")
            prior_.read();
        else
            prior_.setFromMixtureSet(mixtureSet, *labelWrapper_);
        // The prior classes are the NN output classes.
        require_eq(labelWrapper_->nClassesToAccumulate(), prior_.size());
    }
}

void TensorflowFeatureScorer::initializeTensorflow() {
    session_.addGraph(*graph_);
    loader_->initialize(session_);

    state_manager_ = StateManager::create(select("state-manager"), *graph_, session_);
    state_manager_->setInitialState();

    output_tensor_names_.push_back(tensor_output_map_.get_info("posterior").tensor_name());

    auto state_outputs = state_manager_->getOutputs();

    output_tensor_names_.insert(output_tensor_names_.end(), state_outputs.begin(), state_outputs.end());
    target_tensor_names_ = state_manager_->getTargets();

    tensorflowInitialized_.store(true);
}

/**
 * Stores the current feature and the number of buffered features.
 * All computations are done in TensorflowFeatureScorer.
 * This class is used only because it is required by the
 * FeatureScorer interface.
 */
class TensorflowFeatureScorer::ContextScorer : public FeatureScorer::ContextScorer {
public:
    ContextScorer(const TensorflowFeatureScorer* parent, u32 currentFeature, u32 batchIteration)
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
    const TensorflowFeatureScorer* parent_;
    u32                            currentFeature_;
    u32                            batchIteration_;
};

void TensorflowFeatureScorer::_addFeature(const Mm::FeatureVector& f) const {
    require(!bufferFilled());
    require(!scoresComputed_);
    if (inputDimension_ != f.size()) {
        criticalError("feature-scorer was configured with input dimension %i but we got features with dimension %zu",
                      inputDimension_, f.size());
    }
    buffer_.push_back(f);
}

void TensorflowFeatureScorer::addFeature(const Mm::FeatureVector& f) const {
    // Lazily call reset() when flush() went through all the buffer before.
    if (currentFeature_ > 0 && scoresComputed_ && currentFeature_ >= scores_->nRows()) {
        reset();
    }
    _addFeature(f);
}

void TensorflowFeatureScorer::reset() const {
    if (!buffer_.empty()) {
        buffer_.clear();
        buffer_.shrink_to_fit();
    }
    scoresComputed_ = false;
    currentFeature_ = 0;
    batchIteration_++;
}

void TensorflowFeatureScorer::finalize() const {
    waitForInitialization();
    state_manager_->setInitialState();
}

Mm::EmissionIndex TensorflowFeatureScorer::nMixtures() const {
    require_gt(nClasses_, 0);
    return nClasses_;
}

void TensorflowFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
    require_gt(inputDimension_, 0);
    description.mainStream().setValue(Mm::FeatureDescription::nameDimension, inputDimension_);
}

// See comment in header. The scorer is not for `f`.
Mm::FeatureScorer::Scorer TensorflowFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    _addFeature(f);  // Don't reset() yet.
    return flush();
}

Mm::FeatureScorer::Scorer TensorflowFeatureScorer::flush() const {
    const_cast<TensorflowFeatureScorer*>(this)->_compute();
    require_lt(currentFeature_, scores_->nRows());
    Scorer scorer(new ContextScorer(this, currentFeature_, batchIteration_));
    currentFeature_++;
    // We must not call reset() here because the calls to getScore() will be delayed.
    return scorer;
}

Tensor TensorflowFeatureScorer::_createInputTensor() const {
    // copy from deque into a matrix
    size_t num_frames = buffer_.size();
    require_gt(num_frames, 0);
    std::vector<Math::FastMatrix<f32>> nnBuffer(1);  // single "batch"
    nnBuffer[0].resize(buffer_[0].size(), num_frames);

    for (u32 t = 0; t < num_frames; ++t) {
        for (u32 i = 0; i < buffer_[0].size(); ++i) {
            nnBuffer[0].at(i, t) = buffer_[t].at(i);
        }
    }
    return Tensor::create(nnBuffer, true);
}

void TensorflowFeatureScorer::_compute() {
    if (!scoresComputed_) {
        size_t                                      num_frames = buffer_.size();
        std::vector<std::pair<std::string, Tensor>> inputs;
        auto const&                                 tensor_info = tensor_input_map_.get_info("features");
        inputs.push_back({tensor_info.tensor_name(), _createInputTensor()});
        if (not tensor_info.seq_length_tensor_name().empty()) {
            inputs.push_back({tensor_info.seq_length_tensor_name(),
                              Tensor::create(std::vector<s32>{static_cast<s32>(num_frames)})});
        }

        auto                t_start = std::chrono::system_clock::now();
        std::vector<Tensor> tf_output;

        waitForInitialization();

        // evaluate graph
        session_.run(inputs, output_tensor_names_, target_tensor_names_, tf_output);

        tf_output[0].get<>(0, *scores_);

        std::vector<Tensor> state_vars(tf_output.size() - 1ul);
        for (size_t i = 1ul; i < tf_output.size(); i++) {
            state_vars.emplace_back(std::move(tf_output[i]));
        }
        state_manager_->updateState(state_vars);

        auto t_end     = std::chrono::system_clock::now();
        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        printf("num_frames: %zu elapsed: %f AM_RTF: %f\n", num_frames, t_elapsed, t_elapsed / (num_frames / 100.0));

        // mark computed
        scoresComputed_ = true;
    }
}

Mm::Score TensorflowFeatureScorer::getScore(Mm::EmissionIndex e, u32 position) const {
    const_cast<TensorflowFeatureScorer*>(this)->_compute();
    require_lt(position, scores_->nRows());

    Mm::Score score = 0.0;
    if (labelWrapper_->isClassToAccumulate(e)) {
        u32 idx = labelWrapper_->getOutputIndexFromClassIndex(e);
        score   = -scores_->at(position, idx); /* note the minus! this requires the output layer to use activation = 'log_softmax' */

        if (useOutputAsIs_) {
            score = -score;
        }
        else if (returnScoresInNegLog_) {
            score = -Core::log(-score);  // transfer to -log space
        }

        if (prior_.scale() != 0) {
            score -= -prior_.at(idx) * prior_.scale();  // priors are in +log space. substract them
        }
    }
    else {
        if (returnScoresInNegLog_) {
            score = Core::Type<Mm::Score>::max;  // = score 0 in std space
        }
        else {
            score = 0;
        }
    }
    return score;
}

Mm::FeatureScorer::Scorer TensorflowFeatureScorer::getTimeIndexedScorer(u32 time) const {
    const_cast<TensorflowFeatureScorer*>(this)->_compute();
    require_lt(time, scores_->nRows());
    Scorer scorer(new ContextScorer(this, time, batchIteration_));
    return scorer;
}

// Must never be full. We want to support segments of any len, and we want to
// get all features in advance before we calculate the scores (to support bi-RNNs).
bool TensorflowFeatureScorer::bufferFilled() const {
    return false;
}  // == cannot call addFeature() anymore

bool TensorflowFeatureScorer::bufferEmpty() const {
    return scoresComputed_ && currentFeature_ >= scores_->nRows();
}  // == cannot call flush() anymore

u32 TensorflowFeatureScorer::bufferSize() const {
    return std::numeric_limits<u32>::max();
}

void TensorflowFeatureScorer::waitForInitialization() const {
    if (not asyncInitialization_) {
        return;
    }
    while (not tensorflowInitialized_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace Tensorflow
