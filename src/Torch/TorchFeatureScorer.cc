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

#include "TorchFeatureScorer.hh"

#include <chrono>
#include <cstring>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

/*
 * Stores the current feature and the number of buffered features.
 * All computations are done in TorchFeatureScorer.
 * This class is used only because it is required by the
 * FeatureScorer interface.
 */
class TorchFeatureScorer::ContextScorer : public FeatureScorer::ContextScorer {
public:
    ContextScorer(const TorchFeatureScorer* parent, u32 currentFeature, u32 batchIteration)
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
    const TorchFeatureScorer* parent_;
    u32                       currentFeature_;
    u32                       batchIteration_;
};

const Core::ParameterBool TorchFeatureScorer::paramApplyLogOnOutput(
        "apply-log-on-output",
        "whether to apply the log-function on the output, useful if the model outputs softmax instead of log-softmax",
        false);

const Core::ParameterBool TorchFeatureScorer::paramNegateOutput(
        "negate-output",
        "whether to negate output (because the model outputs log-softmax and not negative log-softmax)",
        true);

const Core::ParameterBool TorchFeatureScorer::paramUseOutputAsIs(
        "use-output-as-is",
        "return the output of the neural network without modification",
        false);

const Core::ParameterInt TorchFeatureScorer::paramFeatureDimension(
        "feature-dimension",
        "expected feature input dimension",
        80);

const Core::ParameterInt TorchFeatureScorer::paramOutputDimension(
        "output-dimension",
        "expected output dimension",
        0);

TorchFeatureScorer::TorchFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          Mm::FeatureScorer(config),
          applyLogOnOutput_(paramApplyLogOnOutput(config)),
          negateOutput_(paramNegateOutput(config)),
          useOutputAsIs_(paramUseOutputAsIs(config)),
          prior_(config),
          labelWrapper_(),
          expectedFeatureDim_(paramFeatureDimension(config)),
          expectedOutputDim_(paramOutputDimension(config)),
          inputBuffer_(),
          currentFeature_(0),
          batchIteration_(0),
          scoresComputed_(false),
          scores_(new Math::FastMatrix<Float>()),
          torchModel_(select("model")),
          stateManager_() {
    stateManager_ = StateManager::create(select("state-manager"));

    if (torchModel_.hasJsonIoSpec()) {
        initializeStatesFromModelSpec();
    }
    stateManager_->setInitialStates(stateVariables_);

    require_gt(expectedFeatureDim_, 0);
    require_gt(expectedOutputDim_, 0);

    labelWrapper_ = std::unique_ptr<Nn::ClassLabelWrapper>(new Nn::ClassLabelWrapper(select("class-labels"), expectedOutputDim_));

    if (prior_.scale() != 0.0f) {
        if (prior_.fileName() != "") {
            prior_.read();
        }
        else if (mixtureSet->nDensities() > 0) {
            prior_.setFromMixtureSet(mixtureSet, *labelWrapper_);
        }
        else {
            prior_.initUniform(expectedOutputDim_);
        }

        // The prior classes are the NN output classes.
        require_eq(labelWrapper_->nClassesToAccumulate(), prior_.size());
    }
}

void TorchFeatureScorer::initializeStatesFromModelSpec() {
    if (!torchModel_.hasParsedIoSpec()) {
        criticalError("JSON I/O spec is configured, but no parsed I/O spec is available.");
    }

    const auto& ioSpec = torchModel_.parsedIoSpec();

    if (ioSpec.featuresShape.empty()) {
        criticalError("JSON I/O spec does not contain a valid features shape");
    }
    if (ioSpec.outputsShape.empty()) {
        criticalError("JSON I/O spec does not contain a valid log_probs shape");
    }

    expectedFeatureDim_ = static_cast<int>(ioSpec.featuresShape.back());
    expectedOutputDim_  = static_cast<int>(ioSpec.outputsShape.back());

    stateVariables_.clear();
    stateVariables_.reserve(ioSpec.states.size());

    for (const auto& stateSpec : ioSpec.states) {
        TorchStateVariable stateVar;
        stateVar.name        = stateSpec.name;
        stateVar.kind        = stateSpec.kind;
        stateVar.inputIndex  = ioSpec.statesInputIndex + stateSpec.inputIndex;
        stateVar.outputIndex = ioSpec.statesOutputIndex + stateSpec.outputIndex;
        stateVar.layer       = stateSpec.layer;
        stateVar.shape       = stateSpec.shape;
        stateVar.dtype       = stateSpec.dtype;

        stateVariables_.push_back(std::move(stateVar));
    }
}

Mm::EmissionIndex TorchFeatureScorer::nMixtures() const {
    return expectedOutputDim_;
}

void TorchFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
    description.mainStream().setValue(Mm::FeatureDescription::nameDimension, expectedFeatureDim_);
}

Mm::FeatureScorer::Scorer TorchFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    addFeatureInternal(f);
    return flush();
}

Mm::Score TorchFeatureScorer::getScore(Mm::EmissionIndex e, u32 position) const {
    const_cast<TorchFeatureScorer*>(this)->computeScoresInternal();
    require_lt(position, scores_->nRows());

    Mm::Score score = Core::Type<Mm::Score>::max;

    if (labelWrapper_->isClassToAccumulate(e)) {
        u32 idx = labelWrapper_->getOutputIndexFromClassIndex(e);
        score   = scores_->at(position, idx);
        if (useOutputAsIs_) {
            return score;
        }
        if (applyLogOnOutput_) {
            score = Core::log(score);
        }
        if (negateOutput_) {
            score = -score;
        }

        if (prior_.scale() != 0.0f) {
            score -= -prior_.at(idx) * prior_.scale();  // priors are in +log space, so substract them
        }
    }

    return score;
}

void TorchFeatureScorer::reset() const {
    if (!inputBuffer_.empty()) {
        inputBuffer_.clear();
        inputBuffer_.shrink_to_fit();
    }
    scoresComputed_ = false;
    currentFeature_ = 0;
    batchIteration_++;
}

void TorchFeatureScorer::finalize() const {
    stateManager_->setInitialStates(stateVariables_);
}

void TorchFeatureScorer::addFeature(const Mm::FeatureVector& f) const {
    // Lazily call reset() when flush() went through all the buffer before.
    if (currentFeature_ > 0 && scoresComputed_ && currentFeature_ >= scores_->nRows()) {
        reset();
    }
    addFeatureInternal(f);
}

Mm::FeatureScorer::Scorer TorchFeatureScorer::flush() const {
    const_cast<TorchFeatureScorer*>(this)->computeScoresInternal();
    require_lt(currentFeature_, scores_->nRows());
    Scorer scorer(new ContextScorer(this, currentFeature_, batchIteration_));
    currentFeature_++;
    // We must not call reset() here because the calls to getScore() will be delayed.
    return scorer;
}

Mm::FeatureScorer::Scorer TorchFeatureScorer::getTimeIndexedScorer(u32 time) const {
    const_cast<TorchFeatureScorer*>(this)->computeScoresInternal();
    require_lt(time, scores_->nRows());
    Scorer scorer(new ContextScorer(this, time, batchIteration_));
    return scorer;
}

void TorchFeatureScorer::addFeatureInternal(const Mm::FeatureVector& f) const {
    require(!bufferFilled());
    require(!scoresComputed_);
    if (static_cast<size_t>(expectedFeatureDim_) != f.size()) {
        criticalError("feature-scorer was configured with input dimension %li but we got features with dimension %zu", expectedFeatureDim_, f.size());
    }
    inputBuffer_.push_back(f);
}

void TorchFeatureScorer::computeScoresInternal() {
    if (!scoresComputed_) {
        // Create session inputs
        size_t T_in = inputBuffer_.size();

        torch::Tensor features = torch::empty({1, static_cast<int64_t>(T_in), static_cast<int64_t>(expectedFeatureDim_)}, torch::kFloat32);
        torch::Tensor lengths  = torch::tensor({static_cast<int64_t>(T_in)}, torch::kInt64);
        for (size_t t = 0; t < T_in; ++t) {
            std::memcpy(features[0][t].data_ptr<float>(), inputBuffer_[t].data(), expectedFeatureDim_ * sizeof(float));
        }

        std::vector<torch::Tensor> session_outputs;
        std::vector<torch::Tensor> session_inputs = torchModel_.makeInputs(features, lengths);
        stateManager_->extendInputs(session_inputs, stateVariables_);

        // Run session
        auto t_start = std::chrono::steady_clock::now();
        torchModel_.session().run(session_inputs, session_outputs);
        auto t_end = std::chrono::steady_clock::now();

        stateManager_->updateStates(session_outputs, stateVariables_);

        // Read model outputs
        torch::Tensor out   = torchModel_.outputsFrom(session_outputs).contiguous();
        size_t        T_out = out.size(1);

        scores_->resize(T_out, expectedOutputDim_);

        auto outAcc = out.accessor<float, 3>();
        for (size_t t = 0; t < T_out; ++t) {
            for (size_t c = 0; c < static_cast<size_t>(expectedOutputDim_); ++c) {
                scores_->at(t, c) = outAcc[0][t][c];
            }
        }

        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        log("num_frames: %zu elapsed: %f AM_RTF: %f", T_in, t_elapsed, t_elapsed / (T_in / 100.0));

        // mark computed
        scoresComputed_ = true;
    }
}

}  // namespace Torch