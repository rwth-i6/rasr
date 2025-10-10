/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "OnnxFeatureScorer.hh"

#include "Value.hh"

namespace {

std::vector<Onnx::IOSpecification> getIOSpec(bool static_dims, int64_t num_features, int64_t num_classes) {
    return std::vector<Onnx::IOSpecification>({
            Onnx::IOSpecification{"features", Onnx::IODirection::INPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::FLOAT}, {{-1, -1, num_features}, {1, static_dims ? -2 : -1, num_features}}},
            Onnx::IOSpecification{"features-size", Onnx::IODirection::INPUT, true, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::INT32}, {{-1}}},
            Onnx::IOSpecification{"output", Onnx::IODirection::OUTPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::FLOAT}, {{-1, static_dims ? -2 : -1, num_classes}, {1, static_dims ? -2 : -1, num_classes}}},
    });
}

}  // namespace

namespace Onnx {

/**
 * Stores the current feature and the number of buffered features.
 * All computations are done in OnnxFeatureScorer.
 * This class is used only because it is required by the
 * FeatureScorer interface.
 */
class OnnxFeatureScorer::ContextScorer : public FeatureScorer::ContextScorer {
public:
    ContextScorer(const OnnxFeatureScorer* parent, u32 currentFeature, u32 batchIteration)
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
    const OnnxFeatureScorer* parent_;
    u32                      currentFeature_;
    u32                      batchIteration_;
};

const Core::ParameterBool OnnxFeatureScorer::paramAllowStaticDimensions("allow-static-dimensions", "whether to allow static input/output dimensions for the time axis", false);

const Core::ParameterBool OnnxFeatureScorer::paramApplyLogOnOutput("apply-log-on-output", "whether to apply the log-function on the output, usefull if the model outputs softmax instead of log-softmax", false);

const Core::ParameterBool OnnxFeatureScorer::paramNegateOutput("negate-output", "wether to negate output (because the model outputs log softmax and not negative log softmax", true);
const Core::ParameterBool OnnxFeatureScorer::paramUseOutputAsIs(
        "use-output-as-is", "return the output of the neural network without modification", false);

OnnxFeatureScorer::OnnxFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          Mm::FeatureScorer(config),
          allowStaticDimensions_(paramAllowStaticDimensions(config)),
          applyLogOnOutput_(paramApplyLogOnOutput(config)),
          negateOutput_(paramNegateOutput(config)),
          useOutputAsIs_(paramUseOutputAsIs(config)),
          prior_(config),
          labelWrapper_(),
          expectedFeatureDim_(0l),
          expectedOutputDim_(0l),
          inputBuffer_(),
          currentFeature_(0),
          batchIteration_(0),
          scoresComputed_(false),
          scores_(new Math::FastMatrix<Float>()),
          session_(select("session")),
          ioSpec_(getIOSpec(allowStaticDimensions_, -2, -2)),
          mapping_(select("io-map"), ioSpec_),
          validator_(select("validator")),
          state_manager_() {
    bool valid = validator_.validate(ioSpec_, mapping_, session_);
    if (not valid) {
        log("Failed to validate input model");
    }

    expectedFeatureDim_ = session_.getInputShape(mapping_.getOnnxName("features")).back();
    expectedOutputDim_  = session_.getOutputShape(mapping_.getOnnxName("output")).back();
    require_gt(expectedFeatureDim_, 0);
    require_gt(expectedOutputDim_, 0);

    labelWrapper_ = std::unique_ptr<Nn::ClassLabelWrapper>(new Nn::ClassLabelWrapper(select("class-labels"), expectedOutputDim_));

    if (prior_.scale() != 0.0f) {
        if (prior_.fileName() != "") {
            prior_.read();
        }
        else if (mixtureSet->nDensities() > 0) {  // is not empty mixture set
            prior_.setFromMixtureSet(mixtureSet, *labelWrapper_);
        }
        else {
            prior_.initUniform(expectedOutputDim_);
        }

        // The prior classes are the NN output classes.
        require_eq(labelWrapper_->nClassesToAccumulate(), prior_.size());
    }

    state_manager_   = StateManager::create(select("state-manager"));
    state_variables_ = session_.getStateVariablesMetadata();
    state_manager_->setInitialStates(state_variables_);
}

Mm::EmissionIndex OnnxFeatureScorer::nMixtures() const {
    return expectedOutputDim_;
}

void OnnxFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
    description.mainStream().setValue(Mm::FeatureDescription::nameDimension, expectedFeatureDim_);
}

// See comment in header. The scorer is not for `f`.
Mm::FeatureScorer::Scorer OnnxFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    addFeatureInternal(f);  // Don't reset() yet.
    return flush();
}

Mm::Score OnnxFeatureScorer::getScore(Mm::EmissionIndex e, u32 position) const {
    const_cast<OnnxFeatureScorer*>(this)->computeScoresInternal();
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
            score -= -prior_.at(idx) * prior_.scale();  // priors are in +log space. substract them
        }
    }

    return score;
}

void OnnxFeatureScorer::reset() const {
    if (!inputBuffer_.empty()) {
        inputBuffer_.clear();
        inputBuffer_.shrink_to_fit();
    }
    scoresComputed_ = false;
    currentFeature_ = 0;
    batchIteration_++;
}

void OnnxFeatureScorer::finalize() const {
    state_manager_->setInitialStates(state_variables_);
}

void OnnxFeatureScorer::addFeature(const Mm::FeatureVector& f) const {
    // Lazily call reset() when flush() went through all the buffer before.
    if (currentFeature_ > 0 && scoresComputed_ && currentFeature_ >= scores_->nRows()) {
        reset();
    }
    addFeatureInternal(f);
}

Mm::FeatureScorer::Scorer OnnxFeatureScorer::flush() const {
    const_cast<OnnxFeatureScorer*>(this)->computeScoresInternal();
    require_lt(currentFeature_, scores_->nRows());
    Scorer scorer(new ContextScorer(this, currentFeature_, batchIteration_));
    currentFeature_++;
    // We must not call reset() here because the calls to getScore() will be delayed.
    return scorer;
}

Mm::FeatureScorer::Scorer OnnxFeatureScorer::getTimeIndexedScorer(u32 time) const {
    const_cast<OnnxFeatureScorer*>(this)->computeScoresInternal();
    require_lt(time, scores_->nRows());
    Scorer scorer(new ContextScorer(this, time, batchIteration_));
    return scorer;
}

void OnnxFeatureScorer::addFeatureInternal(const Mm::FeatureVector& f) const {
    require(!bufferFilled());
    require(!scoresComputed_);
    if (static_cast<size_t>(expectedFeatureDim_) != f.size()) {
        criticalError("feature-scorer was configured with input dimension %li but we got features with dimension %zu",
                      expectedFeatureDim_, f.size());
    }
    inputBuffer_.push_back(f);
}

void OnnxFeatureScorer::computeScoresInternal() {
    if (!scoresComputed_) {
        size_t                                     num_frames = inputBuffer_.size();
        std::vector<std::pair<std::string, Value>> inputs;
        std::vector<std::string>                   output_names;

        // input features and output scores
        inputs.emplace_back(mapping_.getOnnxName("features"), createInputValue());
        if (mapping_.hasOnnxName("features-size")) {
            inputs.emplace_back(mapping_.getOnnxName("features-size"),
                                Value::create(std::vector<s32>{static_cast<s32>(num_frames)}));
        }
        output_names.push_back(mapping_.getOnnxName("output"));

        // input and output states
        state_manager_->extendFeedDict(inputs, state_variables_);
        state_manager_->extendTargets(output_names, state_variables_);

        auto t_start = std::chrono::steady_clock::now();

        std::vector<Value> output;
        session_.run(std::move(inputs), output_names, output);
        output[0].get<>(0, *scores_);  // output scores

        // get new states
        std::vector<Value> output_states;
        for (size_t i = 1ul; i < output.size(); i++) {  // other model outputs
            output_states.emplace_back(std::move(output[i]));
        }
        state_manager_->updateStates(output_states);

        auto t_end     = std::chrono::steady_clock::now();
        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        log("num_frames: %zu elapsed: %f AM_RTF: %f", num_frames, t_elapsed, t_elapsed / (num_frames / 100.0));

        // mark computed
        scoresComputed_ = true;
    }
}

Value OnnxFeatureScorer::createInputValue() const {
    // copy from deque into a matrix
    size_t num_frames = inputBuffer_.size();
    require_gt(num_frames, 0);
    std::vector<Math::FastMatrix<f32>> nnBuffer(1);  // single "batch"
    nnBuffer[0].resize(inputBuffer_[0].size(), num_frames);

    for (u32 t = 0; t < num_frames; ++t) {
        for (u32 i = 0; i < inputBuffer_[0].size(); ++i) {
            nnBuffer[0].at(i, t) = inputBuffer_[t].at(i);
        }
    }
    return Value::create(nnBuffer, true);
}

}  // namespace Onnx
