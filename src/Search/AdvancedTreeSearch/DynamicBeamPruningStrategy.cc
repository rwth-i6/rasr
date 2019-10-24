#include "DynamicBeamPruningStrategy.hh"

namespace {

enum DynamicBeamPruningStrategyType {
    None,
    MaximumDelayBeamPruningStrategyType
};

const Core::Choice dynamicBeamPruningStrategyChoice(
        "none", DynamicBeamPruningStrategyType::None,
        "maximum-delay", MaximumDelayBeamPruningStrategyType,
        Core::Choice::endMark());

const Core::ParameterChoice paramDynamicBeamPruningStrategyType(
        "type",
        &dynamicBeamPruningStrategyChoice,
        "which dynamic beam pruning strategy should be used",
        DynamicBeamPruningStrategyType::None);

}  // namespace

namespace Search {

std::unique_ptr<DynamicBeamPruningStrategy> createDynamicBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning) {
    switch (paramDynamicBeamPruningStrategyType(config)) {
        case MaximumDelayBeamPruningStrategyType:
            return std::unique_ptr<DynamicBeamPruningStrategy>(new MaximumDelayBeamPruningStrategy(config, initialPruning));
        case None:
        default:
            return std::unique_ptr<DynamicBeamPruningStrategy>();
    }
}

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramAddInitialDelayPerFrameTime(
        "add-initial-delay-per-frame-time",
        "As it is difficult to get access to the number of frames in this class, we assume that the AM takes this many "
        "milliseconds to process one frame and distribute the initial delay over the utterance using this duration.",
        2.0, 0.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramDecrementBeamThreshold(
        "decrement-beam-threshold",
        "number of milliseconds of effective delay that trigger decrementing the beam size",
        500.0, 0.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramIncrementBeamThreshold(
        "increment-beam-threshold",
        "number of milliseconds of effective delay that trigger incrementing the beam size",
        100.0, 0.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramMaximumBeamScale(
        "maximum-beam-scale",
        "maximum scaling factor for beam-pruning",
        1.0, 0.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramMinimumBeamScale(
        "minimum-beam-scale",
        "minimum scaling factor for beam-pruning",
        1.0, 0.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramDecrementBeamFactor(
        "decrement-beam-factor",
        "when beam-pruning is decremented it is scaled by this factor",
        0.95, 0.0, 1.0);

const Core::ParameterFloat MaximumDelayBeamPruningStrategy::paramIncrementBeamFactor(
        "increment-beam-factor",
        "when beam-pruning is incremented it is scaled by this factor",
        1.0 / 0.95, 1.0);

SearchAlgorithm::PruningRef MaximumDelayBeamPruningStrategy::startNewSegment() {
    currentScale_ = 1.0;
    return initialPruning_;
}

void MaximumDelayBeamPruningStrategy::frameFinished(TimeframeIndex time, f64 current_frame_time, f64 delay) {
    if (time == 1) {
        initialDelay_ = delay - current_frame_time;
    }
    delay -= initialDelay_ + std::min(initialDelay_, addInitialDelayPerFrameTime_ * time);
    if (delay >= decrementBeamThreshold_) {
        currentScale_ *= decrementBeamFactor_;
        currentScale_ = std::max<Score>(minimumBeamScale_, currentScale_);
    }
    else if (delay <= incrementBeamThreshold_) {
        currentScale_ *= incrementBeamFactor_;
        currentScale_ = std::min<Score>(maximumBeamScale_, currentScale_);
    }
    std::cerr << "frame: " << time << " time: " << current_frame_time << " delay: " << delay << " initial delay: " << initialDelay_ << " scale: " << currentScale_ << std::endl;
}

SearchAlgorithm::PruningRef MaximumDelayBeamPruningStrategy::newPruningThresholds() {
    SearchAlgorithm::PruningRef res = initialPruning_->clone();
    res->extend(currentScale_, 0.0, 0);
    return res;
}

}  // namespace Search
