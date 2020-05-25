#ifndef _SEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH
#define _SEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH

#include <Core/Component.hh>
#include <Search/Search.hh>

namespace Search {

class DynamicBeamPruningStrategy;
std::unique_ptr<DynamicBeamPruningStrategy> createDynamicBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning);

class DynamicBeamPruningStrategy : public Core::Component {
public:
    using Precursor = Core::Component;

    DynamicBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning);
    virtual ~DynamicBeamPruningStrategy() = default;

    virtual SearchAlgorithm::PruningRef startNewSegment();
    virtual void                        frameFinished(TimeframeIndex time, f64 current_frame_time, f64 delay) = 0;
    virtual SearchAlgorithm::PruningRef newPruningThresholds()                                                = 0;

protected:
    SearchAlgorithm::PruningRef initialPruning_;
};

class MaximumDelayBeamPruningStrategy : public DynamicBeamPruningStrategy {
public:
    using Precursor = DynamicBeamPruningStrategy;

    static const Core::ParameterFloat paramAddInitialDelayPerFrameTime;
    static const Core::ParameterFloat paramDecrementBeamThreshold;
    static const Core::ParameterFloat paramIncrementBeamThreshold;
    static const Core::ParameterFloat paramMaximumBeamScale;
    static const Core::ParameterFloat paramMinimumBeamScale;
    static const Core::ParameterFloat paramDecrementBeamFactor;
    static const Core::ParameterFloat paramIncrementBeamFactor;

    MaximumDelayBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning);
    virtual ~MaximumDelayBeamPruningStrategy() = default;

    virtual SearchAlgorithm::PruningRef startNewSegment();
    virtual void                        frameFinished(TimeframeIndex time, f64 current_frame_time, f64 delay);
    virtual SearchAlgorithm::PruningRef newPruningThresholds();

private:
    f64   addInitialDelayPerFrameTime_;
    f64   decrementBeamThreshold_;
    f64   incrementBeamThreshold_;
    Score maximumBeamScale_;
    Score minimumBeamScale_;
    Score decrementBeamFactor_;
    Score incrementBeamFactor_;

    Score currentScale_;
    f64   initialDelay_;
};

// iniline implementations

inline DynamicBeamPruningStrategy::DynamicBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning)
        : Precursor(config), initialPruning_(initialPruning) {
}

inline SearchAlgorithm::PruningRef DynamicBeamPruningStrategy::startNewSegment() {
    return SearchAlgorithm::PruningRef();
}

inline MaximumDelayBeamPruningStrategy::MaximumDelayBeamPruningStrategy(Core::Configuration const& config, SearchAlgorithm::PruningRef initialPruning)
        : Precursor(config, initialPruning),
          addInitialDelayPerFrameTime_(paramAddInitialDelayPerFrameTime(config)),
          decrementBeamThreshold_(paramDecrementBeamThreshold(config)),
          incrementBeamThreshold_(paramIncrementBeamThreshold(config)),
          maximumBeamScale_(paramMaximumBeamScale(config)),
          minimumBeamScale_(paramMinimumBeamScale(config)),
          decrementBeamFactor_(paramDecrementBeamFactor(config)),
          incrementBeamFactor_(paramIncrementBeamFactor(config)) {
}

}  // namespace Search

#endif /* _SEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH */
