/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#ifndef ADVANCEDTREESEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH
#define ADVANCEDTREESEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH

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
        : Precursor(config),
          initialPruning_(initialPruning) {
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

#endif  // ADVANCEDTREESEARCH_DYNAMIC_BEAM_PRUNING_STRATEGY_HH
