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

#include "BatchedSearchBase.hh"

#include <algorithm>

#include <Core/XmlStream.hh>

namespace Search {

BatchedSearchBase::SegmentStateBase::SegmentStateBase(SegmentHandle handle, std::string const& name)
        : handle(handle),
          name(name),
          totalTimesteps(0ul),
          noMoreFeatures(false),
          finished(false),
          group(0ul),
          noMoreFeaturesWhenScored(false) {}

const Core::ParameterInt BatchedSearchBase::paramMaxActiveSegments(
        "max-active-segments",
        "Maximum number of segments that can be decoded at the same time. 0 means unlimited."
        " If any label scorer can only handle one segment at a time, this is always 1.",
        16,
        0);

const Core::ParameterInt BatchedSearchBase::paramNumSearchThreads(
        "num-search-threads",
        "Number of threads that run the search work of the active segments (creating, pruning and recombining"
        " hypotheses) in parallel. The label scorers are still called once per step for all segments.",
        1,
        1);

const Core::ParameterInt BatchedSearchBase::paramNumPipelineGroups(
        "num-pipeline-groups",
        "Number of groups the active segments are split into. While the label scorer call of one group runs in a"
        " background thread (e.g. on the GPU), the search work of another group is done, so that the label scorer"
        " computations and the search work overlap. Each label scorer call only contains the segments of one group,"
        " so max-active-segments should be increased accordingly. 1 means no pipelining.",
        1,
        1);

BatchedSearchBase::BatchedSearchBase(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV3(config),
          labelScorers_(),
          statisticsChannel_(config, "statistics"),
          stepwiseStatisticsChannel_(config, "stepwise-statistics"),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          scoringWaitTime_(),
          searchStepTime_(),
          parallelSearchTime_(0.0),
          threadBusyTimes_(),
          numSegmentsPerStep_("num-segments-per-step"),
          numContextsPerScoringCall_("num-contexts-per-scoring-call"),
          globalSearchStep_(0ul),
          maxActiveSegments_(paramMaxActiveSegments(config)),
          segments_(),
          nextSegmentHandle_(1ul),  // Handle 0 is the default segment of the label scorers which is used for single-segment search
          numActiveSegments_(0ul),
          numSearchThreads_(paramNumSearchThreads(config)),
          searchThreadPool_(),
          numPipelineGroups_(paramNumPipelineGroups(config)),
          nextGroup_(0ul),
          labelScorerMutex_(),
          pendingScorings_(numPipelineGroups_),
          scoringThread_() {
    if (numSearchThreads_ > 1ul) {
        log() << "Use " << numSearchThreads_ << " threads for the search work of the segments";
        // The calling thread also takes part, so one thread less is needed in the pool
        searchThreadPool_ = std::make_unique<FunctionThreadPool>();
        searchThreadPool_->init(numSearchThreads_ - 1ul);
    }
    threadBusyTimes_.resize(numSearchThreads_);
    if (numPipelineGroups_ > 1ul) {
        log() << "Split the active segments into " << numPipelineGroups_ << " pipeline groups";
        scoringThread_ = std::make_unique<FunctionThreadPool>();
        scoringThread_->init(1u);
    }
    if (stepwiseStatisticsChannel_.isOpen()) {
        disableParallelSearch("stepwise statistics are written");
    }
}

BatchedSearchBase::~BatchedSearchBase() {
    // Finish the label scorer calls in the background before anything else is destroyed
    scoringThread_.reset();
}

size_t BatchedSearchBase::maxActiveSegments() const {
    for (auto const& labelScorer : labelScorers_) {
        if (not labelScorer->supportsMultipleSegments()) {
            return 1ul;
        }
    }
    return maxActiveSegments_ == 0ul ? Core::Type<size_t>::max : maxActiveSegments_;
}

size_t BatchedSearchBase::numActiveSegments() const {
    return numActiveSegments_;
}

SearchAlgorithmV3::SegmentHandle BatchedSearchBase::addSegment(std::string const& segmentName) {
    if (numActiveSegments_ >= maxActiveSegments()) {
        criticalError() << "Can't add another segment since already " << numActiveSegments_ << " segments are active (maximum: " << maxActiveSegments() << ")";
    }

    initializationTime_.start();

    SegmentHandle handle = nextSegmentHandle_++;
    std::string   name   = segmentName.empty() ? std::to_string(handle) : segmentName;

    // The new segment goes to the pipeline group with the fewest unfinished segments
    std::vector<size_t> groupSizes(numPipelineGroups_, 0ul);
    for (auto const& [otherHandle, otherSegment] : segments_) {
        if (not otherSegment->finished) {
            ++groupSizes[otherSegment->group];
        }
    }
    size_t group = std::min_element(groupSizes.begin(), groupSizes.end()) - groupSizes.begin();

    {
        auto lock = lockLabelScorers();
        for (auto& labelScorer : labelScorers_) {
            labelScorer->addSegment(handle);
        }

        auto [it, inserted] = segments_.emplace(handle, createSegmentState(handle, name));
        verify(inserted);
        it->second->group = group;
    }

    ++numActiveSegments_;

    initializationTime_.stop();

    return handle;
}

void BatchedSearchBase::signalNoMoreFeatures(SegmentHandle segment) {
    auto& state = getSegmentBase(segment);
    if (state.noMoreFeatures) {
        return;
    }
    state.noMoreFeatures = true;
    if (state.finished) {
        return;
    }

    featureProcessingTime_.start();
    auto lock = lockLabelScorers();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreSegmentFeatures(segment);
    }
    featureProcessingTime_.stop();
}

void BatchedSearchBase::putFeature(SegmentHandle segment, Nn::DataView const& feature) {
    auto& state = getSegmentBase(segment);
    if (state.noMoreFeatures) {
        criticalError() << "Received feature for segment " << state.name << " after signalNoMoreFeatures";
    }
    if (state.finished) {
        return;
    }

    featureProcessingTime_.start();
    auto lock = lockLabelScorers();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addSegmentInput(segment, feature);
    }
    ++state.totalTimesteps;
    featureProcessingTime_.stop();
}

void BatchedSearchBase::putFeatures(SegmentHandle segment, Nn::DataView const& features, size_t nTimesteps) {
    auto& state = getSegmentBase(segment);
    if (state.noMoreFeatures) {
        criticalError() << "Received features for segment " << state.name << " after signalNoMoreFeatures";
    }
    if (state.finished) {
        return;
    }

    featureProcessingTime_.start();
    auto lock = lockLabelScorers();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addSegmentInputs(segment, features, nTimesteps);
    }
    state.totalTimesteps += nTimesteps;
    featureProcessingTime_.stop();
}

bool BatchedSearchBase::isFinished(SegmentHandle segment) const {
    return getSegmentBase(segment).finished;
}

void BatchedSearchBase::removeSegment(SegmentHandle segment) {
    auto& state = getSegmentBase(segment);
    if (not state.finished) {
        // A label scorer call in the background may still contain contexts of the segment
        waitForAllPendingScorings();
        // Data in the label scorers of finished segments is already released
        auto lock = lockLabelScorers();
        for (auto& labelScorer : labelScorers_) {
            labelScorer->removeSegment(segment);
        }
        --numActiveSegments_;
    }
    segments_.erase(segment);
}

std::vector<SearchAlgorithmV3::SegmentHandle> BatchedSearchBase::segments() const {
    std::vector<SegmentHandle> result;
    result.reserve(segments_.size());
    for (auto const& [handle, state] : segments_) {
        result.push_back(handle);
    }
    return result;
}

bool BatchedSearchBase::decodeStep() {
    // Without pipelining, all segments are in group 0. Otherwise the groups take turns. A group that can't make
    // progress is skipped so that `false` is only returned if no group can make progress.
    for (size_t i = 0ul; i < numPipelineGroups_; ++i) {
        size_t group = nextGroup_;
        nextGroup_   = (nextGroup_ + 1ul) % numPipelineGroups_;
        if (decodeGroupStep(group)) {
            return true;
        }
    }
    return false;
}

void BatchedSearchBase::setLabelScorers(std::vector<Core::Ref<Nn::LabelScorer>> const& labelScorers) {
    labelScorers_ = labelScorers;
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->supportsMultipleSegments()) {
            warning() << "Label scorer " << scorerIdx + 1 << " can only handle one segment at a time, segments are therefore decoded one after another";
        }
    }
    log() << "Decode at most " << maxActiveSegments() << " segments at the same time";
}

BatchedSearchBase::SegmentStateBase& BatchedSearchBase::getSegmentBase(SegmentHandle segment) {
    auto it = segments_.find(segment);
    if (it == segments_.end()) {
        criticalError() << "Unknown segment handle " << segment;
    }
    return *it->second;
}

BatchedSearchBase::SegmentStateBase const& BatchedSearchBase::getSegmentBase(SegmentHandle segment) const {
    auto it = segments_.find(segment);
    if (it == segments_.end()) {
        criticalError() << "Unknown segment handle " << segment;
    }
    return *it->second;
}

BatchedSearchBase::SegmentStateBase* BatchedSearchBase::findSegment(SegmentHandle segment) {
    auto it = segments_.find(segment);
    return it == segments_.end() ? nullptr : it->second.get();
}

void BatchedSearchBase::finishSegmentSearch(SegmentStateBase& segment) {
    verify(not segment.finished);
    segment.finished = true;
    --numActiveSegments_;

    logSegmentStatistics(segment);

    releaseSegmentState(segment);
    auto lock = lockLabelScorers();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->removeSegment(segment.handle);
    }
}

std::unique_lock<std::mutex> BatchedSearchBase::lockLabelScorers() {
    return std::unique_lock<std::mutex>(labelScorerMutex_);
}

BatchedSearchBase::ScoreAccessors BatchedSearchBase::getScoreAccessors(size_t scorerIdx, std::vector<Nn::ScoringContextRef> const& contexts) {
    numContextsPerScoringCall_ += contexts.size();
    auto lock = lockLabelScorers();
    scoringTime_.start();
    auto scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(contexts);
    scoringTime_.stop();
    return scoreAccessors;
}

std::optional<Nn::ScoreAccessorRef> BatchedSearchBase::getScoreAccessor(size_t scorerIdx, Nn::ScoringContextRef const& context) {
    auto                   lock = lockLabelScorers();
    Core::StopWatch::Scope timer(scoringTime_);
    return labelScorers_[scorerIdx]->getScoreAccessor(context);
}

void BatchedSearchBase::cleanupLabelScorerCaches(size_t scorerIdx, Core::CollapsedVector<Nn::ScoringContextRef> const& activeContexts) {
    if (activeContexts.size() > 0ul) {
        auto lock = lockLabelScorers();
        labelScorers_[scorerIdx]->cleanupCaches(activeContexts);
    }
}

size_t BatchedSearchBase::numPipelineGroups() const {
    return numPipelineGroups_;
}

void BatchedSearchBase::startScoring(size_t group, std::vector<SegmentHandle> segments, std::vector<Nn::ScoringContextRef> contexts) {
    verify(scoringThread_);
    verify(not pendingScorings_[group]);
    numContextsPerScoringCall_ += contexts.size();
    // `std::function` must be copyable, so the task is shared
    auto task               = std::make_shared<std::packaged_task<ScoreAccessors()>>([this, contexts = std::move(contexts)]() {
        auto lock = lockLabelScorers();
        scoringTime_.start();
        auto scoreAccessors = labelScorers_.front()->getScoreAccessors(contexts);
        scoringTime_.stop();
        return scoreAccessors;
    });
    pendingScorings_[group] = PendingScoring{std::move(segments), task->get_future()};
    scoringThread_->submit([task]() { (*task)(); });
}

std::optional<BatchedSearchBase::PendingScoring> BatchedSearchBase::takePendingScoring(size_t group) {
    std::optional<PendingScoring> pending;
    std::swap(pending, pendingScorings_[group]);
    return pending;
}

BatchedSearchBase::ScoreAccessors BatchedSearchBase::waitForScoring(PendingScoring& pending) {
    scoringWaitTime_.start();
    auto scoreAccessors = pending.scoreAccessors.get();
    scoringWaitTime_.stop();
    return scoreAccessors;
}

size_t BatchedSearchBase::numSearchThreads() const {
    return numSearchThreads_;
}

void BatchedSearchBase::disableParallelSearch(std::string const& reason) {
    if (numSearchThreads_ > 1ul) {
        warning() << "Use a single search thread instead of " << numSearchThreads_ << " since " << reason;
        numSearchThreads_ = 1ul;
        searchThreadPool_.reset();
        threadBusyTimes_.assign(1ul, ThreadBusyTime());
    }
}

void BatchedSearchBase::logGlobalStatistics() {
    verify(numActiveSegments_ == 0ul);
    if (numSegmentsPerStep_.nObservations() == 0u) {
        return;
    }

    auto lock = lockLabelScorers();

    if (statisticsChannel_.isOpen()) {
        auto& channel = statisticsChannel_;
        channel << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
        channel << Core::XmlFull("initialization-time", initializationTime_.elapsedMilliseconds());
        channel << Core::XmlFull("feature-processing-time", featureProcessingTime_.elapsedMilliseconds());
        channel << Core::XmlFull("scoring-time", scoringTime_.elapsedMilliseconds());
        if (numPipelineGroups_ > 1ul) {
            channel << Core::XmlFull("scoring-wait-time", scoringWaitTime_.elapsedMilliseconds());
        }
        channel << Core::XmlOpen("search-step-time") + Core::XmlAttribute("total", searchStepTime_.elapsedMilliseconds());
        logSearchStepBreakdown();
        channel << Core::XmlClose("search-step-time");
        // Parallel part of the search steps: wall-clock time and the time the search threads were busy or idle in it
        double busyTime = 0.0;
        for (auto const& threadBusyTime : threadBusyTimes_) {
            busyTime += threadBusyTime.time.count();
        }
        double parallelTime = parallelSearchTime_.count();
        double idleTime     = std::max(0.0, numSearchThreads() * parallelTime - busyTime);
        channel << Core::XmlFull("parallel-search-time", 1000.0 * parallelTime) + Core::XmlAttribute("threads", numSearchThreads());
        channel << Core::XmlFull("search-thread-busy-time", 1000.0 * busyTime);
        channel << Core::XmlFull("search-thread-idle-time", 1000.0 * idleTime);
        channel << Core::XmlClose("timing-statistics");
        channel << Core::XmlOpen("batching-statistics");
        numSegmentsPerStep_.write(channel);
        numContextsPerScoringCall_.write(channel);
        channel << Core::XmlClose("batching-statistics");
    }

    // No segment is registered in the label scorers anymore, so `reset` only resets their statistics
    for (auto& labelScorer : labelScorers_) {
        labelScorer->logStatistics();
        labelScorer->reset();
    }
    logAdditionalStatistics();

    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    scoringWaitTime_.reset();
    searchStepTime_.reset();
    resetSearchStepBreakdown();
    parallelSearchTime_ = std::chrono::duration<double>(0.0);
    for (auto& threadBusyTime : threadBusyTimes_) {
        threadBusyTime.time = std::chrono::duration<double>(0.0);
    }
    numSegmentsPerStep_.clear();
    numContextsPerScoringCall_.clear();
}

void BatchedSearchBase::waitForAllPendingScorings() {
    for (auto& pending : pendingScorings_) {
        if (pending) {
            pending->scoreAccessors.wait();
        }
    }
}

}  // namespace Search
