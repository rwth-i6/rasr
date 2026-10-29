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

#ifndef BATCHED_SEARCH_BASE_HH
#define BATCHED_SEARCH_BASE_HH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include <Core/Parameter.hh>
#include <Core/Statistics.hh>
#include <Core/StopWatch.hh>
#include <Core/ThreadPool.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Search/SearchV3.hh>

namespace Search {

/*
 * Common segment handling of multi-segment search algorithms based on label scorers:
 *  - Segment handles and the per-segment search state (derived from `SegmentStateBase`)
 *  - Forwarding of segments and their features to the label scorers (the handle is used as label scorer segment id)
 *  - Finishing segments (releasing their data in the label scorers) and removing them
 *  - Optionally pipelining (`num-pipeline-groups`): the active segments are split into groups that take turns, so that
 *    the label scorer call of one group runs in a background thread while the search work of another group is done
 *  - Timing and batching statistics accumulated over all segments
 * The derived search algorithm creates the search state of new segments and implements the actual search
 * (`decodeGroupStep` and the result functions).
 *
 * Thread-safety of the label scorers: all label scorer calls except `extendedScoringContext` and the score accessor
 * functions must hold the label scorer lock (`lockLabelScorers`, done by the functions of this class). The exceptions
 * may be called in parallel for different segments, also while a (locked) label scorer call for other segments runs.
 */
class BatchedSearchBase : public SearchAlgorithmV3 {
public:
    static const Core::ParameterInt paramMaxActiveSegments;
    static const Core::ParameterInt paramNumSearchThreads;
    static const Core::ParameterInt paramNumPipelineGroups;

    BatchedSearchBase(Core::Configuration const& config);
    virtual ~BatchedSearchBase();

    size_t                     maxActiveSegments() const override;
    size_t                     numActiveSegments() const override;
    SegmentHandle              addSegment(std::string const& name = "") override;
    void                       signalNoMoreFeatures(SegmentHandle segment) override;
    void                       putFeature(SegmentHandle segment, Nn::DataView const& feature) override;
    void                       putFeatures(SegmentHandle segment, Nn::DataView const& features, size_t nTimesteps) override;
    bool                       isFinished(SegmentHandle segment) const override;
    void                       removeSegment(SegmentHandle segment) override;
    std::vector<SegmentHandle> segments() const override;

    // Performs a search step for the next pipeline group that can make progress (all segments without pipelining)
    bool decodeStep() override;

protected:
    /*
     * Search state of a single segment. Derived search algorithms extend it by their beam etc.
     */
    struct SegmentStateBase {
        SegmentHandle handle;
        std::string   name;
        size_t        totalTimesteps;            // Number of features passed so far
        bool          noMoreFeatures;            // `signalNoMoreFeatures` was called
        bool          finished;                  // The search of the segment is finished
        size_t        group;                     // Pipeline group of the segment
        bool          noMoreFeaturesWhenScored;  // `noMoreFeatures` at the time the contexts of the current step were scored
                                                 // (if a step fails because of missing scores, the segment can only be finished if no more features were expected at that time)

        SegmentStateBase(SegmentHandle handle, std::string const& name);
        virtual ~SegmentStateBase() = default;
    };

    using ScoreAccessors = std::vector<std::optional<Nn::ScoreAccessorRef>>;

    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers_;

    // Statistics of each segment and timing statistics of all segments. Disabled unless a target is configured.
    mutable Core::XmlChannel statisticsChannel_;
    // Statistics for every search step. Disabled unless a target is configured. Disables the parallel search work.
    Core::XmlChannel stepwiseStatisticsChannel_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;      // Label scorer calls (also those in the background)
    Core::StopWatch scoringWaitTime_;  // Waiting for label scorer calls in the background
    Core::StopWatch searchStepTime_;

    /*
     * Time spent in `parallelForSegments` (wall-clock) and the time each search thread was busy with segments in it.
     * The difference of `numSearchThreads() * parallelSearchTime_` and the summed busy times is the time the threads
     * were idle, e.g. waiting for the slowest segment of a step or because a step has fewer segments than threads.
     */
    struct alignas(64) ThreadBusyTime {  // One cache line per thread
        std::chrono::duration<double> time{0.0};
    };
    std::chrono::duration<double> parallelSearchTime_;
    std::vector<ThreadBusyTime>   threadBusyTimes_;

    Core::Statistics<u32> numSegmentsPerStep_;
    Core::Statistics<u32> numContextsPerScoringCall_;
    size_t                globalSearchStep_;

    // Create the search state of a new segment. The segment is already registered in the label scorers.
    virtual std::unique_ptr<SegmentStateBase> createSegmentState(SegmentHandle handle, std::string const& name) = 0;

    /*
     * Perform a search step for the segments of the given pipeline group (all segments are in group 0 without
     * pipelining). Returns whether anything changed. With pipelining, the derived search
     *  - uses the scores of the pending label scorer call of the group (see `takePendingScoring`) for the segments
     *    whose contexts were scored in it, other segments of the group take part in the next step of the group,
     *  - starts the label scorer call of the first label scorer for the next step of the group at the end of the step
     *    (`startScoring`).
     */
    virtual bool decodeGroupStep(size_t group) = 0;

    // Log the statistics of a segment once its search is finished
    virtual void logSegmentStatistics(SegmentStateBase const& segment) const = 0;

    // Release data of a finished segment that is not needed for the results anymore (e.g. scoring contexts)
    virtual void releaseSegmentState(SegmentStateBase& segment) {}

    // Store the label scorers and warn about label scorers that can only handle one segment at a time
    void setLabelScorers(std::vector<Core::Ref<Nn::LabelScorer>> const& labelScorers);

    SegmentStateBase&       getSegmentBase(SegmentHandle segment);
    SegmentStateBase const& getSegmentBase(SegmentHandle segment) const;

    template<typename SegmentState>
    SegmentState& getSegment(SegmentHandle segment) {
        return static_cast<SegmentState&>(getSegmentBase(segment));
    }

    template<typename SegmentState>
    SegmentState const& getSegment(SegmentHandle segment) const {
        return static_cast<SegmentState const&>(getSegmentBase(segment));
    }

    // Call `f(SegmentState&)` for all segments in the order they were added
    template<typename SegmentState, typename Function>
    void forEachSegment(Function&& f) {
        for (auto& [handle, segment] : segments_) {
            f(static_cast<SegmentState&>(*segment));
        }
    }

    // Segment with the given handle or nullptr if there is none (e.g. because it was removed)
    SegmentStateBase* findSegment(SegmentHandle segment);

    // Mark the search of the segment as finished, log its statistics and release its data in the label scorers
    void finishSegmentSearch(SegmentStateBase& segment);

    // Lock that must be held for label scorer calls
    std::unique_lock<std::mutex> lockLabelScorers();

    // Synchronous label scorer calls (holding the label scorer lock)
    ScoreAccessors                      getScoreAccessors(size_t scorerIdx, std::vector<Nn::ScoringContextRef> const& contexts);
    std::optional<Nn::ScoreAccessorRef> getScoreAccessor(size_t scorerIdx, Nn::ScoringContextRef const& context);

    // Pass the given contexts (of all segments) to the label scorer for cache cleanup
    void cleanupLabelScorerCaches(size_t scorerIdx, Core::CollapsedVector<Nn::ScoringContextRef> const& activeContexts);

    size_t numPipelineGroups() const;

    // Label scorer call of a pipeline group that runs in the background
    struct PendingScoring {
        std::vector<SegmentHandle>  segments;  // Segments whose contexts are scored
        std::future<ScoreAccessors> scoreAccessors;
    };

    // Start the label scorer call of the first label scorer for the given contexts of the given segments of a
    // pipeline group in the background
    void startScoring(size_t group, std::vector<SegmentHandle> segments, std::vector<Nn::ScoringContextRef> contexts);

    // Take the pending label scorer call of the group (if any); `waitForScoring` returns its result
    std::optional<PendingScoring> takePendingScoring(size_t group);
    ScoreAccessors                waitForScoring(PendingScoring& pending);

    // Number of threads that run the per-segment search work in parallel
    size_t numSearchThreads() const;

    // Run the per-segment search work in a single thread from now on, e.g. because it writes stepwise log output
    void disableParallelSearch(std::string const& reason);

    /*
     * Runs the per-segment search work of a step (e.g. scoring and pruning extensions, creating the new beam) in
     * parallel for the segments, while the label scorer calls stay batched over all segments.
     *
     * Call `f(segment, threadIdx)` for all given segments. With several search threads, the segments are processed in
     * parallel. `threadIdx < numSearchThreads()` identifies the executing thread (e.g. for per-thread buffers), all calls
     * with the same `threadIdx` are made one after another. Returns once all calls are done.
     * `f` may only modify the given segment and the buffers of its thread. Label scorers may only be used for functions
     * that are safe to call in parallel for different segments (`extendedScoringContext`, score accessors).
     * Logging and finishing segments must happen outside of `f`.
     */
    template<typename SegmentState, typename Function>
    void parallelForSegments(std::vector<SegmentState*> const& segments, Function&& f) {
        auto start = std::chrono::steady_clock::now();
        // Each thread takes the next unprocessed segment until none are left
        // The calling thread is thread 0
        std::atomic<size_t> nextSegment(0ul);
        auto                work = [&](size_t threadIdx) {
            auto workStart = std::chrono::steady_clock::now();
            for (size_t i = nextSegment.fetch_add(1ul); i < segments.size(); i = nextSegment.fetch_add(1ul)) {
                f(*segments[i], threadIdx);
            }
            threadBusyTimes_[threadIdx].time += std::chrono::steady_clock::now() - workStart;
        };
        size_t numThreads = std::min(numSearchThreads_, segments.size());
        for (size_t threadIdx = 1ul; threadIdx < numThreads; ++threadIdx) {
            searchThreadPool_->submit([&work, threadIdx]() { work(threadIdx); });
        }
        work(0ul);
        if (numThreads > 1ul) {
            searchThreadPool_->wait();
        }
        parallelSearchTime_ += std::chrono::steady_clock::now() - start;
    }

    // Log the timing and batching statistics accumulated over all segments and the statistics of the label scorers
    // and reset them. Only allowed when no segment is active.
    void logGlobalStatistics();

    // Hooks for the derived searches to break down `search-step-time` (called inside that element of the statistics
    // channel) and to reset these timers
    virtual void logSearchStepBreakdown() const {}
    virtual void resetSearchStepBreakdown() {}

    // Hook for the derived searches to log further statistics of all segments (called by `logGlobalStatistics`)
    virtual void logAdditionalStatistics() const {}

private:
    size_t                                                     maxActiveSegments_;
    std::map<SegmentHandle, std::unique_ptr<SegmentStateBase>> segments_;
    SegmentHandle                                              nextSegmentHandle_;
    size_t                                                     numActiveSegments_;

    using FunctionThreadPool = Core::ThreadPool<std::function<void()>, Core::FunctionMapper>;

    size_t                              numSearchThreads_;
    std::unique_ptr<FunctionThreadPool> searchThreadPool_;  // numSearchThreads_ - 1 threads, only if numSearchThreads_ > 1

    size_t                                     numPipelineGroups_;
    size_t                                     nextGroup_;
    std::mutex                                 labelScorerMutex_;
    std::vector<std::optional<PendingScoring>> pendingScorings_;  // One per pipeline group
    std::unique_ptr<FunctionThreadPool>        scoringThread_;    // One thread, only with pipelining. Declared last, so it is destroyed first

    // Wait until all label scorer calls in the background are done (their results stay available)
    void waitForAllPendingScorings();
};

}  // namespace Search

#endif  // BATCHED_SEARCH_BASE_HH
