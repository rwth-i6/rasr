#ifndef _LM_RECURRENT_LANGUAGE_MODEL_HH
#define _LM_RECURRENT_LANGUAGE_MODEL_HH

#include <deque>
#include <future>
#include <thread>
#include <vector>

#include <Math/FastMatrix.hh>
#include <Core/readerwriterqueue.h>

#include "AbstractNNLanguageModel.hh"
#include "AbstractStateManager.hh"
#include "Module.hh"
#include "SearchSpaceAwareLanguageModel.hh"

namespace Lm {
template<typename value_t, typename state_variable_t>
class RecurrentLanguageModel;

namespace detail {

struct ScoresWithContext : public Lm::NNCacheWithStats {
    virtual ~ScoresWithContext() = default;

    std::atomic<bool>                           computed;
    Lm::History                                 parent;
    Lm::CompressedVectorPtr<float>              nn_output;
    std::vector<Lm::CompressedVectorPtr<float>> state;
    Lm::SearchSpaceInformation                  info;
    Search::TimeframeIndex                      last_used;
    Search::TimeframeIndex                      last_info;
    bool                                        was_expanded;
};

struct FwdRequest {
    ScoresWithContext* initial_cache;
    ScoresWithContext* final_cache;
    size_t             length;

    bool operator==(FwdRequest const& other) const {
        return final_cache == other.final_cache;
    }
};

struct RequestGraph {
    std::vector<ScoresWithContext*>  entries;
    std::vector<std::vector<size_t>> children;
    std::vector<size_t>              roots;

    void                    add_cache(ScoresWithContext* cache);
    void                    get_requests_dfs(std::vector<FwdRequest>& requests, ScoresWithContext* initial, size_t entry, size_t length) const;
    std::vector<FwdRequest> get_requests() const;
};

void dump_scores(ScoresWithContext const& cache, std::string const& prefix);

template<typename value_t, typename state_variable_t>
void clear_queue(typename Lm::RecurrentLanguageModel<value_t, state_variable_t>::HistoryQueue& queue) {
    Lm::History const* hist = nullptr;
    while (queue.try_dequeue(hist)) {
        delete hist;
    }
}

struct TimeStatistics {
    std::chrono::duration<double, std::milli> total_duration;
    std::chrono::duration<double, std::milli> early_request_duration;
    std::chrono::duration<double, std::milli> request_duration;
    std::chrono::duration<double, std::milli> prepare_duration;
    std::chrono::duration<double, std::milli> merge_state_duration;
    std::chrono::duration<double, std::milli> set_state_duration;
    std::chrono::duration<double, std::milli> run_nn_output_duration;
    std::chrono::duration<double, std::milli> set_nn_output_duration;
    std::chrono::duration<double, std::milli> get_new_state_duration;
    std::chrono::duration<double, std::milli> split_state_duration;
    std::chrono::duration<double, std::milli> softmax_output_duration;

    TimeStatistics  operator+(TimeStatistics const& other) const;
    TimeStatistics& operator+=(TimeStatistics const& other);
    void            write(Core::XmlChannel& channel) const;
    void            write(std::ostream& out) const;
};

}  // namespace detail

template<typename value_t, typename state_variable_t>
class RecurrentLanguageModel : public AbstractNNLanguageModel, public SearchSpaceAwareLanguageModel {
public:
    typedef AbstractNNLanguageModel                               Precursor;
    typedef moodycamel::BlockingReaderWriterQueue<History const*> HistoryQueue;

    static const Core::ParameterBool   paramTransformOuputLog;
    static const Core::ParameterBool   paramTransformOuputNegate;
    static const Core::ParameterInt    paramMinBatchSize;
    static const Core::ParameterInt    paramOptBatchSize;
    static const Core::ParameterInt    paramMaxBatchSize;
    static const Core::ParameterInt    paramHistoryPruningThreshold;
    static const Core::ParameterInt    paramPrunedHistoryLength;
    static const Core::ParameterFloat  paramBatchPruningThreshold;
    static const Core::ParameterBool   paramAllowReducedHistory;
    static const Core::ParameterBool   paramDumpInputs;
    static const Core::ParameterString paramDumpInputsPrefix;
    static const Core::ParameterBool   paramDumpScores;
    static const Core::ParameterString paramDumpScoresPrefix;
    static const Core::ParameterBool   paramLogMemory;
    static const Core::ParameterBool   paramFreeMemory;
    static const Core::ParameterInt    paramFreeMemoryDelay;
    static const Core::ParameterBool   paramAsync;
    static const Core::ParameterBool   paramSingleStepOnly;
    static const Core::ParameterBool   paramVerbose;

    RecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l, std::unique_ptr<AbstractStateManager<value_t, state_variable_t>> state_manager);
    virtual ~RecurrentLanguageModel();

    virtual History startHistory() const;
    virtual History extendedHistory(History const& hist, Token w) const;
    virtual History extendedHistory(History const& hist, Bliss::Token::Id w) const;
    virtual History reducedHistory(History const& hist, u32 limit) const;
    virtual History reduceHistoryByN(History const&, u32 n) const;
    virtual Score   score(History const& hist, Token w) const;
    virtual bool    scoreCached(History const& hist, Token w) const;

    virtual void startFrame(Search::TimeframeIndex time) const;
    virtual void setInfo(History const& hist, SearchSpaceInformation const& info) const;

protected:
    virtual void load();

    virtual void                 setState(std::vector<std::pair<std::string, value_t>> const& inputs, std::vector<std::string> const& targets) const                                                                            = 0;
    virtual void                 extendInputs(std::vector<std::pair<std::string, value_t>>& inputs, Math::FastMatrix<s32> const& words, Math::FastVector<s32> const& word_lengths, std::vector<s32> const& state_lengths) const = 0;
    virtual void                 extendTargets(std::vector<std::string>& targets) const                                                                                                                                         = 0;
    virtual void                 getOutputs(std::vector<std::pair<std::string, value_t>>& inputs, std::vector<value_t>& outputs, std::vector<std::string> const& targets) const                                                 = 0;
    virtual std::vector<value_t> fetchStates(std::vector<value_t>& outputs) const                                                                                                                                               = 0;
    virtual Score                transformOutput(Lm::CompressedVectorPtr<float> const& nn_output, size_t index) const                                                                                                           = 0;

    std::vector<state_variable_t> state_variables_;

    void setEmptyHistory();

private:
    using ScoresWithContext = detail::ScoresWithContext;

    bool                   transform_output_log_;
    bool                   transform_output_negate_;
    size_t                 min_batch_size_;
    size_t                 opt_batch_size_;
    size_t                 max_batch_size_;
    size_t                 history_pruning_threshold_;
    size_t                 pruned_history_length_;
    Score                  batch_pruning_threshold_;
    bool                   allow_reduced_history_;
    bool                   dump_inputs_;
    std::string            dump_inputs_prefix_;
    bool                   dump_scores_;
    std::string            dump_scores_prefix_;
    bool                   log_memory_;
    bool                   free_memory_;
    Search::TimeframeIndex free_memory_delay_;
    bool                   single_step_only_;
    bool                   verbose_;

    mutable Core::XmlChannel       statistics_;
    mutable Search::TimeframeIndex current_time_;
    mutable std::vector<double>    run_time_;
    mutable std::vector<size_t>    run_count_;
    mutable double                 total_wait_time_;
    mutable double                 total_start_frame_time_;
    mutable double                 total_expand_hist_time_;
    mutable detail::TimeStatistics fwd_statistics_;
    mutable size_t                 dump_inputs_counter_;

    std::unique_ptr<AbstractStateManager<value_t, state_variable_t>> state_manager_;

    std::function<Score(Score)>       output_transform_function_;
    CompressedVectorFactoryPtr<float> state_comp_vec_factory_;
    CompressedVectorFactoryPtr<float> nn_output_comp_vec_factory_;

    History empty_history_;  // a history used to provide the previous (all zero) state to the first real history (1 sentence-begin token)

    // members for async forwarding
    bool        should_stop_;
    std::thread background_forwarder_thread_;
    bool        async_;

    void background_forward() const;

    mutable std::atomic<History const*>  to_fwd_;
    mutable std::promise<History const*> to_fwd_finished_;

    mutable std::vector<History const*> pending_;
    mutable HistoryQueue                fwd_queue_;
    mutable HistoryQueue                finished_queue_;

    History extendHistoryWithOutputIdx(History const& hist, size_t w) const;

    template<bool async>
    void forward(Lm::History const* hist) const;
};

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramTransformOuputLog("transform-output-log", "apply log to tensorflow output", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramTransformOuputNegate("transform-output-negate", "negate tensorflow output (after log)", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramMinBatchSize("min-batch-size", "minimum number of histories forwarded in one go", 32);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramOptBatchSize("opt-batch-size", "optimum number of histories forwarded in one go", 128);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramMaxBatchSize("max-batch-size", "maximum number of histories forwarded in one go", 2048);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramHistoryPruningThreshold("history-pruning-threshold", "if the history is longer than this parameter it will be pruned", std::numeric_limits<int>::max(), 0);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramPrunedHistoryLength("pruned-history-length", "length of the pruned history (should be smaller than history-pruning-threshold)", std::numeric_limits<int>::max(), 0);

template<typename value_t, typename state_variable_t>
const Core::ParameterFloat RecurrentLanguageModel<value_t, state_variable_t>::paramBatchPruningThreshold("batch-pruning-threshold", "pruning threshold for all hypothesis beyond min-batch-size during eager forwarding", 10.0);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramAllowReducedHistory("allow-reduced-history", "wether this LM will actually reduce the history length", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramDumpInputs("dump-inputs", "write all inputs from this LM to disk", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterString RecurrentLanguageModel<value_t, state_variable_t>::paramDumpInputsPrefix("dump-inputs-prefix", "prefix for the input dumps", "inputs");

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramDumpScores("dump-scores", "write all scores from this LM to disk", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterString RecurrentLanguageModel<value_t, state_variable_t>::paramDumpScoresPrefix("dump-scores-prefix", "prefix for the score dumps", "scores");

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramLogMemory("log-memory", "wether memory usage from nn-outputs / states should be logged", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramFreeMemory("free-memory", "wether nn-outputs should be deleted after some delay", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterInt RecurrentLanguageModel<value_t, state_variable_t>::paramFreeMemoryDelay("free-memory-delay", "how many time frames without usage before nn-outputs are deleted", 40);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramAsync("async", "wether to forward histories in a separate thread", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramSingleStepOnly("single-step-only", "workaround for some bug that results in wrong scores when recombination is done in combination with async evaluation", false);

template<typename value_t, typename state_variable_t>
const Core::ParameterBool RecurrentLanguageModel<value_t, state_variable_t>::paramVerbose("verbose", "wether to print detailed statistics to stderr", false);

template<typename value_t, typename state_variable_t>
RecurrentLanguageModel<value_t, state_variable_t>::RecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l, std::unique_ptr<AbstractStateManager<value_t, state_variable_t>> state_manager)
        : Core::Component(c),
          Precursor(c, l),
          transform_output_log_(paramTransformOuputLog(config)),
          transform_output_negate_(paramTransformOuputNegate(config)),
          min_batch_size_(paramMinBatchSize(config)),
          opt_batch_size_(paramOptBatchSize(config)),
          max_batch_size_(paramMaxBatchSize(config)),
          history_pruning_threshold_(paramHistoryPruningThreshold(config)),
          pruned_history_length_(paramPrunedHistoryLength(config)),
          batch_pruning_threshold_(paramBatchPruningThreshold(config)),
          allow_reduced_history_(paramAllowReducedHistory(config)),
          dump_inputs_(paramDumpInputs(config)),
          dump_inputs_prefix_(paramDumpInputsPrefix(config)),
          dump_scores_(paramDumpScores(config)),
          dump_scores_prefix_(paramDumpScoresPrefix(config)),
          log_memory_(paramLogMemory(config)),
          free_memory_(paramFreeMemory(config)),
          free_memory_delay_(paramFreeMemoryDelay(config)),
          single_step_only_(paramSingleStepOnly(config)),
          verbose_(paramVerbose(config)),
          statistics_(config, "statistics"),
          current_time_(0u),
          run_time_(max_batch_size_, 0.0),
          run_count_(max_batch_size_, 0ul),
          total_wait_time_(0.0),
          total_start_frame_time_(0.0),
          total_expand_hist_time_(0.0),
          fwd_statistics_(),
          dump_inputs_counter_(0ul),
          state_manager_(std::move(state_manager)),
          output_transform_function_(),
          state_comp_vec_factory_(Lm::Module::instance().createCompressedVectorFactory(select("state-compression"))),
          nn_output_comp_vec_factory_(Lm::Module::instance().createCompressedVectorFactory(select("nn-output-compression"))),
          empty_history_(),
          should_stop_(false),
          background_forwarder_thread_(),
          async_(paramAsync(config)),
          to_fwd_(nullptr),
          to_fwd_finished_(),
          pending_(),
          fwd_queue_(32768),
          finished_queue_(32768) {
    if (transform_output_log_ and transform_output_negate_) {
        output_transform_function_ = [](Score v) {
            return -std::log(v);
        };
    }
    else if (transform_output_log_) {
        output_transform_function_ = [](Score v) {
            return std::log(v);
        };
    }
    else if (transform_output_negate_) {
        output_transform_function_ = [](Score v) {
            return -v;
        };
    }

    if (async_) {
        background_forwarder_thread_ = std::thread(std::bind(&RecurrentLanguageModel<value_t, state_variable_t>::background_forward, this));
    }

    require_le(pruned_history_length_, history_pruning_threshold_);
}

template<typename value_t, typename state_variable_t>
RecurrentLanguageModel<value_t, state_variable_t>::~RecurrentLanguageModel() {
    detail::clear_queue<value_t, state_variable_t>(finished_queue_);

    if (async_) {
        should_stop_ = true;
        background_forwarder_thread_.join();
    }

    size_t total_run_count = 0ul;
    size_t total_fwd_hist  = 0ul;
    double total_run_time  = 0.0;

    statistics_ << Core::XmlOpen("fwd-time");
    for (size_t i = 0ul; i < run_count_.size(); i++) {
        if (run_count_[i] > 0ul) {
            statistics_ << (i + 1) << " " << run_count_[i] << " " << run_time_[i] << "\n";
            total_run_count += run_count_[i];
            total_fwd_hist += (i + 1) * run_count_[i];
            total_run_time += run_time_[i];
        }
    }
    statistics_ << Core::XmlClose("fwd-time");

    statistics_ << Core::XmlOpen("fwd-summary");
    statistics_ << Core::XmlOpen("total-run-count") << total_run_count << Core::XmlClose("total-run-count");
    statistics_ << Core::XmlOpen("total-fwd-hist") << total_fwd_hist << Core::XmlClose("total-fwd-hist");
    statistics_ << Core::XmlOpen("total-run-time") + Core::XmlAttribute("unit", "milliseconds") << total_run_time << Core::XmlClose("total-run-time");
    statistics_ << Core::XmlOpen("total-wait-time") + Core::XmlAttribute("unit", "milliseconds") << total_wait_time_ << Core::XmlClose("total-wait-time");
    statistics_ << Core::XmlOpen("total-start-frame-time") + Core::XmlAttribute("unit", "milliseconds") << total_start_frame_time_ << Core::XmlClose("total-start-frame-time");
    statistics_ << Core::XmlOpen("total-expand-hist-time") + Core::XmlAttribute("unit", "milliseconds") << total_expand_hist_time_ << Core::XmlClose("total-expand-hist-time");
    statistics_ << Core::XmlOpen("fwd-times");
    fwd_statistics_.write(statistics_);
    statistics_ << Core::XmlClose("fwd-times");
    statistics_ << Core::XmlClose("fwd-summary");
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::startHistory() const {
    NNHistoryManager*  hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence    ts(1ul, lexicon_mapping_[sentenceBeginToken()->id()]);
    HistoryHandle      h     = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->parent            = empty_history_;
    History hist(history(h));
    return hist;
}

template<typename value_t, typename state_variable_t>
void RecurrentLanguageModel<value_t, state_variable_t>::setEmptyHistory() {
    NNHistoryManager*  hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence    ts;
    HistoryHandle      h     = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->state             = state_manager_->initialState(state_variables_, *state_comp_vec_factory_);

    if (cache->state.empty()) {
        error("LM has no state variables. Did you forget to compile with 'initial_state': 'keep_over_epoch_no_init' for TensorFlow or 'initial_state': 'placeholder' for Onnx?");
    }

    std::vector<f32> temp(1);
    auto             compression_param_estimator = nn_output_comp_vec_factory_->getEstimator();
    compression_param_estimator->accumulate(temp.data(), temp.size());
    auto compression_params = compression_param_estimator->estimate();
    // pretend this history has already been evaluated
    cache->nn_output = nn_output_comp_vec_factory_->compress(temp.data(), temp.size(), compression_params.get());
    cache->computed.store(true);
    cache->last_used = std::numeric_limits<Search::TimeframeIndex>::max();
    empty_history_   = history(h);
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::extendedHistory(History const& hist, Token w) const {
    return extendedHistory(hist, w->id());
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::extendedHistory(History const& hist, Bliss::Token::Id w) const {
    return extendHistoryWithOutputIdx(hist, lexicon_mapping_[w]);
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::reducedHistory(History const& hist, u32 limit) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    if (not allow_reduced_history_ or sc->history->size() <= limit) {
        return hist;
    }
    History h = startHistory();
    for (u32 w = limit; w > 0; w--) {
        h = extendHistoryWithOutputIdx(h, sc->history->at(sc->history->size() - w));
    }
    return h;
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::reduceHistoryByN(History const& hist, u32 n) const {
    if (not allow_reduced_history_) {
        return hist;
    }
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    History                  h  = startHistory();
    for (u32 w = n; w < sc->history->size(); w++) {
        h = extendHistoryWithOutputIdx(h, sc->history->at(w));
    }
    return h;
}

template<typename value_t, typename state_variable_t>
Score RecurrentLanguageModel<value_t, state_variable_t>::score(History const& hist, Token w) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));

    if (not sc->computed.load()) {
        auto start = std::chrono::steady_clock::now();
        if (async_) {
            // promise should only be used once
            to_fwd_finished_                   = std::promise<History const*>();
            std::future<History const*> future = to_fwd_finished_.get_future();
            to_fwd_.store(&hist);
            future.wait();
        }
        else {
            forward<false>(&hist);
        }
        auto   end       = std::chrono::steady_clock::now();
        double wait_time = std::chrono::duration<double, std::milli>(end - start).count();
        total_wait_time_ += wait_time;
        if (verbose_) {
            std::cerr << "wait: " << wait_time << " " << sc->info.numStates << " " << sc->info.bestScoreOffset << std::endl;
        }
    }

    require(sc->computed.load());

    size_t output_idx = lexicon_mapping_[w->id()];
    useOutput(*sc, output_idx);
    sc->last_used  = current_time_;
    auto  start    = std::chrono::steady_clock::now();
    Score score    = output_transform_function_(transformOutput(sc->nn_output, output_idx));
    auto  end      = std::chrono::steady_clock::now();
    auto  duration = std::chrono::duration<double, std::milli>(end - start);
    fwd_statistics_.softmax_output_duration += duration;
    fwd_statistics_.total_duration += duration;
    return score;
}

template<typename value_t, typename state_variable_t>
bool RecurrentLanguageModel<value_t, state_variable_t>::scoreCached(History const& hist, Token w) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    return sc->computed.load();
}

template<typename value_t, typename state_variable_t>
void RecurrentLanguageModel<value_t, state_variable_t>::load() {
    loadVocabulary();
}

template<typename value_t, typename state_variable_t>
void RecurrentLanguageModel<value_t, state_variable_t>::startFrame(Search::TimeframeIndex time) const {
    auto timer_start = std::chrono::steady_clock::now();

    current_time_ = time;

    size_t nn_output_cache_size = 0ul;
    size_t state_cache_size     = 0ul;
    size_t num_histories        = 0ul;

    detail::clear_queue<value_t, state_variable_t>(finished_queue_);

    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    hm->visit([&](HistoryHandle h) {
        num_histories += 1ul;
        ScoresWithContext* c        = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
        bool               computed = c->computed.load();
        if (free_memory_ and computed and c->was_expanded and c->info.numStates == 0 and c->last_used < current_time_ - std::min(free_memory_delay_, current_time_)) {
            c->nn_output->clear();
            c->computed.store(false);
        }
        else if (async_ and not computed and not(c->was_expanded and c->info.numStates == 0)) {
            fwd_queue_.enqueue(new History(history(h)));
        }
        if (c->nn_output) {
            nn_output_cache_size += c->nn_output->usedMemory();
        }
        for (auto const& state_vec : c->state) {
            if (state_vec) {
                state_cache_size += state_vec->usedMemory();
            }
        }
    });

    if (log_memory_ and statistics_.isOpen()) {
        statistics_ << Core::XmlOpen("memory-usage") + Core::XmlAttribute("time-frame", current_time_);
        statistics_ << Core::XmlOpen("nn-output-cache-size") + Core::XmlAttribute("unit", "MB") << (nn_output_cache_size / (1024. * 1024.)) << Core::XmlClose("nn-output-cache-size");
        statistics_ << Core::XmlOpen("state-cache-size") + Core::XmlAttribute("unit", "MB") << (state_cache_size / (1024. * 1024.)) << Core::XmlClose("state-cache-size");
        statistics_ << Core::XmlOpen("num-histories") << num_histories << Core::XmlClose("num-histories");
        statistics_ << Core::XmlClose("memory-usage");
    }

    auto   timer_end            = std::chrono::steady_clock::now();
    double start_frame_duration = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    total_start_frame_time_ += start_frame_duration;
}

template<typename value_t, typename state_variable_t>
void RecurrentLanguageModel<value_t, state_variable_t>::setInfo(History const& hist, SearchSpaceInformation const& info) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
    sc->info              = info;
    sc->last_info         = current_time_;
}

template<typename value_t, typename state_variable_t>
History RecurrentLanguageModel<value_t, state_variable_t>::extendHistoryWithOutputIdx(History const& hist, size_t w) const {
    auto                     timer_start = std::chrono::steady_clock::now();
    NNHistoryManager*        hm          = dynamic_cast<NNHistoryManager*>(historyManager_);
    ScoresWithContext const* sc          = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    TokenIdSequence          ts(*sc->history);
    ts.push_back(w);
    HistoryHandle      h     = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    if (cache->parent.handle() == nullptr) {
        cache->parent                   = hist;
        ScoresWithContext* parent_cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
        parent_cache->was_expanded      = true;
        if (async_) {
            fwd_queue_.enqueue(new History(history(h)));
        }
    }
    History ext_hist(history(h));
    if (cache->history->size() > history_pruning_threshold_) {
        ext_hist = reducedHistory(ext_hist, pruned_history_length_);
    }
    auto   timer_end        = std::chrono::steady_clock::now();
    double expand_hist_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    total_expand_hist_time_ += expand_hist_time;
    return ext_hist;
}

template<typename value_t, typename state_variable_t>
void RecurrentLanguageModel<value_t, state_variable_t>::background_forward() const {
    while (not should_stop_) {
        forward<true>(to_fwd_.exchange(nullptr));
    }
    History const* h = nullptr;
    while (fwd_queue_.try_dequeue(h)) {
        finished_queue_.enqueue(h);
    }
    for (History const* h : pending_) {
        finished_queue_.enqueue(h);
    }
    pending_.clear();
}

template<typename value_t, typename state_variable_t>
template<bool async>
void RecurrentLanguageModel<value_t, state_variable_t>::forward(Lm::History const* hist) const {
    ScoresWithContext* sc = nullptr;
    if (hist != nullptr) {
        sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist->handle()));
    }
    if (async and sc != nullptr and sc->computed.load()) {  // nothing to do (only happens in async case)
        to_fwd_finished_.set_value(hist);
        return;
    }
    auto start = std::chrono::steady_clock::now();

    detail::RequestGraph request_graph;
    if (sc != nullptr) {
        request_graph.add_cache(const_cast<ScoresWithContext*>(sc));
    }

    std::vector<detail::FwdRequest> requests;
    std::vector<Lm::History>        request_histories;  // make sure none of the request caches go away while we compute the scores
    size_t                          max_length = 0ul;

    size_t                          num_pending_requests = pending_.size();
    std::unordered_set<void const*> handles;  // only relevant in async case
    handles.reserve(pending_.size());
    std::vector<ScoresWithContext*> early_requests;
    std::vector<History const*>     early_request_histories;  // make sure none of the request caches go away while we compute the scores (only relevant in async case)

    if (async) {
        auto process_hist = [&](History const* hist) {
            ScoresWithContext* c        = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist->handle()));
            ScoresWithContext* parent_c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(c->parent.handle()));
            if (handles.find(hist->handle()) == handles.end() and not c->computed.load() and c != sc and c->parent.handle() != nullptr and c->ref_count > 1 and (not single_step_only_ or parent_c->computed.load())) {
                early_requests.emplace_back(c);
                early_request_histories.emplace_back(hist);
                handles.insert(hist->handle());
            }
            else {
                finished_queue_.enqueue(hist);
            }
        };

        std::for_each(pending_.begin(), pending_.end(), process_hist);
        pending_.clear();

        History const* hist_buf = nullptr;
        bool           success  = false;
        bool           first    = true;
        do {
            if (first) {
                success = fwd_queue_.wait_dequeue_timed(hist_buf, 1000);
            }
            else {
                success = fwd_queue_.try_dequeue(hist_buf);
            }
            if (success) {
                process_hist(hist_buf);
                first = false;
            }
        } while (success);
    }
    else {
        NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
        hm->visit([&](HistoryHandle h) {
            ScoresWithContext* c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
            if (not c->computed.load() and c != sc and not(c->was_expanded and c->info.numStates == 0)) {
                early_requests.emplace_back(c);
            }
        });
    }

    size_t num_early_requests = early_requests.size();

    auto end_early_requests = std::chrono::steady_clock::now();

    if (async and sc == nullptr and early_requests.empty()) {
        // can only happen in async case
        return;
    }

    // because the scores can be updated while we are sorting we need to store them, so we get a consistent view
    std::vector<std::pair<unsigned, Score>> idxs;
    idxs.reserve(early_requests.size());
    for (size_t i = 0ul; i < early_requests.size(); i++) {
        idxs.emplace_back(i, early_requests[i]->info.minLabelDistance * 1000 + early_requests[i]->info.bestScoreOffset);
    }
    std::sort(idxs.begin(), idxs.end(), [](std::pair<size_t, Score> const& a, std::pair<size_t, Score> const& b) {
        return a.second < b.second;
    });

    for (auto idx : idxs) {
        request_graph.add_cache(early_requests[idx.first]);
    }

    // we do not need early_requests anymore
    early_requests.clear();
    idxs.clear();

    requests = request_graph.get_requests();

    // prune requests
    if (min_batch_size_ > 0ul and requests.size() > min_batch_size_) {
        size_t i         = min_batch_size_;
        Score  ref_score = requests.front().final_cache->info.bestScoreOffset + batch_pruning_threshold_;
        if (not Math::isinf(ref_score)) {
            while ((i + 1) < requests.size() and requests[i + 1].final_cache->info.bestScoreOffset <= ref_score) {
                i += 1ul;
            }
            requests.resize(i);
        }
    }

    if (min_batch_size_ > 0ul and opt_batch_size_ > 0ul and requests.size() > opt_batch_size_ + min_batch_size_) {
        requests.resize(opt_batch_size_);
    }
    if (max_batch_size_ > 0ul and requests.size() > max_batch_size_) {
        requests.resize(max_batch_size_);
    }

    Score worst_score = std::numeric_limits<Score>::min();
    for (auto const& r : requests) {
        max_length  = std::max(max_length, r.length);
        worst_score = std::max(worst_score, r.final_cache->info.bestScoreOffset);
    }

    auto end_requests = std::chrono::steady_clock::now();

    // prepare the data in Sprint Datastructures
    Math::FastMatrix<s32> words(requests.size(), max_length);
    Math::FastVector<s32> word_lengths(requests.size());
    for (size_t r = 0ul; r < requests.size(); r++) {
        auto&  history = *(requests[r].final_cache->history);
        size_t offset  = history.size() - requests[r].length;
        for (size_t w = 0u; w < requests[r].length; w++) {
            words.at(r, w) = static_cast<s32>(history[offset + w]);
        }
        for (size_t w = requests[r].length; w < max_length; w++) {
            words.at(r, w) = 0;
        }
        word_lengths[r]                  = requests[r].length;
        ScoresWithContext* initial_cache = requests[r].initial_cache;
        require(initial_cache != nullptr);
        require_eq(state_variables_.size(), initial_cache->state.size());
    }

    bool   full_prefix_required = state_manager_->requiresAllParentStates();
    size_t total_prefix_length  = 0ul;
    size_t total_suffix_length  = 0ul;

    std::vector<size_t> prefix_lengths(requests.size());
    std::vector<size_t> suffix_lengths(requests.size());
    for (size_t r = 0ul; r < requests.size(); r++) {
        prefix_lengths[r] = requests[r].initial_cache->history->size();
        suffix_lengths[r] = requests[r].length;
        total_prefix_length += prefix_lengths[r];
        total_suffix_length += suffix_lengths[r];
    }

    std::vector<typename AbstractStateManager<value_t, state_variable_t>::HistoryState const*> prefix_states(full_prefix_required ? total_prefix_length : requests.size());
    size_t                                                                                     current_offset = 0ul;
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* current_cache = requests[r].initial_cache;
        if (full_prefix_required) {
            size_t prefix_length = prefix_lengths[r];
            for (size_t i = 0ul; i < prefix_length; i++) {
                prefix_states[current_offset + prefix_length - i - 1] = &current_cache->state;
                current_cache                                         = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(current_cache->parent.handle()));
            }
            current_offset += prefix_length;
        }
        else {
            prefix_states[r] = &current_cache->state;
        }
    }

    auto end_prepare = std::chrono::steady_clock::now();

    // build tensors + set state variables
    std::vector<std::pair<std::string, value_t>> inputs;
    std::vector<std::string>                     targets;
    state_manager_->mergeStates(state_variables_, prefix_lengths, prefix_states, inputs, targets);

    std::vector<s32> state_lengths(prefix_lengths.begin(), prefix_lengths.end());

    if (dump_inputs_) {
        std::string out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_state_";
        for (size_t i = 0ul; i < inputs.size(); i++) {
            inputs[i].second.template save<s16>(out + std::to_string(i));
        }
    }

    auto end_merge_state = std::chrono::steady_clock::now();

    setState(inputs, targets);

    auto end_set_state = std::chrono::steady_clock::now();

    extendInputs(inputs, words, word_lengths, state_lengths);
    extendTargets(targets);

    std::vector<value_t> outputs;
    getOutputs(inputs, outputs, targets);

    if (dump_inputs_) {
        std::string out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_nn_in_";
        for (size_t i = 0ul; i < inputs.size(); i++) {
            inputs[i].second.template save<s32>(out + std::to_string(i));
        }
        out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_nn_out_";
        for (size_t i = 0ul; i < outputs.size(); i++) {
            outputs[i].template save<f32>(out + std::to_string(i));
        }
        dump_inputs_counter_ += 1ul;
    }

    auto end_nn_output = std::chrono::steady_clock::now();

    // store outputs in caches
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* cache = requests[r].final_cache;
        // only final cache get the states
        for (size_t w = requests[r].length; w > 0;) {
            --w;
            cache->last_used                         = current_time_;
            int          num_outputs                 = outputs[0ul].dimSize(2);
            auto         compression_param_estimator = nn_output_comp_vec_factory_->getEstimator();
            float const* data                        = outputs[0ul].template data<f32>(r, w, 0);
            compression_param_estimator->accumulate(data, num_outputs);
            auto compression_params = compression_param_estimator->estimate();
            cache->nn_output        = nn_output_comp_vec_factory_->compress(data, num_outputs, compression_params.get());
            cache->computed.store(true);
            cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(cache->parent.handle()));
        }
        require_eq(cache, requests[r].initial_cache);
    }

    auto end_set_nn_output = std::chrono::steady_clock::now();

    // fetch new values of state variables, needs to be done in separate Session::run call (for GPU devices)
    std::vector<value_t> state_vars = fetchStates(outputs);

    auto end_get_new_state = std::chrono::steady_clock::now();

    auto split_states = state_manager_->splitStates(state_variables_, suffix_lengths, state_vars, *state_comp_vec_factory_);

    size_t output_offset = 0ul;
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* current_cache = requests[r].final_cache;
        size_t             suffix_length = suffix_lengths[r];
        while (suffix_length > 0ul) {
            current_cache->state = std::move(split_states[output_offset + suffix_length - 1]);
            current_cache        = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(current_cache->parent.handle()));
            suffix_length -= 1ul;
        }
        output_offset += suffix_lengths[r];
    }

    auto end_split_state = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration = end_split_state - end_prepare;
    size_t                                    bucket   = requests.size() - 1;
    run_time_.at(bucket) += duration.count();
    run_count_.at(bucket) += 1ul;

    if (dump_scores_) {
        for (auto const& r : requests) {
            detail::dump_scores(*r.final_cache, dump_scores_prefix_);
        }
    }

    if (async) {
        for (auto hist : early_request_histories) {
            ScoresWithContext* c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist->handle()));
            if (c->computed.load() or c->ref_count == 1ul or c->info.numStates == 0) {
                finished_queue_.enqueue(hist);
            }
            else {
                pending_.push_back(hist);
            }
        }
        if (sc != nullptr) {
            to_fwd_finished_.set_value(hist);
        }
    }

    auto end = std::chrono::steady_clock::now();

    detail::TimeStatistics stats;
    stats.total_duration          = std::chrono::duration<double, std::milli>(end - start);
    stats.early_request_duration  = std::chrono::duration<double, std::milli>(end_early_requests - start);
    stats.request_duration        = std::chrono::duration<double, std::milli>(end_requests - end_early_requests);
    stats.prepare_duration        = std::chrono::duration<double, std::milli>(end_prepare - end_requests);
    stats.merge_state_duration    = std::chrono::duration<double, std::milli>(end_merge_state - end_prepare);
    stats.set_state_duration      = std::chrono::duration<double, std::milli>(end_set_state - end_merge_state);
    stats.run_nn_output_duration  = std::chrono::duration<double, std::milli>(end_nn_output - end_set_state);
    stats.set_nn_output_duration  = std::chrono::duration<double, std::milli>(end_set_nn_output - end_nn_output);
    stats.get_new_state_duration  = std::chrono::duration<double, std::milli>(end_get_new_state - end_set_nn_output);
    stats.split_state_duration    = std::chrono::duration<double, std::milli>(end_split_state - end_get_new_state);
    stats.softmax_output_duration = std::chrono::duration<double, std::milli>();
    if (verbose_) {
        stats.write(std::cerr);
        std::cerr << " #pr:" << num_pending_requests
                  << " #er:" << num_early_requests
                  << " #r:" << requests.size() << std::endl;
    }
    fwd_statistics_ += stats;
}

}  // namespace Lm

#endif  // _LM_RECURRENT_LANGUAGE_MODEL_HH
