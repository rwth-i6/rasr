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
#include "TFRecurrentLanguageModel.hh"

#include <functional>

#include "BlasNceSoftmaxAdapter.hh"
#include "LstmStateManager.hh"
#include "Module.hh"
#include "NceSoftmaxAdapter.hh"
#include "PassthroughSoftmaxAdapter.hh"
#include "QuantizedBlasNceSoftmaxAdapter.hh"
#include "TransformerStateManager.hh"

namespace {
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

    void add_cache(ScoresWithContext* cache) {
        std::vector<ScoresWithContext*> request_chain;
        request_chain.push_back(cache);
        ScoresWithContext* parent = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(cache->parent.handle()));
        request_chain.push_back(parent);
        while (parent->state.empty()) {
            parent = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(parent->parent.handle()));
            request_chain.push_back(parent);
        }

        std::vector<size_t>* child_idxs = &roots;
        while (not request_chain.empty()) {
            // find root node
            size_t child_idx = child_idxs->size();
            for (size_t c = 0ul; c < child_idxs->size(); c++) {
                if (entries[child_idxs->at(c)] == request_chain.back()) {
                    child_idx = c;
                    break;
                }
            }
            size_t next_child_idx = 0ul;
            if (child_idx == child_idxs->size()) {
                child_idxs->push_back(entries.size());
                entries.push_back(request_chain.back());
                next_child_idx = child_idxs->at(child_idx);
                children.emplace_back();  // can invalidate child_idxs
            }
            else {
                next_child_idx = child_idxs->at(child_idx);
            }
            child_idxs = &children[next_child_idx];
            request_chain.pop_back();
        }
    }

    void get_requests_dfs(std::vector<FwdRequest>& requests, ScoresWithContext* initial, size_t entry, size_t length) {
        if (children[entry].empty()) {
            requests.emplace_back(FwdRequest{initial, entries[entry], length});
        }
        else {
            for (size_t e : children[entry]) {
                get_requests_dfs(requests, initial, e, length + 1ul);
            }
        }
    }

    std::vector<FwdRequest> get_requests() {
        std::vector<FwdRequest> result;
        for (size_t r : roots) {
            for (size_t c : children[r]) {
                get_requests_dfs(result, entries[r], c, 1ul);
            }
        }
        return result;
    }
};

void dump_scores(ScoresWithContext const& cache, std::string const& prefix) {
    std::stringstream path;
    path << prefix;
    for (auto token : *cache.history) {
        path << "_" << token;
    }
    std::ofstream out(path.str(), std::ios::out | std::ios::trunc);
    out << "nn_output:\n";
    std::vector<float> nn_output(cache.nn_output->size());
    cache.nn_output->uncompress(nn_output.data(), nn_output.size());
    for (auto nn_out : nn_output) {
        out << nn_out << '\n';
    }
    for (size_t s = 0ul; s < cache.state.size(); s++) {
        out << "state " << s << ":\n";
        std::vector<float> state_data(cache.state[s]->size());
        cache.state[s]->uncompress(state_data.data(), state_data.size());
        for (auto v : state_data) {
            out << v << '\n';
        }
    }
}

void clear_queue(Lm::TFRecurrentLanguageModel::HistoryQueue& queue) {
    Lm::History const* hist = nullptr;
    while (queue.try_dequeue(hist)) {
        delete hist;
    }
}
}  // namespace

namespace Lm {

enum StateManagerType {
    LstmStateManagerType,
    TransformerStateManagerType,
    TransformerStateManager16BitType,
    TransformerStateManager8BitType,
    TransformerStateManagerWithCommonPrefixType,
    TransformerStateManagerWithCommonPrefix16BitType,
    TransformerStateManagerWithCommonPrefix8BitType,
};

const Core::Choice stateManagerTypeChoice(
        "lstm", LstmStateManagerType,
        "transformer", TransformerStateManagerType,
        "transformer-16bit", TransformerStateManager16BitType,
        "transformer-8bit", TransformerStateManager8BitType,
        "transformer-with-common-prefix", TransformerStateManagerWithCommonPrefixType,
        "transformer-with-common-prefix-16bit", TransformerStateManagerWithCommonPrefix16BitType,
        "transformer-with-common-prefix-8bit", TransformerStateManagerWithCommonPrefix8BitType,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice,
        "type of the state manager",
        LstmStateManagerType);

std::unique_ptr<StateManager> createStateManager(Core::Configuration const& config) {
    StateManager* res = nullptr;
    switch (stateManagerTypeParam(config)) {
        case LstmStateManagerType: res = new Lm::LstmStateManager(config); break;
        case TransformerStateManagerType: res = new Lm::TransformerStateManager<float>(config); break;
        case TransformerStateManager16BitType: res = new Lm::TransformerStateManager<int16_t>(config); break;
        case TransformerStateManager8BitType: res = new Lm::TransformerStateManager<int8_t>(config); break;
        case TransformerStateManagerWithCommonPrefixType: res = new Lm::TransformerStateManagerWithCommonPrefix<float>(config); break;
        case TransformerStateManagerWithCommonPrefix16BitType: res = new Lm::TransformerStateManagerWithCommonPrefix<int16_t>(config); break;
        case TransformerStateManagerWithCommonPrefix8BitType: res = new Lm::TransformerStateManagerWithCommonPrefix<int8_t>(config); break;
        default: defect();
    }
    return std::unique_ptr<StateManager>(res);
}

enum SoftmaxAdapterType {
    BlasNceSoftmaxAdapterType,
    NceSoftmaxAdapterType,
    PassthroughSoftmaxAdapterType,
    QuantizedBlasNceSoftmaxAdapter16BitType
};

const Core::Choice softmaxAdapterTypeChoice(
        "blas_nce", BlasNceSoftmaxAdapterType,  // included for backward compatibility
        "blas-nce", BlasNceSoftmaxAdapterType,  // more consistent with RASR conventions
        "nce", NceSoftmaxAdapterType,
        "passthrough", PassthroughSoftmaxAdapterType,
        "quantized-blas-nce-16bit", QuantizedBlasNceSoftmaxAdapter16BitType,
        Core::Choice::endMark());

const Core::ParameterChoice softmaxAdapterTypeParam(
        "type", &softmaxAdapterTypeChoice,
        "type of the softmax adapter",
        PassthroughSoftmaxAdapterType);

std::unique_ptr<SoftmaxAdapter> createSoftmaxAdapter(Core::Configuration const& config) {
    switch (softmaxAdapterTypeParam(config)) {
        case BlasNceSoftmaxAdapterType: return std::unique_ptr<SoftmaxAdapter>(new Lm::BlasNceSoftmaxAdapter(config));
        case NceSoftmaxAdapterType: return std::unique_ptr<SoftmaxAdapter>(new Lm::NceSoftmaxAdapter(config));
        case PassthroughSoftmaxAdapterType: return std::unique_ptr<SoftmaxAdapter>(new Lm::PassthroughSoftmaxAdapter(config));
        case QuantizedBlasNceSoftmaxAdapter16BitType: return std::unique_ptr<SoftmaxAdapter>(new Lm::QuantizedBlasNceSoftmaxAdapter16Bit(config));
        default: defect();
    }
}

TFRecurrentLanguageModel::TimeStatistics TFRecurrentLanguageModel::TimeStatistics::operator+(TimeStatistics const& other) const {
    TimeStatistics res;

    res.total_duration          = total_duration + other.total_duration;
    res.early_request_duration  = early_request_duration + other.early_request_duration;
    res.request_duration        = request_duration + other.request_duration;
    res.prepare_duration        = prepare_duration + other.prepare_duration;
    res.merge_state_duration    = merge_state_duration + other.merge_state_duration;
    res.set_state_duration      = set_state_duration + other.set_state_duration;
    res.run_nn_output_duration  = run_nn_output_duration + other.run_nn_output_duration;
    res.set_nn_output_duration  = set_nn_output_duration + other.set_nn_output_duration;
    res.get_new_state_duration  = get_new_state_duration + other.get_new_state_duration;
    res.split_state_duration    = split_state_duration + other.split_state_duration;
    res.softmax_output_duration = softmax_output_duration + other.softmax_output_duration;

    return res;
}

TFRecurrentLanguageModel::TimeStatistics& TFRecurrentLanguageModel::TimeStatistics::operator+=(TimeStatistics const& other) {
    total_duration += other.total_duration;
    early_request_duration += other.early_request_duration;
    request_duration += other.request_duration;
    prepare_duration += other.prepare_duration;
    merge_state_duration += other.merge_state_duration;
    set_state_duration += other.set_state_duration;
    run_nn_output_duration += other.run_nn_output_duration;
    set_nn_output_duration += other.set_nn_output_duration;
    get_new_state_duration += other.get_new_state_duration;
    split_state_duration += other.split_state_duration;
    softmax_output_duration += other.softmax_output_duration;

    return *this;
}

void TFRecurrentLanguageModel::TimeStatistics::write(Core::XmlChannel& channel) const {
    channel << Core::XmlOpen("total-duration") + Core::XmlAttribute("unit", "milliseconds") << total_duration.count() << Core::XmlClose("total-duration");
    channel << Core::XmlOpen("early-request-duration") + Core::XmlAttribute("unit", "milliseconds") << early_request_duration.count() << Core::XmlClose("early-request-duration");
    channel << Core::XmlOpen("request-duration") + Core::XmlAttribute("unit", "milliseconds") << request_duration.count() << Core::XmlClose("request-duration");
    channel << Core::XmlOpen("prepare-duration") + Core::XmlAttribute("unit", "milliseconds") << prepare_duration.count() << Core::XmlClose("prepare-duration");
    channel << Core::XmlOpen("merge-state-duration") + Core::XmlAttribute("unit", "milliseconds") << merge_state_duration.count() << Core::XmlClose("merge-state-duration");
    channel << Core::XmlOpen("set-state-duration") + Core::XmlAttribute("unit", "milliseconds") << set_state_duration.count() << Core::XmlClose("set-state-duration");
    channel << Core::XmlOpen("run-nn-output-duration") + Core::XmlAttribute("unit", "milliseconds") << run_nn_output_duration.count() << Core::XmlClose("run-nn-output-duration");
    channel << Core::XmlOpen("set-nn-output-duration") + Core::XmlAttribute("unit", "milliseconds") << set_nn_output_duration.count() << Core::XmlClose("set-nn-output-duration");
    channel << Core::XmlOpen("get-new-state-duration") + Core::XmlAttribute("unit", "milliseconds") << get_new_state_duration.count() << Core::XmlClose("get-new-state-duration");
    channel << Core::XmlOpen("split-state-duration") + Core::XmlAttribute("unit", "milliseconds") << split_state_duration.count() << Core::XmlClose("split-state-duration");
    channel << Core::XmlOpen("softmax-output-duration") + Core::XmlAttribute("unit", "milliseconds") << softmax_output_duration.count() << Core::XmlClose("softmax-output-duration");
}

void TFRecurrentLanguageModel::TimeStatistics::write(std::ostream& out) const {
    out << "fwd: " << total_duration.count()
        << " er:" << early_request_duration.count()
        << " r:" << request_duration.count()
        << " p:" << prepare_duration.count()
        << " ms: " << merge_state_duration.count()
        << " sst:" << set_state_duration.count()
        << " rs:" << run_nn_output_duration.count()
        << " sno:" << set_nn_output_duration.count()
        << " gns:" << get_new_state_duration.count()
        << " ss: " << split_state_duration.count()
        << " smo:" << softmax_output_duration.count();
}

const Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputLog("transform-output-log", "apply log to tensorflow output", false);
const Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputNegate("transform-output-negate", "negate tensorflow output (after log)", false);
const Core::ParameterInt    TFRecurrentLanguageModel::paramMinBatchSize("min-batch-size", "minimum number of histories forwarded in one go", 32);
const Core::ParameterInt    TFRecurrentLanguageModel::paramOptBatchSize("opt-batch-size", "optimum number of histories forwarded in one go", 128);
const Core::ParameterInt    TFRecurrentLanguageModel::paramMaxBatchSize("max-batch-size", "maximum number of histories forwarded in one go", 2048);
const Core::ParameterInt    TFRecurrentLanguageModel::paramHistoryPruningThreshold("history-pruning-threshold", "if the history is longer than this parameter it will be pruned", std::numeric_limits<int>::max(), 0);
const Core::ParameterInt    TFRecurrentLanguageModel::paramPrunedHistoryLength("pruned-history-length", "length of the pruned history (should be smaller than history-pruning-threshold)", std::numeric_limits<int>::max(), 0);
const Core::ParameterFloat  TFRecurrentLanguageModel::paramBatchPruningThreshold("batch-pruning-threshold", "pruning threshold for all hypothesis beyond min-batch-size during eager forwarding", 10.0);
const Core::ParameterBool   TFRecurrentLanguageModel::paramAllowReducedHistory("allow-reduced-history", "wether this LM will actually reduce the history length", false);
const Core::ParameterBool   TFRecurrentLanguageModel::paramDumpInputs("dump-inputs", "write all inputs from this LM to disk", false);
const Core::ParameterString TFRecurrentLanguageModel::paramDumpInputsPrefix("dump-inputs-prefix", "prefix for the input dumps", "inputs");
const Core::ParameterBool   TFRecurrentLanguageModel::paramDumpScores("dump-scores", "write all scores from this LM to disk", false);
const Core::ParameterString TFRecurrentLanguageModel::paramDumpScoresPrefix("dump-scores-prefix", "prefix for the score dumps", "scores");
const Core::ParameterBool   TFRecurrentLanguageModel::paramLogMemory("log-memory", "wether memory usage from nn-outputs / states should be logged", false);
const Core::ParameterBool   TFRecurrentLanguageModel::paramFreeMemory("free-memory", "wether nn-outputs should be deleted after some delay", false);
const Core::ParameterInt    TFRecurrentLanguageModel::paramFreeMemoryDelay("free-memory-delay", "how many time frames without usage before nn-outputs are deleted", 40);
const Core::ParameterBool   TFRecurrentLanguageModel::paramAsync("async", "wether to forward histories in a separate thread", false);
const Core::ParameterBool   TFRecurrentLanguageModel::paramSingleStepOnly("single-step-only", "workaround for some bug that results in wrong scores when recombination is done in combination with async evaluation", false);
const Core::ParameterBool   TFRecurrentLanguageModel::paramVerbose("verbose", "wether to print detailed statistics to stderr", false);

TFRecurrentLanguageModel::TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
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
          async_(paramAsync(config)),
          single_step_only_(paramSingleStepOnly(config)),
          verbose_(paramVerbose(config)),
          session_(select("session")),
          loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
          graph_(loader_->load_graph()),
          tensor_input_map_(select("input-map")),
          tensor_output_map_(select("output-map")),
          state_comp_vec_factory_(Lm::Module::instance().createCompressedVectorFactory(select("state-compression"))),
          nn_output_comp_vec_factory_(Lm::Module::instance().createCompressedVectorFactory(select("nn-output-compression"))),
          state_manager_(createStateManager(select("state-manager"))),
          softmax_adapter_(createSoftmaxAdapter(select("softmax-adapter"))),
          statistics_(config, "statistics"),
          current_time_(0u),
          run_time_(max_batch_size_, 0.0),
          run_count_(max_batch_size_, 0ul),
          total_wait_time_(0.0),
          total_start_frame_time_(0.0),
          total_expand_hist_time_(0.0),
          fwd_statistics_(),
          dump_inputs_counter_(0ul),
          background_forwarder_thread_(),
          should_stop_(false),
          to_fwd_(nullptr),
          to_fwd_finished_(),
          pending_(),
          fwd_queue_(32768),
          finished_queue_(32768) {
    require_le(pruned_history_length_, history_pruning_threshold_);
    session_.addGraph(*graph_);
    loader_->initialize(session_);

    auto const& softmax_info = tensor_output_map_.get_info("softmax");
    output_tensor_names_.push_back(softmax_info.tensor_name());
    state_variables_.reserve(state_variables_.size());
    for (std::string const& s : graph_->state_vars()) {
        auto const& var = graph_->variables().find(s)->second;
        state_variables_.emplace_back(var);
        initializer_tensor_names_.push_back(var.initializer_name);
        read_vars_tensor_names_.push_back(var.snapshot_name);
    }

    if (transform_output_log_ and transform_output_negate_) {
        output_transform_function_ = [](Score v) { return -std::log(v); };
    }
    else if (transform_output_log_) {
        output_transform_function_ = [](Score v) { return std::log(v); };
    }
    else if (transform_output_negate_) {
        output_transform_function_ = [](Score v) { return -v; };
    }

    NNHistoryManager*  hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence    ts;
    HistoryHandle      h     = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->state             = state_manager_->initialState(state_variables_, *state_comp_vec_factory_);
    std::vector<f32> temp(1);
    auto             compression_param_estimator = nn_output_comp_vec_factory_->getEstimator();
    compression_param_estimator->accumulate(temp.data(), temp.size());
    auto compression_params = compression_param_estimator->estimate();
    // pretend this history has already been evaluated
    cache->nn_output = nn_output_comp_vec_factory_->compress(temp.data(), temp.size(), compression_params.get());
    cache->computed.store(true);
    cache->last_used = std::numeric_limits<Search::TimeframeIndex>::max();
    empty_history_   = history(h);

    softmax_adapter_->init(session_, tensor_input_map_, tensor_output_map_);

    if (async_) {
        background_forwarder_thread_ = std::thread(std::bind(&TFRecurrentLanguageModel::background_forward, this));
    }
}

TFRecurrentLanguageModel::~TFRecurrentLanguageModel() {
    clear_queue(finished_queue_);

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

History TFRecurrentLanguageModel::startHistory() const {
    NNHistoryManager*  hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence    ts(1ul, lexicon_mapping_[sentenceBeginToken()->id()]);
    HistoryHandle      h     = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->parent            = empty_history_;
    History hist(history(h));
    return hist;
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Token w) const {
    return extendedHistory(hist, w->id());
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Bliss::Token::Id w) const {
    return extendHistoryWithOutputIdx(hist, lexicon_mapping_[w]);
}

History TFRecurrentLanguageModel::reducedHistory(History const& hist, u32 limit) const {
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

Score TFRecurrentLanguageModel::score(History const& hist, Token w) const {
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
    Score score    = output_transform_function_(softmax_adapter_->get_score(sc->nn_output, output_idx));
    auto  end      = std::chrono::steady_clock::now();
    auto  duration = std::chrono::duration<double, std::milli>(end - start);
    fwd_statistics_.softmax_output_duration += duration;
    fwd_statistics_.total_duration += duration;
    return score;
}

bool TFRecurrentLanguageModel::scoreCached(History const& hist, Token w) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    return sc->computed.load();
}

void TFRecurrentLanguageModel::load() {
    loadVocabulary();
}

void TFRecurrentLanguageModel::startFrame(Search::TimeframeIndex time) const {
    auto timer_start = std::chrono::steady_clock::now();

    current_time_ = time;

    size_t nn_output_cache_size = 0ul;
    size_t state_cache_size     = 0ul;
    size_t num_histories        = 0ul;

    clear_queue(finished_queue_);

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

void TFRecurrentLanguageModel::setInfo(History const& hist, SearchSpaceInformation const& info) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
    sc->info              = info;
    sc->last_info         = current_time_;
}

History TFRecurrentLanguageModel::extendHistoryWithOutputIdx(History const& hist, size_t w) const {
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

void TFRecurrentLanguageModel::background_forward() const {
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

template<bool async>
void TFRecurrentLanguageModel::forward(Lm::History const* hist) const {
    ScoresWithContext* sc = nullptr;
    if (hist != nullptr) {
        sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist->handle()));
    }
    if (async and sc != nullptr and sc->computed.load()) {  // nothing to do (only happens in async case)
        to_fwd_finished_.set_value(hist);
        return;
    }
    auto start = std::chrono::steady_clock::now();

    RequestGraph request_graph;
    if (sc != nullptr) {
        request_graph.add_cache(const_cast<ScoresWithContext*>(sc));
    }

    std::vector<FwdRequest>  requests;
    std::vector<Lm::History> request_histories;  // make sure none of the request caches go away while we compute the scores
    size_t                   max_length = 0ul;

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

    std::vector<StateManager::HistoryState const*> prefix_states(full_prefix_required ? total_prefix_length : requests.size());
    size_t                                         current_offset = 0ul;
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
    std::vector<std::pair<std::string, Tensorflow::Tensor>> inputs;
    std::vector<std::string>                                targets;
    state_manager_->mergeStates(state_variables_, prefix_lengths, prefix_states, inputs, targets);
    std::vector<s32> state_lengths(prefix_lengths.begin(), prefix_lengths.end());

    if (dump_inputs_) {
        std::string out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_state_";
        for (size_t i = 0ul; i < inputs.size(); i++) {
            inputs[i].second.save<s16>(out + std::to_string(i));
        }
    }

    auto end_merge_state = std::chrono::steady_clock::now();

    session_.run(inputs, targets);

    auto end_set_state = std::chrono::steady_clock::now();

    // run nn-output calculation
    inputs.clear();
    auto const& word_info = tensor_input_map_.get_info("word");
    inputs.emplace_back(std::make_pair(word_info.tensor_name(), Tensorflow::Tensor::create(words)));
    if (not word_info.seq_length_tensor_name().empty()) {
        inputs.emplace_back(std::make_pair(word_info.seq_length_tensor_name(), Tensorflow::Tensor::create(word_lengths)));
    }
    if (tensor_input_map_.has_info("state-lengths")) {
        auto const& state_lengths_info = tensor_input_map_.get_info("state-lengths");
        inputs.emplace_back(std::make_pair(state_lengths_info.tensor_name(), Tensorflow::Tensor::create(state_lengths)));
    }
    std::vector<Tensorflow::Tensor> outputs;
    session_.run(inputs, output_tensor_names_, graph_->update_ops(), outputs);

    if (dump_inputs_) {
        std::string out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_nn_in_";
        for (size_t i = 0ul; i < inputs.size(); i++) {
            inputs[i].second.save<s32>(out + std::to_string(i));
        }
        out = dump_inputs_prefix_ + "_" + std::to_string(dump_inputs_counter_) + "_nn_out_";
        for (size_t i = 0ul; i < outputs.size(); i++) {
            outputs[i].save<f32>(out + std::to_string(i));
        }
        dump_inputs_counter_ += 1ul;
    }

    auto end_nn_output = std::chrono::steady_clock::now();

    // store outputs in caches
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* cache = requests[r].final_cache;
        for (size_t w = requests[r].length; w > 0;) {
            --w;
            cache->last_used                         = current_time_;
            int          num_outputs                 = outputs[0ul].dimSize(2);
            auto         compression_param_estimator = nn_output_comp_vec_factory_->getEstimator();
            float const* data                        = outputs[0ul].data<f32>(r, w, 0);
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
    session_.run({}, read_vars_tensor_names_, {}, outputs);

    auto end_get_new_state = std::chrono::steady_clock::now();

    auto split_states = state_manager_->splitStates(state_variables_, suffix_lengths, outputs, *state_comp_vec_factory_);

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
            dump_scores(*r.final_cache, dump_scores_prefix_);
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

    TimeStatistics stats;
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
