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
#include "TFRecurrentLanguageModel.hh"

#include <functional>

namespace {
    struct ScoresWithContext : public Lm::NNCacheWithStats {
        std::atomic<bool>                  computed;
        Lm::History                        parent;
        Math::FastVector<Lm::Score>        scores;
        std::vector<Math::FastVector<f32>> state; // for now float is hardcoded, TODO: replace this with Tensor and add merge utility for tensors
        Lm::SearchSpaceInformation         info;
        Search::TimeframeIndex             last_used;
        Search::TimeframeIndex             last_info;
        bool                               was_expanded;
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
        out << "scores:\n";
        for (auto score : cache.scores) {
            out << score << '\n';
        }
        for (size_t s = 0ul; s < cache.state.size(); s++) {
            out << "state " << s << ":\n";
            for (auto v : cache.state[s]) {
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
} // namespace

namespace Lm {

TFRecurrentLanguageModel::TimeStatistics TFRecurrentLanguageModel::TimeStatistics::operator+(TimeStatistics const& other) const {
    TimeStatistics res;

    res.total_duration         = total_duration         + other.total_duration;
    res.early_request_duration = early_request_duration + other.early_request_duration;
    res.request_duration       = request_duration       + other.request_duration;
    res.prepare_duration       = prepare_duration       + other.prepare_duration;
    res.set_state_duration     = set_state_duration     + other.set_state_duration;
    res.run_score_duration     = run_score_duration     + other.run_score_duration;
    res.set_score_duration     = set_score_duration     + other.set_score_duration;
    res.set_new_state_duration = set_new_state_duration + other.set_new_state_duration;

    return res;
}

TFRecurrentLanguageModel::TimeStatistics& TFRecurrentLanguageModel::TimeStatistics::operator+=(TimeStatistics const& other) {
    total_duration         += other.total_duration;
    early_request_duration += other.early_request_duration;
    request_duration       += other.request_duration;
    prepare_duration       += other.prepare_duration;
    set_state_duration     += other.set_state_duration;
    run_score_duration     += other.run_score_duration;
    set_score_duration     += other.set_score_duration;
    set_new_state_duration += other.set_new_state_duration;

    return *this;
}

void TFRecurrentLanguageModel::TimeStatistics::write(Core::XmlChannel& channel) const {
    channel << Core::XmlOpen("total-duration")         + Core::XmlAttribute("unit", "milliseconds") << total_duration.count()         << Core::XmlClose("total-duration");
    channel << Core::XmlOpen("early-request-duration") + Core::XmlAttribute("unit", "milliseconds") << early_request_duration.count() << Core::XmlClose("early-request-duration");
    channel << Core::XmlOpen("request-duration")       + Core::XmlAttribute("unit", "milliseconds") << request_duration.count()       << Core::XmlClose("request-duration");
    channel << Core::XmlOpen("prepare-duration")       + Core::XmlAttribute("unit", "milliseconds") << prepare_duration.count()       << Core::XmlClose("prepare-duration");
    channel << Core::XmlOpen("set-state-duration")     + Core::XmlAttribute("unit", "milliseconds") << set_state_duration.count()     << Core::XmlClose("set-state-duration");
    channel << Core::XmlOpen("run-score-duration")     + Core::XmlAttribute("unit", "milliseconds") << run_score_duration.count()     << Core::XmlClose("run-score-duration");
    channel << Core::XmlOpen("set-score-duration")     + Core::XmlAttribute("unit", "milliseconds") << set_score_duration.count()     << Core::XmlClose("set-score-duration");
    channel << Core::XmlOpen("set-new-state-duration") + Core::XmlAttribute("unit", "milliseconds") << set_new_state_duration.count() << Core::XmlClose("set-new-state-duration");
}

void TFRecurrentLanguageModel::TimeStatistics::write(std::ostream& out) const {
    out << "fwd: " << total_duration.count()
        << " er:"  << early_request_duration.count()
        << " r:"   << request_duration.count()
        << " p:"   << prepare_duration.count()
        << " sst:" << set_state_duration.count()
        << " rs:"  << run_score_duration.count()
        << " ssc:" << set_score_duration.count()
        << " sns:" << set_new_state_duration.count();
}

Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputLog    ("transform-output-log",    "apply log to tensorflow output",       false);
Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputNegate ("transform-output-negate", "negate tensorflow output (after log)", false);
Core::ParameterInt    TFRecurrentLanguageModel::paramMinBatchSize         ("min-batch-size",          "minimum number of histories forwarded in one go",   32);
Core::ParameterInt    TFRecurrentLanguageModel::paramOptBatchSize         ("opt-batch-size",          "optimum number of histories forwarded in one go",  128);
Core::ParameterInt    TFRecurrentLanguageModel::paramMaxBatchSize         ("max-batch-size",          "maximum number of histories forwarded in one go", 2048);
Core::ParameterFloat  TFRecurrentLanguageModel::paramBatchPruningThreshold("batch-pruning-threshold", "pruning threshold for all hypothesis beyond min-batch-size during eager forwarding", 10.0);
Core::ParameterBool   TFRecurrentLanguageModel::paramAllowReducedHistory  ("allow-reduced-history",   "wether this LM will actually reduce the history length", false);
Core::ParameterBool   TFRecurrentLanguageModel::paramDumpScores           ("dump-scores",             "write all scores from this LM to disk", false);
Core::ParameterString TFRecurrentLanguageModel::paramDumpScoresPrefix     ("dump-scores-prefix",      "prefix for the score dumps",            "scores");
Core::ParameterBool   TFRecurrentLanguageModel::paramLogMemory            ("log-memory",              "wether memory usage from scores / states should be logged", false);
Core::ParameterBool   TFRecurrentLanguageModel::paramFreeMemory           ("free-memory",             "wether scores should be deleted after some delay", false);
Core::ParameterInt    TFRecurrentLanguageModel::paramFreeMemoryDelay      ("free-memory-delay",       "how many time frames without usage before scores are deleted", 40);
Core::ParameterBool   TFRecurrentLanguageModel::paramAsync                ("async",                   "wether to forward histories in a separate thread", false);
Core::ParameterBool   TFRecurrentLanguageModel::paramSingleStepOnly       ("single-step-only",        "workaround for some bug that results in wrong scores when recombination is done in combination with async evaluation", false);
Core::ParameterBool   TFRecurrentLanguageModel::paramVerbose              ("verbose",                 "wether to print detailed statistics to stderr", false);

TFRecurrentLanguageModel::TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
                         : Core::Component(c), Precursor(c, l),
                           transform_output_log_(paramTransformOuputLog(config)), transform_output_negate_(paramTransformOuputNegate(config)),
                           min_batch_size_(paramMinBatchSize(config)), opt_batch_size_(paramOptBatchSize(config)), max_batch_size_(paramMaxBatchSize(config)),
                           batch_pruning_threshold_(paramBatchPruningThreshold(config)), allow_reduced_history_(paramAllowReducedHistory(config)),
                           dump_scores_(paramDumpScores(config)), dump_scores_prefix_(paramDumpScoresPrefix(config)),
                           log_memory_(paramLogMemory(config)), free_memory_(paramFreeMemory(config)), free_memory_delay_(paramFreeMemoryDelay(config)),
                           async_(paramAsync(config)), single_step_only_(paramSingleStepOnly(config)), verbose_(paramVerbose(config)),
                           session_(select("session")), loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
                           graph_(loader_->load_graph()), tensor_input_map_(select("input-map")), tensor_output_map_(select("output-map")),
                           statistics_(config, "statistics"),
                           current_time_(0u), run_time_(max_batch_size_, 0.0), run_count_(max_batch_size_, 0ul),
                           total_wait_time_(0.0), total_start_frame_time_(0.0), total_expand_hist_time_(0.0), fwd_statistics_(),
                           background_forwarder_thread_(), should_stop_(false), to_fwd_(nullptr), to_fwd_finished_(),
                           pending_(), fwd_queue_(32768), finished_queue_(32768) {
    session_.addGraph(*graph_);
    loader_->initialize(session_);

    auto const& softmax_info = tensor_output_map_.get_info("softmax");
    output_tensor_names_.push_back(softmax_info.tensor_name());
    for (std::string const& s : graph_->state_vars()) {
        auto const& var = graph_->variables().find(s)->second;
        initializer_tensor_names_.push_back(var.initializer_name);
        read_vars_tensor_names_.push_back(var.snapshot_name);
    }

    if (transform_output_log_ and transform_output_negate_) {
        output_transform_function_ = [](Score v){ return -std::log(v); };
    }
    else if (transform_output_log_) {
        output_transform_function_ = [](Score v){ return std::log(v); };
    }
    else if (transform_output_negate_) {
        output_transform_function_ = [](Score v){ return -v; };
    }

    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence ts;
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    for (std::string const& state : graph_->state_vars()) {
        auto const& var = graph_->variables().find(state)->second;
        require_gt(var.shape.size(), 0ul);
        s64 state_size = var.shape.back();
        require_ge(state_size, 0); // variable must not be of unknown size
        cache->state.emplace_back(static_cast<size_t>(state_size));
        cache->state.back().fill(0.0f);
    }
    cache->computed.store(true);
    cache->scores.resize(1u); // pretend this history has already been evaluated
    cache->last_used = std::numeric_limits<Search::TimeframeIndex>::max();
    empty_history_ = history(h);

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
            total_fwd_hist  += (i+1) * run_count_[i];
            total_run_time  += run_time_[i];
        }
    }
    statistics_ << Core::XmlClose("fwd-time");

    statistics_ << Core::XmlOpen("fwd-summary");
    statistics_ << Core::XmlOpen("total-run-count") << total_run_count  << Core::XmlClose("total-run-count");
    statistics_ << Core::XmlOpen("total-fwd-hist")  << total_fwd_hist   << Core::XmlClose("total-fwd-hist");
    statistics_ << Core::XmlOpen("total-run-time")             + Core::XmlAttribute("unit", "milliseconds") << total_run_time              << Core::XmlClose("total-run-time");
    statistics_ << Core::XmlOpen("total-wait-time")            + Core::XmlAttribute("unit", "milliseconds") << total_wait_time_            << Core::XmlClose("total-wait-time");
    statistics_ << Core::XmlOpen("total-start-frame-time")     + Core::XmlAttribute("unit", "milliseconds") << total_start_frame_time_     << Core::XmlClose("total-start-frame-time");
    statistics_ << Core::XmlOpen("total-expand-hist-time")     + Core::XmlAttribute("unit", "milliseconds") << total_expand_hist_time_     << Core::XmlClose("total-expand-hist-time");
    statistics_ << Core::XmlOpen("fwd-times");
    fwd_statistics_.write(statistics_);
    statistics_ << Core::XmlClose("fwd-times");
    statistics_ << Core::XmlClose("fwd-summary");
}

History TFRecurrentLanguageModel::startHistory() const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence ts(1ul, lexicon_mapping_[sentenceBeginToken()->id()]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->parent = empty_history_;
    History hist(history(h));
    return hist;
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Token w) const {
    return extendedHistory(hist, w->id());
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Bliss::Token::Id w) const {
    auto timer_start = std::chrono::steady_clock::now();
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    TokenIdSequence ts(*sc->history);
    ts.push_back(lexicon_mapping_[w]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    if (cache->parent.handle() == nullptr) {
        cache->parent = hist;
        ScoresWithContext* parent_cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
        parent_cache->was_expanded = true;
        if (async_) {
            fwd_queue_.enqueue(new History(history(h)));
        }
    }
    History ext_hist(history(h));
    auto timer_end = std::chrono::steady_clock::now();
    double expand_hist_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    total_expand_hist_time_ += expand_hist_time;
    return ext_hist;
}


History TFRecurrentLanguageModel::reducedHistory(History const& hist, u32 limit) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    if (not allow_reduced_history_ or sc->history->size() <= limit) {
        return hist;
    }
    History h = startHistory();
    for (u32 w = limit; w > 0; w--) {
        h = extendedHistory(h, sc->history->at(sc->history->size() - w));
    }
    return h;
}

Score TFRecurrentLanguageModel::score(History const& hist, Token w) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));

    if (not sc->computed.load()) {
        auto start = std::chrono::steady_clock::now();
        if (async_) {
            // promise should only be used once
            to_fwd_finished_ = std::promise<History const*>();
            std::future<History const*> future = to_fwd_finished_.get_future();
            to_fwd_.store(&hist);
            future.wait();
        }
        else {
            forward<false>(&hist);
        }
        auto end = std::chrono::steady_clock::now();
        double wait_time = std::chrono::duration<double, std::milli>(end - start).count();
        total_wait_time_ += wait_time;
        if (verbose_) {
            std::cerr << "wait: " << wait_time << " " << sc->info.numStates << " " << sc->info.bestScoreOffset << std::endl;
        }
    }

    require(sc->computed.load());

    size_t output_idx = lexicon_mapping_[w->id()];
    useOutput(*sc, output_idx);
    sc->last_used = current_time_;
    return sc->scores.at(output_idx);
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

    size_t score_cache_size = 0ul;
    size_t state_cache_size = 0ul;
    size_t num_histories    = 0ul;

    clear_queue(finished_queue_);

    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    hm->visit([&](HistoryHandle h) {
        num_histories += 1ul;
        ScoresWithContext* c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
        bool computed = c->computed.load();
        if (free_memory_ and computed and c->was_expanded and c->info.numStates == 0 and c->last_used < current_time_ - std::min(free_memory_delay_, current_time_)) {
            c->scores.clear();
            c->computed.store(false);
        }
        else if (async_ and not computed and not (c->was_expanded and c->info.numStates == 0)) {
            fwd_queue_.enqueue(new History(history(h)));
        }
        score_cache_size += c->scores.size() * sizeof(decltype(c->scores)::value_type);
        for (auto const& s : c->state) {
            state_cache_size += s.size() * sizeof(float);
        }
    });

    if (log_memory_ and statistics_.isOpen()) {
        statistics_ << Core::XmlOpen("memory-usage")     + Core::XmlAttribute("time-frame", current_time_);
        statistics_ << Core::XmlOpen("score-cache-size") + Core::XmlAttribute("unit", "MB") << (score_cache_size / (1024. * 1024.)) << Core::XmlClose("score-cache-size");
        statistics_ << Core::XmlOpen("state-cache-size") + Core::XmlAttribute("unit", "MB") << (state_cache_size / (1024. * 1024.)) << Core::XmlClose("state-cache-size");
        statistics_ << Core::XmlOpen("num-histories")    + Core::XmlAttribute("unit", "MB") << num_histories                        << Core::XmlClose("num-histories");
        statistics_ << Core::XmlClose("memory-usage");
    }

    auto timer_end = std::chrono::steady_clock::now();
    double start_frame_duration = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();
    total_start_frame_time_ += start_frame_duration;
}

void TFRecurrentLanguageModel::setInfo(History const& hist, SearchSpaceInformation const& info) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
    sc->info = info;
    sc->last_info = current_time_;
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

template <bool async>
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

    size_t num_pending_requests = pending_.size();
    std::unordered_set<void const*> handles; // only relevant in async case
    handles.reserve(pending_.size());
    std::vector<ScoresWithContext*> early_requests;
    std::vector<History const*>     early_request_histories; // make sure none of the request caches go away while we compute the scores (only relevant in async case)

    if (async) {
        auto process_hist = [&](History const* hist) {
            ScoresWithContext* c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist->handle()));
            ScoresWithContext* parent_c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(c->parent.handle()));
            if (handles.find(hist->handle()) == handles.end()
                and not c->computed.load()
                and c != sc
                and c->parent.handle() != nullptr
                and c->ref_count > 1
                and (not single_step_only_ or parent_c->computed.load())) {
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
        bool success = false;
        bool first   = true;
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
        } while(success);
    }
    else {
        NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
        hm->visit([&](HistoryHandle h) {
            ScoresWithContext* c = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
            if (not c->computed.load() and c != sc and not (c->was_expanded and c->info.numStates == 0)) {
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
        size_t i = min_batch_size_;
        Score ref_score = requests.front().final_cache->info.bestScoreOffset + batch_pruning_threshold_;
        if (not Math::isinf(ref_score)) {
            while ((i + 1) < requests.size() and requests[i+1].final_cache->info.bestScoreOffset <= ref_score) {
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
    Math::FastMatrix<s32>              words(requests.size(), max_length);
    Math::FastVector<s32>              word_lengths(requests.size());
    std::vector<Math::FastMatrix<f32>> prev_state;
    prev_state.reserve(graph_->state_vars().size());
    for (size_t r = 0ul; r < requests.size(); r++) {
        auto& history = *(requests[r].final_cache->history);
        size_t offset = history.size() - requests[r].length;
        for (size_t w = 0u; w < requests[r].length; w++) {
            words.at(r, w) = static_cast<s32>(history[offset + w]);
        }
        for (size_t w = requests[r].length; w < max_length; w++) {
            words.at(r, w) = 0;
        }
        word_lengths[r] = requests[r].length;
        ScoresWithContext* initial_cache = requests[r].initial_cache;
        require(initial_cache != nullptr);
        require_eq(graph_->state_vars().size(), initial_cache->state.size());
        for (size_t s = 0ul; s < graph_->state_vars().size(); s++) {
            if (s >= prev_state.size()) {
                prev_state.emplace_back(initial_cache->state[s].size(), requests.size());
            }
            else {
                require_eq(initial_cache->state[s].size(), prev_state[s].nRows());
            }
            // we place the state for each history in columns and transpose later
            std::copy(initial_cache->state[s].begin(), initial_cache->state[s].end(), &prev_state[s].at(0u, r));
        }
    }

    auto end_prepare = std::chrono::steady_clock::now();

    // build tensors + set state variables
    std::vector<std::pair<std::string, Tensorflow::Tensor>> inputs;
    inputs.reserve(prev_state.size());
    for (size_t s = 0ul; s < prev_state.size(); s++) {
        auto const& var = graph_->variables().find(graph_->state_vars()[s])->second;
        inputs.emplace_back(std::make_pair(var.initial_value_name, Tensorflow::Tensor::create(prev_state[s], true)));
    }
    session_.run(inputs, initializer_tensor_names_);

    auto end_set_state = std::chrono::steady_clock::now();

    // run softmax calculation
    inputs.clear();
    auto const& word_info = tensor_input_map_.get_info("word");
    inputs.emplace_back(std::make_pair(word_info.tensor_name(), Tensorflow::Tensor::create(words)));
    if (not word_info.seq_length_tensor_name().empty()) {
        inputs.emplace_back(std::make_pair(word_info.seq_length_tensor_name(), Tensorflow::Tensor::create(word_lengths)));
    }
    std::vector<Tensorflow::Tensor> outputs;
    session_.run(inputs, output_tensor_names_, graph_->update_ops(), outputs);

    auto end_score = std::chrono::steady_clock::now();

    // store outputs in score caches
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* cache = requests[r].final_cache;
        for (size_t w = requests[r].length; w > 0;) {
            --w;
            cache->last_used = current_time_;
            auto& scores = cache->scores;
            outputs[0ul].get(r, w, scores);
            if (output_transform_function_) {
                std::transform(scores.begin(), scores.end(), scores.begin(), output_transform_function_);
            }
            require_eq(scores.size(), num_outputs_);
            cache->computed.store(true);
            cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(cache->parent.handle()));
        }
        require_eq(cache, requests[r].initial_cache);
    }

    auto end_set_score = std::chrono::steady_clock::now();

    // fetch new values of state variables, needs to be done in separate Session::run call (for GPU devices)
    // TODO: atm the model only returns the final state, thus if we have fwd. a multiword sequence we do not get the state of the intermediate words
    session_.run({}, read_vars_tensor_names_, {}, outputs);
    for (size_t s = 0ul; s < prev_state.size(); s++) {
        for (size_t r = 0ul; r < requests.size(); r++) {
            outputs[s].get(r, requests[r].final_cache->state[s]);
        }
    }
    
    auto end_set_new_state = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration = end_set_new_state - end_prepare;
    run_time_ [requests.size()-1ul] += duration.count();
    run_count_[requests.size()-1ul] += 1ul;

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
    stats.total_duration         = std::chrono::duration<double, std::milli>(end                - start);
    stats.early_request_duration = std::chrono::duration<double, std::milli>(end_early_requests - start);
    stats.request_duration       = std::chrono::duration<double, std::milli>(end_requests       - end_early_requests);
    stats.prepare_duration       = std::chrono::duration<double, std::milli>(end_prepare        - end_requests);
    stats.set_state_duration     = std::chrono::duration<double, std::milli>(end_set_state      - end_prepare);
    stats.run_score_duration     = std::chrono::duration<double, std::milli>(end_score          - end_set_state);
    stats.set_score_duration     = std::chrono::duration<double, std::milli>(end_set_score      - end_score);
    stats.set_new_state_duration = std::chrono::duration<double, std::milli>(end_set_new_state  - end_set_score);
    if (verbose_) {
        stats.write(std::cerr);
        std::cerr << " #pr:" << num_pending_requests
                  << " #er:" << num_early_requests
                  << " #r:"  << requests.size() << std::endl;
    }
    fwd_statistics_ += stats;
}

} // namespace Lm

