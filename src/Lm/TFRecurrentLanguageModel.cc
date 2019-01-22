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

namespace {
    struct ScoresWithContext : public Lm::NNCacheWithStats {
        Lm::History                        parent;
        Math::FastVector<Lm::Score>        scores;
        std::vector<Math::FastVector<f32>> state; // for now float is hardcoded, TODO: replace this with Tensor and add merge utility for tensors
        Lm::SearchSpaceInformation         info;
    };

    struct FwdRequest {
        ScoresWithContext* initial_cache;
        ScoresWithContext* final_cache;
        size_t             length;

        bool operator==(FwdRequest const& other) const {
            return final_cache == other.final_cache;
        }
    };

    void add_request(std::vector<FwdRequest>& requests, ScoresWithContext* cache) {
        FwdRequest tmp_new_request{const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(cache->parent.handle())), cache, 1ul};
        FwdRequest* new_request = &tmp_new_request;
        // determine the first parent where we have already computed the context
        bool inserted = false; // actually this is equivalent to new_request == &tmp_new_request, but inserted is more clear
        while (new_request->initial_cache->state.empty()) { // the parent does not have it's scores computed yet
            // check if the parent is present in the requests vector; if so, replace it
            auto iter = requests.begin();
            // linear search, the vector is small, so this is probably OK
            while (iter != requests.end() and iter->final_cache != new_request->initial_cache) {
                ++iter;
            }
            if (iter != requests.end()) {
                // extend the entry that we have found
                iter->final_cache = cache;
                iter->length += new_request->length;
                new_request = &(*iter);  // TODO: think about how to handle scores used for pruning
                inserted = true;
                break;
            }
            else {
                // continue recursion
                new_request->initial_cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(new_request->initial_cache->parent.handle()));
                new_request->length += 1;
            }
        }
        if (not inserted) {
            requests.push_back(*new_request);
        }
    }
} // namespace

namespace Lm {

Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputLog    ("transform-output-log",    "apply log to tensorflow output",       false);
Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputNegate ("transform-output-negate", "negate tensorflow output (after log)", false);
Core::ParameterInt    TFRecurrentLanguageModel::paramMinBatchSize         ("min-batch-size",          "minimum number of histories forwarded in one go",   32);
Core::ParameterInt    TFRecurrentLanguageModel::paramOptBatchSize         ("opt-batch-size",          "optimum number of histories forwarded in one go",  128);
Core::ParameterInt    TFRecurrentLanguageModel::paramMaxBatchSize         ("max-batch-size",          "maximum number of histories forwarded in one go", 2048);
Core::ParameterFloat  TFRecurrentLanguageModel::paramBatchPruningThreshold("batch-pruning-threshold", "pruning threshold for all hypothesis beyond min-batch-size during eager forwarding", 10.0);
Core::ParameterBool   TFRecurrentLanguageModel::paramAllowReducedHistory  ("allow-reduced-history",   "wether this LM will actually reduce the history length", false);
Core::ParameterBool   TFRecurrentLanguageModel::paramDumpScores           ("dump-scores",             "write all scores from this LM to disk", false);
Core::ParameterString TFRecurrentLanguageModel::paramDumpScoresPrefix     ("dump-scores-prefix",      "prefix for the score dumps",            "scores");

TFRecurrentLanguageModel::TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
                         : Core::Component(c), Precursor(c, l),
                           transform_output_log_(paramTransformOuputLog(config)), transform_output_negate_(paramTransformOuputNegate(config)),
                           min_batch_size_(paramMinBatchSize(config)), opt_batch_size_(paramOptBatchSize(config)), max_batch_size_(paramMaxBatchSize(config)),
                           batch_pruning_threshold_(paramBatchPruningThreshold(config)), allow_reduced_history_(paramAllowReducedHistory(config)),
                           dump_scores_(paramDumpScores(config)), dump_scores_prefix_(paramDumpScoresPrefix(config)),
                           session_(select("session")), loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
                           graph_(loader_->load_graph()), tensor_input_map_(select("input-map")), tensor_output_map_(select("output-map")),
                           run_time_(max_batch_size_, 0.0), run_count_(max_batch_size_, 0ul) {
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
    cache->scores.resize(1u); // pretend this history has already been evaluated
    empty_history_ = history(h);
}

TFRecurrentLanguageModel::~TFRecurrentLanguageModel() {
    size_t total_run_count = 0ul;
    size_t total_fwd_hist  = 0ul;
    double total_run_time  = 0.0;

    Core::XmlChannel out(config, "statistics");
    out << Core::XmlOpen("fwd-time");
    for (size_t i = 0ul; i < run_count_.size(); i++) {
        if (run_count_[i] > 0ul) {
            out << (i + 1) << " " << run_count_[i] << " " << run_time_[i] << "\n";
            total_run_count += run_count_[i];
            total_fwd_hist  += (i+1) * run_count_[i];
            total_run_time  += run_time_[i];
        }
    }
    out << Core::XmlClose("fwd-time");

    out << Core::XmlOpen("fwd-summary");
    out << Core::XmlOpen("total-run-count") << total_run_count << Core::XmlClose("total-run-count");
    out << Core::XmlOpen("total-fwd-hist")  << total_fwd_hist  << Core::XmlClose("total-fwd-hist");
    out << Core::XmlOpen("total-run-time")  << total_run_time  << Core::XmlClose("total-run-time");
    out << Core::XmlClose("fwd-summary");
}

History TFRecurrentLanguageModel::startHistory() const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence ts(1ul, lexicon_mapping_[sentenceBeginToken()->id()]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->parent = empty_history_;
    return history(h);
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Token w) const {
    return extendedHistory(hist, w->id());
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Bliss::Token::Id w) const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    TokenIdSequence ts(*sc->history);
    ts.push_back(lexicon_mapping_[w]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    if (cache->parent.handle() == nullptr) {
        cache->parent = hist;
    }
    return history(h);
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
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    size_t output_idx = lexicon_mapping_[w->id()];
    useOutput(*sc, output_idx);
    if (not sc->scores.empty()) {
        return sc->scores[output_idx];
    }

    std::vector<FwdRequest> requests;
    add_request(requests, const_cast<ScoresWithContext*>(sc)); 

    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    NNHistoryManager::NNCacheMap const& cm = hm->getNNCacheMap();

    std::vector<ScoresWithContext*> early_requests;

    for (auto const& kv : cm) {
        ScoresWithContext* c = const_cast<ScoresWithContext*>(static_cast<ScoresWithContext const*>(kv.second));
        if (c->scores.empty() and c != sc) {
            early_requests.emplace_back(c);
        }
    }

    std::sort(early_requests.begin(), early_requests.end(), [](ScoresWithContext* a, ScoresWithContext* b) {
        return a->info.bestScoreOffset < b->info.bestScoreOffset;
    });

    for (auto r : early_requests) {
        add_request(requests, r);
    }

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
    size_t max_length = 0ul;
    for (auto const& r : requests) {
        max_length = std::max(max_length, r.length);
    }

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

    // measure times - tick
    auto timer_start = std::chrono::high_resolution_clock::now();

    // build tensors + set state variables
    std::vector<std::pair<std::string, Tensorflow::Tensor>> inputs;
    inputs.reserve(prev_state.size());
    for (size_t s = 0ul; s < prev_state.size(); s++) {
        auto const& var = graph_->variables().find(graph_->state_vars()[s])->second;
        inputs.emplace_back(std::make_pair(var.initial_value_name, Tensorflow::Tensor::create(prev_state[s], true)));
    }
    session_.run(inputs, initializer_tensor_names_);

    // run softmax calculation
    inputs.clear();
    auto const& word_info = tensor_input_map_.get_info("word");
    inputs.emplace_back(std::make_pair(word_info.tensor_name(), Tensorflow::Tensor::create(words)));
    if (not word_info.seq_length_tensor_name().empty()) {
        inputs.emplace_back(std::make_pair(word_info.seq_length_tensor_name(), Tensorflow::Tensor::create(word_lengths)));
    }
    std::vector<Tensorflow::Tensor> outputs;
    session_.run(inputs, output_tensor_names_, graph_->update_ops(), outputs);

    // store outputs in score caches
    for (size_t r = 0ul; r < requests.size(); r++) {
        ScoresWithContext* cache = requests[r].final_cache;
        for (size_t w = requests[r].length; w > 0;) {
            --w;
            cache->state.resize(prev_state.size());
            auto& scores = cache->scores;
            outputs[0ul].get(r, w, scores);
            if (output_transform_function_) {
                std::transform(scores.begin(), scores.end(), scores.begin(), output_transform_function_);
            }
            require_eq(scores.size(), num_outputs_);
            cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(cache->parent.handle()));
        }
        require_eq(cache, requests[r].initial_cache);
    }

    // fetch new values of state variables, needs to be done in separate Session::run call (for GPU devices)
    // TODO: atm the model only returns the final state, thus if we have fwd. a multiword sequence we do not get the state of the intermediate words
    session_.run({}, read_vars_tensor_names_, {}, outputs);
    for (size_t s = 0ul; s < prev_state.size(); s++) {
        for (size_t r = 0ul; r < requests.size(); r++) {
            outputs[s].get(r, requests[r].final_cache->state[s]);
        }
    }

    // measure times - tock
    auto timer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = timer_end - timer_start;
    run_time_ [requests.size()-1ul] += duration.count();
    run_count_[requests.size()-1ul] += 1ul;

    if (dump_scores_) {
        for (auto const& r : requests) {
            std::stringstream path;
            path << dump_scores_prefix_;
            for (auto token : *r.final_cache->history) {
                path << "_" << token;
            }
            std::ofstream out(path.str(), std::ios::out | std::ios::trunc);
            out << "scores:\n";
            for (auto score : r.final_cache->scores) {
                out << score << '\n';
            }
            for (size_t s = 0ul; s < r.final_cache->state.size(); s++) {
                out << "state " << s << ":\n";
                for (auto v : r.final_cache->state[s]) {
                    out << v << '\n';
                }
            }
        }
    }

    return sc->scores[output_idx];
}

bool TFRecurrentLanguageModel::scoreCached(History const& hist, Token w) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    return not sc->scores.empty();
}

void TFRecurrentLanguageModel::load() {
    loadVocabulary();
}

void TFRecurrentLanguageModel::startFrame(Search::TimeframeIndex time) const {
}

void TFRecurrentLanguageModel::setInfo(History const& hist, SearchSpaceInformation const& info) const {
    ScoresWithContext* sc = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(hist.handle()));
    sc->info = info;
}

} // namespace Lm

