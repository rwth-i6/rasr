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
    };
} // namespace

namespace Lm {

Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputLog   ("transform-output-log",    "apply log to tensorflow output",                 false);
Core::ParameterBool   TFRecurrentLanguageModel::paramTransformOuputNegate("transform-output-negate", "negate tensorflow output (after log)",           false);
Core::ParameterInt    TFRecurrentLanguageModel::paramMaxBatchSize        ("max-batch-size",          "maximum number of histories forwarded in one go", 2048);
Core::ParameterBool   TFRecurrentLanguageModel::paramDumpScores          ("dump-scores",             "write all scores from this LM to disk",          false);
Core::ParameterString TFRecurrentLanguageModel::paramDumpScoresPrefix    ("dump-scores-prefix",      "prefix for the score dumps",                  "scores");

TFRecurrentLanguageModel::TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
                         : Core::Component(c), Precursor(c, l),
                           transform_output_log_(paramTransformOuputLog(config)), transform_output_negate_(paramTransformOuputNegate(config)),
                           max_batch_size_(paramMaxBatchSize(config)), dump_scores_(paramDumpScores(config)), dump_scores_prefix_(paramDumpScoresPrefix(config)),
                           session_(select("session")), loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
                           graph_(loader_->load_graph()), tensor_input_map_(select("input-map")), tensor_output_map_(select("output-map")) {
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

History TFRecurrentLanguageModel::startHistory() const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence ts(1ul, lexicon_mapping_[sentenceBeginToken()->id()]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    cache->parent = empty_history_;
    return history(h);
}

History TFRecurrentLanguageModel::extendedHistory(History const& hist, Token w) const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    TokenIdSequence ts(*sc->history);
    ts.push_back(lexicon_mapping_[w->id()]);
    HistoryHandle h = hm->get<ScoresWithContext>(ts);
    ScoresWithContext* cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(h));
    if (cache->parent.handle() == nullptr) {
        cache->parent = hist;
    }
    return history(h);
}

Score TFRecurrentLanguageModel::score(History const& hist, Token w) const {
    ScoresWithContext const* sc = reinterpret_cast<ScoresWithContext const*>(hist.handle());
    size_t output_idx = lexicon_mapping_[w->id()];
    useOutput(*sc, output_idx);
    if (not sc->scores.empty()) {
        return sc->scores[output_idx];
    }

    std::vector<ScoresWithContext*> caches;
    caches.push_back(const_cast<ScoresWithContext*>(sc));
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    NNHistoryManager::NNCacheMap const& cm = hm->getNNCacheMap();
    for (auto const& kv : cm) {
        ScoresWithContext* c = const_cast<ScoresWithContext*>(static_cast<ScoresWithContext const*>(kv.second));
        if (c->scores.empty() and c != sc) {
            caches.push_back(c);
        }
        if (caches.size() >= max_batch_size_) {
            break;
        }
    }

    // prepare the data in Sprint Datastructures
    Math::FastMatrix<s32>              words(caches.size(), 1u);
    std::vector<Math::FastMatrix<f32>> prev_state;
    prev_state.reserve(graph_->state_vars().size());
    for (size_t c = 0ul; c < caches.size(); c++) {
        words.at(c, 0u) = static_cast<s32>(caches[c]->history->back());
        ScoresWithContext* parent_cache = const_cast<ScoresWithContext*>(reinterpret_cast<ScoresWithContext const*>(caches[c]->parent.handle()));
        require(parent_cache != nullptr);
        require_eq(graph_->state_vars().size(), parent_cache->state.size()); // this will be violated if a lemma has multiple syntactic tokens, TODO: traverse parents and process multiple words
        for (size_t s = 0ul; s < graph_->state_vars().size(); s++) {
            if (s >= prev_state.size()) {
                prev_state.emplace_back(parent_cache->state[s].size(), caches.size());
            }
            else {
                require_eq(parent_cache->state[s].size(), prev_state[s].nRows());
            }
            // we place the state for each history in columns and transpose later
            std::copy(parent_cache->state[s].begin(), parent_cache->state[s].end(), &prev_state[s].at(0u, c));
        }
    }

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
        Math::FastVector<s32> word_lengths(caches.size());
        std::fill(word_lengths.begin(), word_lengths.end(), 1);
        inputs.emplace_back(std::make_pair(word_info.seq_length_tensor_name(), Tensorflow::Tensor::create(word_lengths)));
    }
    std::vector<Tensorflow::Tensor> outputs;
    session_.run(inputs, output_tensor_names_, graph_->update_ops(), outputs);

    // store outputs in score caches
    for (size_t c = 0ul; c < caches.size(); c++) {
        caches[c]->state.resize(prev_state.size());
        auto& scores = caches[c]->scores;
        outputs[0ul].get(c, 0ul, scores);
        if (output_transform_function_) {
            std::transform(scores.begin(), scores.end(), scores.begin(), output_transform_function_);
        }
        require_eq(scores.size(), num_outputs_);
    }

    // fetch new values of state variables, needs to be done in separate Session::run call (for GPU devices)
    session_.run({}, read_vars_tensor_names_, {}, outputs);
    for (size_t s = 0ul; s < prev_state.size(); s++) {
        for (size_t c = 0ul; c < caches.size(); c++) {
            outputs[s].get(c, caches[c]->state[s]);
        }
    }

    if (dump_scores_) {
        for (auto const& c : caches) {
            std::stringstream path;
            path << dump_scores_prefix_;
            for (auto token : *c->history) {
                path << "_" << token;
            }
            std::ofstream out(path.str(), std::ios::out | std::ios::trunc);
            out << "scores:\n";
            for (auto score : c->scores) {
                out << score << '\n';
            }
            for (size_t s = 0ul; s < c->state.size(); s++) {
                out << "state " << s << ":\n";
                for (auto v : c->state[s]) {
                    out << v << '\n';
                }
            }
        }
    }

    useOutput(*sc, output_idx);
    return sc->scores[output_idx];
}

void TFRecurrentLanguageModel::load() {
    loadVocabulary();
}

} // namespace Lm

