#include "RecurrentLanguageModel.hh"

namespace Lm::detail {

void RequestGraph::add_cache(ScoresWithContext* cache) {
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

void RequestGraph::get_requests_dfs(std::vector<FwdRequest>& requests, ScoresWithContext* initial, size_t entry, size_t length) const {
    if (children[entry].empty()) {
        requests.emplace_back(FwdRequest{initial, entries[entry], length});
    }
    else {
        for (size_t e : children[entry]) {
            get_requests_dfs(requests, initial, e, length + 1ul);
        }
    }
}

std::vector<FwdRequest> RequestGraph::get_requests() const {
    std::vector<FwdRequest> result;
    for (size_t r : roots) {
        for (size_t c : children[r]) {
            get_requests_dfs(result, entries[r], c, 1ul);
        }
    }
    return result;
}

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

TimeStatistics TimeStatistics::operator+(TimeStatistics const& other) const {
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

TimeStatistics& TimeStatistics::operator+=(TimeStatistics const& other) {
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

void TimeStatistics::write(Core::XmlChannel& channel) const {
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

void TimeStatistics::write(std::ostream& out) const {
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

}  // namespace Lm::detail
