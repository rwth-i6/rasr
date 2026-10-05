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

#include "LlmScoreCache.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Core/Application.hh>

namespace Search {

namespace {

constexpr LlmHistory rootNode = 0u;

u64 childKey(LlmHistory parent, LlmToken token) {
    return (static_cast<u64>(parent) << 32) | static_cast<u32>(token);
}

}  // namespace

LlmScoreCache::LlmScoreCache()
        : scorer_(),
          nodes_(),
          children_(),
          initialHistory_(rootNode),
          initialDepth_(0u),
          sentenceEndTokens_(),
          tokenizations_(),
          numTokenizationRequests_(0ul),
          numTokenizationMisses_(0ul),
          numScoreRequests_(0ul),
          numScoreMisses_(0ul),
          numScoredTokens_(0ul) {
}

void LlmScoreCache::setScorer(Core::Ref<LlmScorer> scorer) {
    scorer_ = scorer;
    tokenizations_.clear();
    nodes_.clear();
    children_.clear();
}

void LlmScoreCache::reset() {
    require(scorer_);
    scorer_->reset();

    nodes_.clear();
    children_.clear();
    nodes_.push_back({.parent = rootNode, .token = 0, .depth = 0u, .cost = 0.0});

    // The initial tokens are context only and are never scored themselves
    initialHistory_ = rootNode;
    for (LlmToken token : scorer_->initialTokens()) {
        initialHistory_              = extend(initialHistory_, token);
        nodes_[initialHistory_].cost = 0.0;
    }
    initialDepth_ = nodes_[initialHistory_].depth;

    sentenceEndTokens_ = scorer_->sentenceEndTokens();
}

LlmHistory LlmScoreCache::extend(LlmHistory history, LlmToken token) {
    auto [it, inserted] = children_.try_emplace(childKey(history, token), static_cast<LlmHistory>(nodes_.size()));
    if (inserted) {
        if (nodes_.size() >= std::numeric_limits<LlmHistory>::max()) {
            Core::Application::us()->criticalError("Too many LLM histories in one segment");
        }
        nodes_.push_back({.parent = history,
                          .token  = token,
                          .depth  = nodes_[history].depth + 1u,
                          .cost   = std::numeric_limits<Score>::quiet_NaN()});
    }
    return it->second;
}

void LlmScoreCache::tokenize(std::vector<std::string> const& texts, std::vector<LlmTokenSequenceVariants const*>& tokenizations) {
    require(scorer_);
    numTokenizationRequests_ += texts.size();

    missTexts_.clear();
    for (auto const& text : texts) {
        if (tokenizations_.find(text) == tokenizations_.end() and std::find(missTexts_.begin(), missTexts_.end(), text) == missTexts_.end()) {
            missTexts_.push_back(text);
        }
    }

    if (not missTexts_.empty()) {
        numTokenizationMisses_ += missTexts_.size();
        auto result = scorer_->tokenize(missTexts_);
        if (result.size() != missTexts_.size()) {
            Core::Application::us()->criticalError("LLM scorer returned %zu tokenizations for %zu texts", result.size(), missTexts_.size());
        }
        for (size_t i = 0ul; i < missTexts_.size(); ++i) {
            if (result[i].empty()) {
                Core::Application::us()->criticalError("LLM scorer returned no tokenization for \"%s\"", missTexts_[i].c_str());
            }
            for (auto const& variant : result[i]) {
                // An empty variant would make the word free
                if (variant.empty()) {
                    Core::Application::us()->criticalError("LLM scorer returned an empty tokenization for \"%s\"", missTexts_[i].c_str());
                }
            }
            tokenizations_.emplace(missTexts_[i], std::move(result[i]));
        }
    }

    tokenizations.clear();
    for (auto const& text : texts) {
        tokenizations.push_back(&tokenizations_.at(text));
    }
}

void LlmScoreCache::score(std::vector<Request> const& requests, std::vector<Result>& results) {
    require(scorer_);
    numScoreRequests_ += requests.size();

    // Walk down the trie for every request. A request whose final node is already scheduled is the
    // same (history, tokens) pair as an earlier one and is answered by the same LLM call.
    requestNodes_.resize(requests.size());
    missPrefixes_.clear();
    missContinuations_.clear();
    missRequests_.clear();
    std::unordered_map<LlmHistory, size_t> scheduledFinalNodes;

    for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
        auto const& request = requests[requestIdx];
        auto&       path    = requestNodes_[requestIdx];
        path.clear();

        LlmHistory node     = request.history;
        bool       complete = true;
        for (LlmToken token : request.tokens) {
            node = extend(node, token);
            path.push_back(node);
            complete = complete and not std::isnan(nodes_[node].cost);
        }

        if (complete or path.empty()) {
            continue;
        }
        if (scheduledFinalNodes.emplace(path.back(), requestIdx).second) {
            missRequests_.push_back(requestIdx);
            missPrefixes_.emplace_back();
            historyTokens(request.history, missPrefixes_.back());
            missContinuations_.push_back(request.tokens);
        }
    }

    if (not missRequests_.empty()) {
        numScoreMisses_ += missRequests_.size();
        auto costs = scorer_->scoreContinuations(missPrefixes_, missContinuations_);
        if (costs.size() != missRequests_.size()) {
            Core::Application::us()->criticalError("LLM scorer returned %zu results for %zu requests", costs.size(), missRequests_.size());
        }
        for (size_t missIdx = 0ul; missIdx < missRequests_.size(); ++missIdx) {
            auto const& path = requestNodes_[missRequests_[missIdx]];
            if (costs[missIdx].size() != path.size()) {
                Core::Application::us()->criticalError("LLM scorer returned %zu token costs for a continuation of %zu tokens", costs[missIdx].size(), path.size());
            }
            numScoredTokens_ += path.size();
            for (size_t i = 0ul; i < path.size(); ++i) {
                if (std::isnan(costs[missIdx][i])) {
                    Core::Application::us()->criticalError("LLM scorer returned a NaN token cost");
                }
                // Keep a cost that is already known, so hypotheses scored earlier stay consistent even if
                // the LLM is not bit-exact between batches
                if (std::isnan(nodes_[path[i]].cost)) {
                    nodes_[path[i]].cost = costs[missIdx][i];
                }
            }
        }
    }

    results.clear();
    for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
        Score cost = 0.0;
        for (LlmHistory node : requestNodes_[requestIdx]) {
            cost += nodes_[node].cost;
        }
        results.push_back({.cost    = cost,
                           .history = requestNodes_[requestIdx].empty() ? requests[requestIdx].history : requestNodes_[requestIdx].back()});
    }
}

void LlmScoreCache::historyTokens(LlmHistory history, LlmTokenSequence& tokens) const {
    tokens.resize(nodes_[history].depth);
    for (LlmHistory node = history; node != rootNode; node = nodes_[node].parent) {
        tokens[nodes_[node].depth - 1u] = nodes_[node].token;
    }
}

u32 LlmScoreCache::historyLength(LlmHistory history) const {
    return nodes_[history].depth - initialDepth_;
}

void LlmScoreCache::clearStatistics() {
    numTokenizationRequests_ = 0ul;
    numTokenizationMisses_   = 0ul;
    numScoreRequests_        = 0ul;
    numScoreMisses_          = 0ul;
    numScoredTokens_         = 0ul;
}

void LlmScoreCache::logStatistics(Core::XmlWriter& channel) const {
    channel << Core::XmlOpen("llm-statistics");
    channel << Core::XmlFull("tokenization-requests", numTokenizationRequests_);
    channel << Core::XmlFull("tokenization-cache-misses", numTokenizationMisses_);
    channel << Core::XmlFull("score-requests", numScoreRequests_);
    channel << Core::XmlFull("score-cache-misses", numScoreMisses_);
    channel << Core::XmlFull("scored-tokens", numScoredTokens_);
    channel << Core::XmlFull("histories", nodes_.size());
    channel << Core::XmlClose("llm-statistics");
}

}  // namespace Search
