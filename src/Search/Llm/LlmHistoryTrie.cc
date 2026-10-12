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

#include "LlmHistoryTrie.hh"

#include <limits>

#include <Core/Application.hh>

namespace Search {

namespace {

constexpr LlmHistory rootHistory = 0u;

u64 childKey(LlmHistory parent, LlmToken token) {
    return (static_cast<u64>(parent) << 32) | static_cast<u32>(token);
}

}  // namespace

LlmHistoryTrie::LlmHistoryTrie()
        : nodes_(), children_(), initialHistory_(rootHistory) {
}

void LlmHistoryTrie::reset(LlmTokenSequence const& initialTokens) {
    nodes_.clear();
    children_.clear();
    nodes_.push_back({.parent = rootHistory, .token = 0, .depth = 0u, .cost = 0.0});

    // The initial tokens are context only and are never scored
    initialHistory_ = rootHistory;
    for (LlmToken token : initialTokens) {
        initialHistory_ = extend(initialHistory_, token);
        setCost(initialHistory_, 0.0);
    }
}

LlmHistory LlmHistoryTrie::extend(LlmHistory history, LlmToken token) {
    auto [it, inserted] = children_.try_emplace(childKey(history, token), static_cast<LlmHistory>(nodes_.size()));
    if (inserted) {
        if (nodes_.size() == std::numeric_limits<LlmHistory>::max()) {
            Core::Application::us()->criticalError("Too many LLM histories in one segment");
        }
        nodes_.push_back({.parent = history, .token = token, .depth = nodes_[history].depth + 1u, .cost = std::numeric_limits<Score>::quiet_NaN()});
    }
    return it->second;
}

void LlmHistoryTrie::tokens(LlmHistory history, LlmTokenSequence& tokens) const {
    tokens.resize(nodes_[history].depth);
    for (LlmHistory node = history; node != rootHistory; node = nodes_[node].parent) {
        tokens[nodes_[node].depth - 1u] = nodes_[node].token;
    }
}

}  // namespace Search
