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

#ifndef LLM_HISTORY_TRIE_HH
#define LLM_HISTORY_TRIE_HH

#include <cmath>
#include <unordered_map>
#include <vector>

#include "LlmScorer.hh"

namespace Search {

/*
 * All LLM histories of a segment as a trie. Each node is the history of its parent extended by one token and
 * stores that token's cost once it is known.
 */
class LlmHistoryTrie {
public:
    LlmHistoryTrie();

    // Drop all histories and start over with the given initial tokens
    void reset(LlmTokenSequence const& initialTokens);

    LlmHistory initialHistory() const {
        return initialHistory_;
    }

    // `history` extended by `token`, created if necessary
    LlmHistory extend(LlmHistory history, LlmToken token);

    bool hasCost(LlmHistory history) const {
        return not std::isnan(nodes_[history].cost);
    }

    // Cost of the last token of `history`
    Score cost(LlmHistory history) const {
        return nodes_[history].cost;
    }

    void setCost(LlmHistory history, Score cost) {
        nodes_[history].cost = cost;
    }

    LlmToken lastToken(LlmHistory history) const {
        return nodes_[history].token;
    }

    // Number of tokens after the initial tokens
    u32 length(LlmHistory history) const {
        return nodes_[history].depth - nodes_[initialHistory_].depth;
    }

    // Full token sequence including the initial tokens
    void tokens(LlmHistory history, LlmTokenSequence& tokens) const;

    size_t size() const {
        return nodes_.size();
    }

private:
    struct Node {
        LlmHistory parent;
        LlmToken   token;
        u32        depth;
        Score      cost;  // NaN while unknown
    };

    std::vector<Node>                   nodes_;
    std::unordered_map<u64, LlmHistory> children_;  // (parent << 32 | token) -> child
    LlmHistory                          initialHistory_;
};

}  // namespace Search

#endif  // LLM_HISTORY_TRIE_HH
