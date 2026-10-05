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

#ifndef LLM_SCORE_CACHE_HH
#define LLM_SCORE_CACHE_HH

#include <string>
#include <unordered_map>
#include <vector>

#include <Core/XmlStream.hh>

#include "LlmScorer.hh"

namespace Search {

/*
 * Handle of an LLM history, i.e. of a token sequence starting with the initial tokens of the LLM.
 * Two hypotheses have the same LLM history if and only if their handles are equal.
 */
typedef u32 LlmHistory;

/*
 * Caches for scoring words with an `LlmScorer`:
 *
 *  - Tokenizations (all variants), by text. Tokenization does not depend on the segment, so these are kept across
 *    segments.
 *  - Token costs, by history plus token. All histories of a segment form a trie whose nodes are the handles
 *    of `LlmHistory`. A node stores the cost of its last token given its parent, once it is known. Scoring a
 *    token sequence after some history is therefore a walk down the trie, and a sequence which was scored
 *    before -- also as a prefix of a longer one -- is never sent to the LLM again. The trie is cleared at
 *    every segment start.
 *
 * `score` batches all requests that are not fully cached into one call to the LLM.
 */
class LlmScoreCache {
public:
    // A token sequence to score after some history.
    struct Request {
        LlmHistory       history;
        LlmTokenSequence tokens;
    };

    // Result of a `Request`.
    struct Result {
        Score      cost;     // Sum of the (unscaled) costs of all tokens of the request
        LlmHistory history;  // `history` of the request extended by all of its tokens
    };

    LlmScoreCache();

    // Set the scorer to use. Clears all caches.
    void setScorer(Core::Ref<LlmScorer> scorer);

    // Prepare for a new segment: clear the history trie and re-insert the initial tokens of the scorer.
    void reset();

    // History consisting of the initial tokens only.
    LlmHistory initialHistory() const {
        return initialHistory_;
    }

    // Tokens scored once at the end of a segment.
    LlmTokenSequence const& sentenceEndTokens() const {
        return sentenceEndTokens_;
    }

    // Tokenize all texts, asking the scorer only for those that are not cached yet.
    // The variants of `texts[i]` are `*tokenizations[i]`; the pointers stay valid until `setScorer`.
    void tokenize(std::vector<std::string> const& texts, std::vector<LlmTokenSequenceVariants const*>& tokenizations);

    // Score all requests, asking the scorer only for those that are not cached yet.
    void score(std::vector<Request> const& requests, std::vector<Result>& results);

    // Full token sequence of a history, including the initial tokens.
    void historyTokens(LlmHistory history, LlmTokenSequence& tokens) const;

    // Number of tokens of a history after the initial tokens.
    u32 historyLength(LlmHistory history) const;

    size_t numHistories() const {
        return nodes_.size();
    }

    void clearStatistics();
    void logStatistics(Core::XmlWriter& channel) const;

private:
    struct Node {
        LlmHistory parent;
        LlmToken   token;
        u32        depth;  // Number of tokens of this history, including the initial tokens
        Score      cost;   // Cost of `token` given `parent`; NaN while unknown
    };

    // History extended by `token`, creating the trie node if necessary
    LlmHistory extend(LlmHistory history, LlmToken token);

    Core::Ref<LlmScorer> scorer_;

    std::vector<Node>                   nodes_;
    std::unordered_map<u64, LlmHistory> children_;  // (parent << 32 | token) -> child
    LlmHistory                          initialHistory_;
    u32                                 initialDepth_;
    LlmTokenSequence                    sentenceEndTokens_;

    // Node-based, so references to the values stay valid when the map grows
    std::unordered_map<std::string, LlmTokenSequenceVariants> tokenizations_;

    // Scratch buffers for `tokenize` and `score`
    std::vector<std::string>             missTexts_;
    std::vector<std::vector<LlmHistory>> requestNodes_;
    std::vector<LlmTokenSequence>        missPrefixes_;
    std::vector<LlmTokenSequence>        missContinuations_;
    std::vector<size_t>                  missRequests_;

    size_t numTokenizationRequests_;
    size_t numTokenizationMisses_;
    size_t numScoreRequests_;
    size_t numScoreMisses_;
    size_t numScoredTokens_;
};

}  // namespace Search

#endif  // LLM_SCORE_CACHE_HH
