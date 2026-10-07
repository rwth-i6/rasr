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

#ifndef LLM_WORD_SCORER_HH
#define LLM_WORD_SCORER_HH

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Core/Channel.hh>
#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Core/Statistics.hh>
#include <Core/StopWatch.hh>

#include "LlmHistoryTrie.hh"
#include "LlmScorer.hh"

namespace Search {

/*
 * Scores words after LLM histories with an `LlmScorer`. All spelling variants of a word are scored and the
 * cheapest one is kept. Spelling variants and tokenizations are cached across segments, word results and token
 * costs per segment, and everything not cached is sent to the LLM in one batch per call.
 */
class LlmWordScorer : public Core::Component {
public:
    static const Core::ParameterString paramWordSeparator;

    struct Request {
        LlmHistory         history;
        std::string const* word;         // Null for no word
        bool               sentenceEnd;  // Also score the sentence-end tokens after the word
    };

    struct Result {
        Score      cost;     // Unscaled
        LlmHistory history;  // `history` extended by the cheapest variant
    };

    LlmWordScorer(Core::Configuration const& config);

    // Start a new segment
    void reset();

    LlmHistory initialHistory() const {
        return trie_.initialHistory();
    }

    void score(std::vector<Request> const& requests, std::vector<Result>& results);

    // Only `activeHistories` are continued from now on
    void cleanup(std::vector<LlmHistory> const& activeHistories);

    void logStatistics() const;

private:
    struct Word {
        std::string const*            spelling;  // Key of the entry in `words_`
        std::vector<std::string>      variants;
        std::vector<LlmTokenSequence> tokenizations[2];  // Per variant, as first word and after the separator
        u32                           id;
    };

    // Token sequence of one variant of one request, as a range of `variantHistories_`
    struct VariantPath {
        size_t begin;
        size_t end;
    };

    Core::Ref<LlmScorer> scorer_;
    std::string          wordSeparator_;
    LlmTokenSequence     sentenceEndTokens_;
    LlmHistoryTrie       trie_;

    std::unordered_map<std::string, Word>             words_;          // Across segments
    std::unordered_map<std::string, LlmTokenSequence> tokenizations_;  // Across segments
    std::unordered_map<u64, Result>                   wordResults_;    // (history << 32 | word id) -> result

    // Scratch buffers of `score`
    std::vector<Word*>             requestWords_;
    std::vector<size_t>            requestPathOffsets_;
    std::vector<VariantPath>       variantPaths_;
    std::vector<LlmHistory>        variantHistories_;
    std::vector<LlmScoringRequest> llmRequests_;
    std::vector<size_t>            llmRequestPaths_;  // Index into `variantPaths_` of each LLM request
    std::unordered_set<LlmHistory> scheduledHistories_;

    mutable Core::XmlChannel statisticsChannel_;
    Core::StopWatch          scoringTime_;
    Core::StopWatch          spellingTime_;
    Core::StopWatch          tokenizationTime_;
    Core::StopWatch          llmTime_;
    Core::StopWatch          cleanupTime_;
    size_t                   numWordRequests_;
    size_t                   numWordCacheHits_;
    size_t                   numSpelledWords_;
    size_t                   numTokenizedTexts_;
    size_t                   numLlmCalls_;
    size_t                   numScoredTokens_;
    Core::Statistics<u32>    llmBatchSize_;

    bool isFirstWord(LlmHistory history) const {
        return trie_.length(history) == 0u;
    }

    static u64 wordResultKey(LlmHistory history, Word const& word) {
        return (static_cast<u64>(history) << 32) | word.id;
    }

    LlmTokenSequence const& variantTokens(Word const& word, LlmHistory history, size_t variant) const {
        return word.tokenizations[isFirstWord(history) ? 0 : 1][variant];
    }

    // Fill `requestWords_` and make sure spelling variants and tokenizations of all requested words are known
    void prepareWords(std::vector<Request> const& requests);
    void addSpellingVariants(std::vector<Word*> const& words);
    void addTokenizations(std::vector<Request> const& requests);

    // Walk the token sequences of all variants of the uncached requests down the trie and schedule what is missing
    void collectVariantPaths(std::vector<Request> const& requests);
    void scheduleIfMissing(LlmHistory history, VariantPath const& path, size_t pathIndex);

    // Score all scheduled requests with the LLM and store the costs in the trie
    void queryLlm();

    // Result of the cheapest variant of every request; fills the word result cache
    void  selectCheapestVariants(std::vector<Request> const& requests, std::vector<Result>& results);
    Score pathCost(VariantPath const& path) const;
};

}  // namespace Search

#endif  // LLM_WORD_SCORER_HH
