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

#include "LlmWordScorer.hh"

#include <algorithm>

#include <Core/Application.hh>
#include <Core/XmlStream.hh>
#include <Search/Module.hh>

namespace Search {

const Core::ParameterString LlmWordScorer::paramWordSeparator(
        "word-separator",
        "Text in front of every word except the first one of a segment when it is tokenized.",
        " ");

LlmWordScorer::LlmWordScorer(Core::Configuration const& config)
        : Core::Component(config),
          scorer_(Search::Module::instance().llmScorerFactory().createLlmScorer(config)),
          wordSeparator_(paramWordSeparator(config)),
          statisticsChannel_(config, "statistics"),
          numWordRequests_(0ul),
          numWordCacheHits_(0ul),
          numSpelledWords_(0ul),
          numTokenizedTexts_(0ul),
          numLlmCalls_(0ul),
          numScoredTokens_(0ul),
          llmBatchSize_("llm-batch-size") {
}

void LlmWordScorer::reset() {
    scorer_->reset();
    auto initialTokens = scorer_->initialTokens();
    if (initialTokens.empty()) {
        criticalError("The LLM scorer returned no initial tokens");
    }
    trie_.reset(initialTokens);
    sentenceEndTokens_ = scorer_->sentenceEndTokens();
    wordResults_.clear();

    for (auto* timer : {&scoringTime_, &spellingTime_, &tokenizationTime_, &llmTime_, &cleanupTime_}) {
        timer->reset();
    }
    numWordRequests_   = 0ul;
    numWordCacheHits_  = 0ul;
    numSpelledWords_   = 0ul;
    numTokenizedTexts_ = 0ul;
    numLlmCalls_       = 0ul;
    numScoredTokens_   = 0ul;
    llmBatchSize_.clear();
}

void LlmWordScorer::score(std::vector<Request> const& requests, std::vector<Result>& results) {
    Core::StopWatch::Scope timer(scoringTime_);
    prepareWords(requests);
    collectVariantPaths(requests);
    queryLlm();
    selectCheapestVariants(requests, results);
}

void LlmWordScorer::cleanup(std::vector<LlmHistory> const& activeHistories) {
    Core::StopWatch::Scope timer(cleanupTime_);
    scorer_->cleanup(activeHistories);
}

void LlmWordScorer::prepareWords(std::vector<Request> const& requests) {
    std::vector<Word*> unspelled;
    requestWords_.assign(requests.size(), nullptr);
    for (size_t i = 0ul; i < requests.size(); ++i) {
        if (requests[i].word == nullptr) {
            continue;
        }
        ++numWordRequests_;
        auto [it, inserted] = words_.try_emplace(*requests[i].word);
        if (inserted) {
            it->second.spelling = &it->first;
            it->second.id       = words_.size() - 1ul;
            unspelled.push_back(&it->second);
        }
        requestWords_[i] = &it->second;
    }
    addSpellingVariants(unspelled);
    addTokenizations(requests);
}

void LlmWordScorer::addSpellingVariants(std::vector<Word*> const& words) {
    if (words.empty()) {
        return;
    }
    Core::StopWatch::Scope   timer(spellingTime_);
    std::vector<std::string> spellings;
    for (auto const* word : words) {
        spellings.push_back(*word->spelling);
    }
    auto variants = scorer_->spellingVariants(spellings);
    if (variants.size() != words.size()) {
        criticalError("The LLM scorer returned spelling variants for %zu of %zu words", variants.size(), words.size());
    }
    for (size_t i = 0ul; i < words.size(); ++i) {
        if (variants[i].empty()) {
            criticalError("The LLM scorer returned no spelling variant for \"%s\"", spellings[i].c_str());
        }
        words[i]->variants = std::move(variants[i]);
    }
    numSpelledWords_ += words.size();
}

void LlmWordScorer::addTokenizations(std::vector<Request> const& requests) {
    // Words whose variants still need to be tokenized in some form, and the texts that are not cached yet
    std::vector<std::pair<Word*, size_t>> untokenized;
    std::unordered_set<u64>               untokenizedKeys;  // word id * 2 + form
    std::vector<std::string>              texts;
    std::unordered_set<std::string>       uncachedTexts;
    for (size_t i = 0ul; i < requests.size(); ++i) {
        Word* word = requestWords_[i];
        if (word == nullptr) {
            continue;
        }
        size_t form = isFirstWord(requests[i].history) ? 0ul : 1ul;
        if (not word->tokenizations[form].empty() or not untokenizedKeys.insert(word->id * 2ul + form).second) {
            continue;
        }
        untokenized.emplace_back(word, form);
        for (auto const& variant : word->variants) {
            std::string text = (form == 0ul ? "" : wordSeparator_) + variant;
            if (tokenizations_.find(text) == tokenizations_.end() and uncachedTexts.insert(text).second) {
                texts.push_back(std::move(text));
            }
        }
    }
    if (untokenized.empty()) {
        return;
    }

    if (not texts.empty()) {
        Core::StopWatch::Scope timer(tokenizationTime_);
        auto                   tokenized = scorer_->tokenize(texts);
        if (tokenized.size() != texts.size()) {
            criticalError("The LLM scorer returned %zu tokenizations for %zu texts", tokenized.size(), texts.size());
        }
        for (size_t i = 0ul; i < texts.size(); ++i) {
            // An empty tokenization would make the word free
            if (tokenized[i].empty()) {
                criticalError("The LLM scorer returned an empty tokenization for \"%s\"", texts[i].c_str());
            }
            tokenizations_.emplace(texts[i], std::move(tokenized[i]));
        }
        numTokenizedTexts_ += texts.size();
    }

    for (auto [word, form] : untokenized) {
        for (auto const& variant : word->variants) {
            word->tokenizations[form].push_back(tokenizations_.at((form == 0ul ? "" : wordSeparator_) + variant));
        }
    }
}

void LlmWordScorer::collectVariantPaths(std::vector<Request> const& requests) {
    requestPathOffsets_.clear();
    variantPaths_.clear();
    variantHistories_.clear();
    llmRequests_.clear();
    llmRequestPaths_.clear();
    scheduledHistories_.clear();

    for (size_t i = 0ul; i < requests.size(); ++i) {
        auto const& request = requests[i];
        Word const* word    = requestWords_[i];
        requestPathOffsets_.push_back(variantPaths_.size());
        if (word != nullptr and not request.sentenceEnd and wordResults_.count(wordResultKey(request.history, *word))) {
            ++numWordCacheHits_;
            continue;
        }

        size_t numVariants = word != nullptr ? word->variants.size() : 1ul;
        for (size_t variant = 0ul; variant < numVariants; ++variant) {
            VariantPath path{variantHistories_.size(), variantHistories_.size()};
            LlmHistory  history = request.history;
            auto        append  = [&](LlmTokenSequence const& tokens) {
                for (LlmToken token : tokens) {
                    history = trie_.extend(history, token);
                    variantHistories_.push_back(history);
                }
            };
            if (word != nullptr) {
                append(variantTokens(*word, request.history, variant));
            }
            if (request.sentenceEnd) {
                append(sentenceEndTokens_);
            }
            path.end = variantHistories_.size();
            variantPaths_.push_back(path);
            scheduleIfMissing(request.history, path, variantPaths_.size() - 1ul);
        }
    }
    requestPathOffsets_.push_back(variantPaths_.size());
}

void LlmWordScorer::scheduleIfMissing(LlmHistory history, VariantPath const& path, size_t pathIndex) {
    auto begin = variantHistories_.begin() + path.begin;
    auto end   = variantHistories_.begin() + path.end;
    if (std::all_of(begin, end, [this](LlmHistory h) { return trie_.hasCost(h); })) {
        return;
    }
    // Requests with the same final history are the same request
    if (not scheduledHistories_.insert(*(end - 1)).second) {
        return;
    }

    LlmScoringRequest request;
    request.history = history;
    trie_.tokens(history, request.prefix);
    request.tokenHistories.assign(begin, end);
    for (LlmHistory h : request.tokenHistories) {
        request.tokens.push_back(trie_.lastToken(h));
    }
    llmRequests_.push_back(std::move(request));
    llmRequestPaths_.push_back(pathIndex);
}

void LlmWordScorer::queryLlm() {
    if (llmRequests_.empty()) {
        return;
    }
    ++numLlmCalls_;
    llmBatchSize_ += llmRequests_.size();

    std::vector<std::vector<Score>> costs;
    {
        Core::StopWatch::Scope timer(llmTime_);
        costs = scorer_->score(llmRequests_);
    }
    if (costs.size() != llmRequests_.size()) {
        criticalError("The LLM scorer returned %zu results for %zu requests", costs.size(), llmRequests_.size());
    }

    for (size_t i = 0ul; i < llmRequests_.size(); ++i) {
        auto const& histories = llmRequests_[i].tokenHistories;
        if (costs[i].size() != histories.size()) {
            criticalError("The LLM scorer returned %zu costs for %zu tokens", costs[i].size(), histories.size());
        }
        for (size_t j = 0ul; j < histories.size(); ++j) {
            if (std::isnan(costs[i][j])) {
                criticalError("The LLM scorer returned a NaN cost");
            }
            // A known cost is kept, so that earlier results stay consistent if the LLM is not bit-exact across batches
            if (not trie_.hasCost(histories[j])) {
                trie_.setCost(histories[j], costs[i][j]);
            }
        }
        numScoredTokens_ += histories.size();
    }
}

void LlmWordScorer::selectCheapestVariants(std::vector<Request> const& requests, std::vector<Result>& results) {
    results.clear();
    for (size_t i = 0ul; i < requests.size(); ++i) {
        auto const& request = requests[i];
        Word const* word    = requestWords_[i];
        if (requestPathOffsets_[i] == requestPathOffsets_[i + 1]) {
            results.push_back(wordResults_.at(wordResultKey(request.history, *word)));
            continue;
        }

        // On ties the first variant wins, so the order of the variants decides
        Result best{Core::Type<Score>::max, request.history};
        for (size_t p = requestPathOffsets_[i]; p < requestPathOffsets_[i + 1]; ++p) {
            auto const& path = variantPaths_[p];
            Score       cost = pathCost(path);
            if (cost < best.cost) {
                best = {cost, path.begin == path.end ? request.history : variantHistories_[path.end - 1]};
            }
        }
        results.push_back(best);
        if (word != nullptr and not request.sentenceEnd) {
            wordResults_.emplace(wordResultKey(request.history, *word), best);
        }
    }
}

Score LlmWordScorer::pathCost(VariantPath const& path) const {
    Score cost = 0.0;
    for (size_t i = path.begin; i < path.end; ++i) {
        cost += trie_.cost(variantHistories_[i]);
    }
    return cost;
}

void LlmWordScorer::logStatistics() const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }
    statisticsChannel_ << Core::XmlOpen("llm-word-scorer-statistics") + Core::XmlAttribute("component", fullName());
    statisticsChannel_ << Core::XmlOpen("scoring-time") + Core::XmlAttribute("unit", "milliseconds") + Core::XmlAttribute("total", scoringTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("spelling-variants-time", spellingTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("tokenization-time", tokenizationTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("llm-time", llmTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlClose("scoring-time");
    statisticsChannel_ << Core::XmlFull("cleanup-time", cleanupTime_.elapsedMilliseconds()) + Core::XmlAttribute("unit", "milliseconds");
    statisticsChannel_ << Core::XmlFull("num-word-requests", numWordRequests_);
    statisticsChannel_ << Core::XmlFull("num-word-cache-hits", numWordCacheHits_);
    statisticsChannel_ << Core::XmlFull("num-spelled-words", numSpelledWords_);
    statisticsChannel_ << Core::XmlFull("num-tokenized-texts", numTokenizedTexts_);
    statisticsChannel_ << Core::XmlFull("num-llm-calls", numLlmCalls_);
    llmBatchSize_.write(statisticsChannel_);
    statisticsChannel_ << Core::XmlFull("num-scored-tokens", numScoredTokens_);
    statisticsChannel_ << Core::XmlFull("num-histories", trie_.size());
    statisticsChannel_ << Core::XmlClose("llm-word-scorer-statistics");
}

}  // namespace Search
