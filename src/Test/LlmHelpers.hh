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

#ifndef TEST_LLM_HELPERS_HH
#define TEST_LLM_HELPERS_HH

#include <Test/Lexicon.hh>

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <vector>

#include <Search/Llm/LlmScorer.hh>
#include <Search/Module.hh>

/*
 * Helpers for the tests of the LLM searches: a deterministic toy LLM and a lexicon of word-piece labels.
 */
namespace Test {

// SentencePiece word-start marker U+2581, for literal concatenation
#define WORD_START "\xE2\x96\x81"

inline constexpr f32           suppressed  = 30.0f;  // Score of every label a frame is not supposed to emit
inline constexpr Search::Score defaultCost = 5.0;    // Cost of a token whose bigram is not listed

// Known texts are single tokens, other texts one token per character; every token costs a configurable bigram cost
class ToyLlm : public Search::LlmScorer {
public:
    static inline std::vector<std::string>                                     vocabulary = {"<s>", "</s>", "the", " the", " cat", " hat", " a", "The", " The", " Cat"};
    static inline std::map<std::pair<std::string, std::string>, Search::Score> bigramCosts;  // (previous token, token) -> cost
    static inline bool                                                         capitalizedVariants = false;
    static inline size_t                                                       numResets           = 0ul;
    static inline std::vector<std::string>                                     tokenizedTexts;
    static inline std::vector<Search::LlmScoringRequest>                       requests;
    static inline std::vector<std::vector<Search::LlmHistory>>                 cleanups;

    ToyLlm(Core::Configuration const& config)
            : Core::Component(config), Search::LlmScorer(config) {}

    static void clear() {
        bigramCosts.clear();
        capitalizedVariants = false;
        numResets           = 0ul;
        tokenizedTexts.clear();
        requests.clear();
        cleanups.clear();
    }

    static Search::LlmToken token(std::string const& text) {
        auto it = std::find(vocabulary.begin(), vocabulary.end(), text);
        if (it != vocabulary.end()) {
            return it - vocabulary.begin();
        }
        require(text.size() == 1ul);
        return 1000 + static_cast<unsigned char>(text[0]);
    }

    static std::string text(Search::LlmToken token) {
        return token >= 1000 ? std::string(1, static_cast<char>(token - 1000)) : vocabulary.at(token);
    }

    void reset() override {
        ++numResets;
    }

    Search::LlmTokenSequence initialTokens() override {
        return {token("<s>")};
    }

    Search::LlmTokenSequence sentenceEndTokens() override {
        return {token("</s>")};
    }

    std::vector<std::vector<std::string>> spellingVariants(std::vector<std::string> const& words) override {
        if (not capitalizedVariants) {
            return LlmScorer::spellingVariants(words);
        }
        std::vector<std::vector<std::string>> result;
        for (auto const& word : words) {
            std::string capitalized = word;
            capitalized[0]          = std::toupper(capitalized[0]);
            result.push_back({word, capitalized});
        }
        return result;
    }

    std::vector<Search::LlmTokenSequence> tokenize(std::vector<std::string> const& texts) override {
        std::vector<Search::LlmTokenSequence> result;
        for (auto const& t : texts) {
            tokenizedTexts.push_back(t);
            result.emplace_back();
            if (std::find(vocabulary.begin(), vocabulary.end(), t) != vocabulary.end()) {
                result.back().push_back(token(t));
                continue;
            }
            for (char c : t) {
                result.back().push_back(token(std::string(1, c)));
            }
        }
        return result;
    }

    std::vector<std::vector<Search::Score>> score(std::vector<Search::LlmScoringRequest> const& batch) override {
        std::vector<std::vector<Search::Score>> result;
        for (auto const& request : batch) {
            requests.push_back(request);
            result.emplace_back();
            Search::LlmToken previous = request.prefix.back();
            for (Search::LlmToken t : request.tokens) {
                auto it = bigramCosts.find({text(previous), text(t)});
                result.back().push_back(it != bigramCosts.end() ? it->second : defaultCost);
                previous = t;
            }
        }
        return result;
    }

    void cleanup(std::vector<Search::LlmHistory> const& activeHistories) override {
        cleanups.push_back(activeHistories);
    }
};

inline void registerToyLlm() {
    static bool registered = false;
    if (not registered) {
        Search::Module::instance().llmScorerFactory().registerLlmScorer(
                "toy", [](Core::Configuration const& config) { return Core::Ref<Search::LlmScorer>(new ToyLlm(config)); });
        registered = true;
    }
}

inline Core::Ref<Test::Lexicon> pieceLexicon(std::vector<std::string> const& labels) {
    auto lexicon = Core::ref(new Test::Lexicon());
    for (size_t i = 0ul; i < labels.size(); ++i) {
        lexicon->addPhoneme("p" + std::to_string(i), false);
    }
    for (size_t i = 0ul; i < labels.size(); ++i) {
        lexicon->addLemma(labels[i], "p" + std::to_string(i), labels[i] == "<blank>" ? "blank" : "");
    }
    return lexicon;
}

}  // namespace Test

#endif  // TEST_LLM_HELPERS_HH
