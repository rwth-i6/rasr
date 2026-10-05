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

/*
 * Tests for `llm-timesync-beam-search` and its LLM score cache.
 *
 * A deterministic toy "LLM" stands in for a real one: known words (including their leading separator) are single
 * tokens, every other text is spelled by one token per character, and every token costs a configurable bigram cost
 * given its predecessor. Optionally, a capitalized variant is offered for every text as well. The acoustic scores are handed to the search directly through a `StepwiseNoOpLabelScorer`.
 */

#include <Test/Lexicon.hh>
#include <Test/UnitTest.hh>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Nn/LabelScorer/NoOpLabelScorer.hh>
#include <Search/LlmTimesyncBeamSearch/LlmScoreCache.hh>
#include <Search/LlmTimesyncBeamSearch/LlmTimesyncBeamSearch.hh>
#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>

namespace {

// SentencePiece word-start marker U+2581, for literal concatenation
#define WORD_START "\xE2\x96\x81"

// Score given to every label a frame is not supposed to emit
constexpr f32 suppressed = 30.0f;

// Cost of a token whose bigram is not listed
constexpr Search::Score defaultCost = 5.0;

class ToyLlm : public Search::LlmScorer {
public:
    static std::vector<std::string>                                                   vocabulary;
    static std::map<std::pair<std::string, std::string>, Search::Score>               bigramCosts;  // (previous token, token) -> cost
    static size_t                                                                     numResets;
    static bool                                                                       capitalizedVariants;
    static std::vector<std::string>                                                   tokenizedTexts;
    static std::vector<std::pair<Search::LlmTokenSequence, Search::LlmTokenSequence>> scoredRequests;

    ToyLlm(Core::Configuration const& config)
            : Core::Component(config), Search::LlmScorer(config) {}

    static void clear() {
        bigramCosts.clear();
        numResets           = 0ul;
        capitalizedVariants = false;
        tokenizedTexts.clear();
        scoredRequests.clear();
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
        if (token >= 1000) {
            return std::string(1, static_cast<char>(token - 1000));
        }
        return vocabulary.at(token);
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

    static Search::LlmTokenSequence tokenizeOne(std::string const& t) {
        if (std::find(vocabulary.begin(), vocabulary.end(), t) != vocabulary.end()) {
            return {token(t)};
        }
        Search::LlmTokenSequence result;
        for (char c : t) {
            result.push_back(token(std::string(1, c)));
        }
        return result;
    }

    std::vector<Search::LlmTokenSequenceVariants> tokenize(std::vector<std::string> const& texts) override {
        std::vector<Search::LlmTokenSequenceVariants> result;
        for (auto const& t : texts) {
            tokenizedTexts.push_back(t);
            result.push_back({tokenizeOne(t)});
            if (capitalizedVariants) {
                std::string capitalized = t;
                size_t      first       = capitalized.find_first_not_of(' ');
                capitalized[first]      = std::toupper(capitalized[first]);
                result.back().push_back(tokenizeOne(capitalized));
            }
        }
        return result;
    }

    std::vector<std::vector<Search::Score>> scoreContinuations(std::vector<Search::LlmTokenSequence> const& prefixes,
                                                               std::vector<Search::LlmTokenSequence> const& continuations) override {
        std::vector<std::vector<Search::Score>> result;
        for (size_t i = 0ul; i < prefixes.size(); ++i) {
            scoredRequests.push_back({prefixes[i], continuations[i]});
            result.emplace_back();
            Search::LlmToken previous = prefixes[i].back();
            for (Search::LlmToken t : continuations[i]) {
                auto it = bigramCosts.find({text(previous), text(t)});
                result.back().push_back(it != bigramCosts.end() ? it->second : defaultCost);
                previous = t;
            }
        }
        return result;
    }
};

std::vector<std::string>                                                   ToyLlm::vocabulary = {"<s>", "</s>", "the", " the", " cat", " hat", " a", "The", " The", " Cat"};
std::map<std::pair<std::string, std::string>, Search::Score>               ToyLlm::bigramCosts;
size_t                                                                     ToyLlm::numResets           = 0ul;
bool                                                                       ToyLlm::capitalizedVariants = false;
std::vector<std::string>                                                   ToyLlm::tokenizedTexts;
std::vector<std::pair<Search::LlmTokenSequence, Search::LlmTokenSequence>> ToyLlm::scoredRequests;

void registerToyLlm() {
    static bool registered = false;
    if (not registered) {
        Search::Module::instance().llmScorerFactory().registerLlmScorer(
                "toy",
                [](Core::Configuration const& config) { return Core::Ref<Search::LlmScorer>(new ToyLlm(config)); });
        registered = true;
    }
}

}  // namespace

/*
 * ===================
 * === Score cache ===
 * ===================
 */

class LlmScoreCacheTest : public Test::ConfigurableFixture {
public:
    void setUp() {
        ToyLlm::clear();
        cache_.setScorer(Core::ref(new ToyLlm(config)));
        cache_.reset();
    }

protected:
    Search::LlmScoreCache::Result score(Search::LlmHistory history, Search::LlmTokenSequence const& tokens) {
        std::vector<Search::LlmScoreCache::Result> results;
        cache_.score({{history, tokens}}, results);
        return results.front();
    }

    Search::LlmScoreCache cache_;
};

TEST_F(Search, LlmScoreCacheTest, InitialHistoryHoldsTheInitialTokens) {
    Search::LlmTokenSequence tokens;
    cache_.historyTokens(cache_.initialHistory(), tokens);
    EXPECT_EQ(tokens.size(), 1ul);
    EXPECT_EQ(tokens.front(), ToyLlm::token("<s>"));
    EXPECT_EQ(cache_.historyLength(cache_.initialHistory()), 0u);
    EXPECT_EQ(ToyLlm::numResets, 1ul);
}

TEST_F(Search, LlmScoreCacheTest, ScoresAreCachedByHistoryAndTokens) {
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"the", " cat"}, 2.0}};
    auto const root     = cache_.initialHistory();
    auto const the      = ToyLlm::token("the");
    auto const cat      = ToyLlm::token(" cat");

    // Two identical requests in one batch are sent to the LLM once
    std::vector<Search::LlmScoreCache::Result> results;
    cache_.score({{root, {the, cat}}, {root, {the, cat}}}, results);
    EXPECT_EQ(ToyLlm::scoredRequests.size(), 1ul);
    EXPECT_DOUBLE_EQ(results[0].cost, 3.0, 1e-6);
    EXPECT_EQ(results[0].history, results[1].history);
    EXPECT_EQ(cache_.historyLength(results[0].history), 2u);

    // A repeated request and a prefix of a scored request are answered from the cache
    EXPECT_DOUBLE_EQ(score(root, {the, cat}).cost, 3.0, 1e-6);
    auto prefix = score(root, {the});
    EXPECT_DOUBLE_EQ(prefix.cost, 1.0, 1e-6);
    EXPECT_EQ(ToyLlm::scoredRequests.size(), 1ul);

    // Continuing from a history obtained by scoring gives the same history as scoring the whole sequence at once
    auto continued = score(prefix.history, {cat});
    EXPECT_EQ(continued.history, results[0].history);
    EXPECT_DOUBLE_EQ(continued.cost, 2.0, 1e-6);
    EXPECT_EQ(ToyLlm::scoredRequests.size(), 1ul);

    // An empty request costs nothing and keeps the history
    auto empty = score(prefix.history, {});
    EXPECT_EQ(empty.history, prefix.history);
    EXPECT_DOUBLE_EQ(empty.cost, 0.0, 1e-6);
}

TEST_F(Search, LlmScoreCacheTest, LlmSeesTheFullHistoryAsPrefix) {
    auto const root = cache_.initialHistory();
    auto       the  = score(root, {ToyLlm::token("the")});
    score(the.history, {ToyLlm::token(" cat")});

    EXPECT_EQ(ToyLlm::scoredRequests.size(), 2ul);
    auto const& prefix = ToyLlm::scoredRequests.back().first;
    EXPECT_EQ(prefix.size(), 2ul);
    EXPECT_EQ(prefix[0], ToyLlm::token("<s>"));
    EXPECT_EQ(prefix[1], ToyLlm::token("the"));
}

TEST_F(Search, LlmScoreCacheTest, TokenizationIsCachedByText) {
    std::vector<Search::LlmTokenSequenceVariants const*> tokenizations;
    cache_.tokenize({"the", " cat", "the"}, tokenizations);
    EXPECT_EQ(ToyLlm::tokenizedTexts.size(), 2ul);
    EXPECT_EQ(tokenizations[0], tokenizations[2]);

    cache_.tokenize({" cat"}, tokenizations);
    EXPECT_EQ(ToyLlm::tokenizedTexts.size(), 2ul);

    // Tokenizations survive the segment reset
    cache_.reset();
    cache_.tokenize({"the"}, tokenizations);
    EXPECT_EQ(ToyLlm::tokenizedTexts.size(), 2ul);
}

TEST_F(Search, LlmScoreCacheTest, ScoresAreForgottenAtSegmentStart) {
    auto const root = cache_.initialHistory();
    score(root, {ToyLlm::token("the")});
    cache_.reset();
    score(cache_.initialHistory(), {ToyLlm::token("the")});
    EXPECT_EQ(ToyLlm::scoredRequests.size(), 2ul);
}

/*
 * ==============
 * === Search ===
 * ==============
 */

class LlmSearchTest : public Test::ConfigurableFixture {
public:
    void setUp();
    void tearDown();

protected:
    struct Result {
        std::string pieces;  // Non-blank labels along the traceback, joined by blanks
        double      amScore;
        double      lmScore;
    };

    void   buildSearch();
    Result decode(std::vector<std::map<std::string, f32>> const& frameScores);

    std::vector<std::string>                       labels_;
    Core::Ref<Test::Lexicon>                       lexicon_;
    Core::Ref<Speech::ModelCombination>            modelCombination_;
    std::unique_ptr<Search::LlmTimesyncBeamSearch> search_;
};

void LlmSearchTest::setUp() {
    registerToyLlm();
    ToyLlm::clear();

    labels_  = {"<blank>", WORD_START "the", WORD_START "cat", WORD_START "hat", WORD_START "ca", "t", WORD_START "a"};
    lexicon_ = Core::ref(new Test::Lexicon());
    for (size_t i = 0ul; i < labels_.size(); ++i) {
        lexicon_->addPhoneme("p" + std::to_string(i), false);
    }
    for (size_t i = 0ul; i < labels_.size(); ++i) {
        lexicon_->addLemma(labels_[i], "p" + std::to_string(i), i == 0ul ? "blank" : "");
    }

    setParameter("*.max-beam-size", "10");
    setParameter("*.collapse-repeated-labels", "true");
    setParameter("*.llm.type", "toy");
}

void LlmSearchTest::tearDown() {
    search_.reset();
    modelCombination_.reset();
    lexicon_.reset();
}

void LlmSearchTest::buildSearch() {
    Bliss::LexiconRef const lexiconRef(lexicon_.get());
    modelCombination_ = Core::ref(new Speech::ModelCombination(config, lexiconRef, Core::Ref<Am::AcousticModel>(), Core::Ref<Lm::ScaledLanguageModel>()));
    modelCombination_->setLabelScorer(Core::ref(new Nn::StepwiseNoOpLabelScorer(select("label-scorer"))), 0ul);

    search_ = std::make_unique<Search::LlmTimesyncBeamSearch>(config);
    search_->setModelCombination(*modelCombination_);
}

LlmSearchTest::Result LlmSearchTest::decode(std::vector<std::map<std::string, f32>> const& frameScores) {
    search_->enterSegment();
    for (auto const& scores : frameScores) {
        std::shared_ptr<f32[]> frame(new f32[labels_.size()]);
        std::fill(frame.get(), frame.get() + labels_.size(), suppressed);
        for (auto const& [label, score] : scores) {
            auto it = std::find(labels_.begin(), labels_.end(), label);
            require(it != labels_.end());
            frame[it - labels_.begin()] = score;
        }
        search_->putFeature(Nn::DataView(std::shared_ptr<f32 const[]>(frame), labels_.size()));
    }
    search_->finishSegment();

    Result result{std::string(), 0.0, 0.0};
    auto   traceback = search_->getCurrentBestTraceback();
    for (auto const& item : *traceback) {
        result.amScore = item.score.acoustic;
        result.lmScore = item.score.lm;
        if (item.pronunciation == nullptr or item.pronunciation->lemma() == lexicon_->specialLemma("blank")) {
            continue;
        }
        if (not result.pieces.empty()) {
            result.pieces += " ";
        }
        result.pieces += item.pronunciation->lemma()->preferredOrthographicForm().str();
    }
    return result;
}

TEST_F(Search, LlmSearchTest, LlmDecidesBetweenAcousticallyTiedWords) {
    buildSearch();
    std::vector<std::map<std::string, f32>> frames = {
            {{WORD_START "the", 0.0f}},
            {{"<blank>", 0.0f}},
            {{WORD_START "cat", 1.0f}, {WORD_START "hat", 1.0f}},
            {{"<blank>", 0.0f}}};

    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"the", " cat"}, 0.5}, {{"the", " hat"}, 3.0}, {{" cat", "</s>"}, 0.2}, {{" hat", "</s>"}, 0.2}};
    auto cat            = decode(frames);
    EXPECT_EQ(cat.pieces, std::string(WORD_START "the " WORD_START "cat"));
    EXPECT_DOUBLE_EQ(cat.lmScore, 1.0 + 0.5 + 0.2, 1e-4);
    EXPECT_DOUBLE_EQ(cat.amScore, 1.0, 1e-4);

    ToyLlm::bigramCosts[{"the", " cat"}] = 4.0;
    auto hat                             = decode(frames);
    EXPECT_EQ(hat.pieces, std::string(WORD_START "the " WORD_START "hat"));
    EXPECT_DOUBLE_EQ(hat.lmScore, 1.0 + 3.0 + 0.2, 1e-4);
}

TEST_F(Search, LlmSearchTest, WordOfSeveralPiecesIsOneWord) {
    buildSearch();
    // "▁ca t" spells the word "cat", which the toy LLM knows as the single token " cat"
    auto result = decode({{{WORD_START "the", 0.0f}}, {{WORD_START "ca", 0.0f}}, {{"t", 0.0f}}});
    EXPECT_EQ(result.pieces, std::string(WORD_START "the " WORD_START "ca t"));
    EXPECT_DOUBLE_EQ(result.lmScore, 3 * defaultCost, 1e-4);
    EXPECT_TRUE(std::find(ToyLlm::tokenizedTexts.begin(), ToyLlm::tokenizedTexts.end(), " cat") != ToyLlm::tokenizedTexts.end());
}

TEST_F(Search, LlmSearchTest, OnlyTheFirstWordGoesWithoutSeparator) {
    buildSearch();
    // The blank keeps the repeated label from being collapsed into one
    decode({{{WORD_START "the", 0.0f}}, {{"<blank>", 0.0f}}, {{WORD_START "the", 0.0f}}});
    auto const& texts = ToyLlm::tokenizedTexts;
    EXPECT_TRUE(std::find(texts.begin(), texts.end(), "the") != texts.end());
    EXPECT_TRUE(std::find(texts.begin(), texts.end(), " the") != texts.end());
}

TEST_F(Search, LlmSearchTest, LlmIsNeverAskedTwiceInASegment) {
    buildSearch();
    decode({{{WORD_START "the", 0.0f}}, {{"<blank>", 0.0f}}, {{WORD_START "cat", 1.0f}, {WORD_START "hat", 1.0f}}, {{"<blank>", 0.0f}}});

    std::set<std::pair<Search::LlmTokenSequence, Search::LlmTokenSequence>> unique(ToyLlm::scoredRequests.begin(), ToyLlm::scoredRequests.end());
    EXPECT_EQ(unique.size(), ToyLlm::scoredRequests.size());

    std::set<std::string> uniqueTexts(ToyLlm::tokenizedTexts.begin(), ToyLlm::tokenizedTexts.end());
    EXPECT_EQ(uniqueTexts.size(), ToyLlm::tokenizedTexts.size());
}

TEST_F(Search, LlmSearchTest, ScaleAndWordPenalty) {
    setParameter("*.llm-scale", "2.0");
    setParameter("*.word-penalty", "-1.0");
    buildSearch();
    // Two words plus sentence end, all at the default cost
    auto result = decode({{{WORD_START "the", 0.0f}}, {{WORD_START "a", 0.0f}}});
    EXPECT_DOUBLE_EQ(result.lmScore, 2.0 * 3 * 5.0 - 2.0, 1e-4);
}

TEST_F(Search, LlmSearchTest, CheapestVariantGivesScoreAndHistory) {
    ToyLlm::capitalizedVariants = true;
    buildSearch();
    // "The" is cheaper than "the" at the start, so the second word must be scored after "The"
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 2.0}, {{"<s>", "The"}, 0.5}, {{"The", " cat"}, 0.25}, {{"The", " Cat"}, 3.0}, {{" cat", "</s>"}, 0.125}};
    auto result         = decode({{{WORD_START "the", 0.0f}}, {{WORD_START "cat", 0.0f}}});
    EXPECT_DOUBLE_EQ(result.lmScore, 0.5 + 0.25 + 0.125, 1e-4);

    bool sawPrefixWithCapitalizedThe = false;
    for (auto const& [prefix, continuation] : ToyLlm::scoredRequests) {
        sawPrefixWithCapitalizedThe = sawPrefixWithCapitalizedThe or (prefix.size() == 2ul and prefix[1] == ToyLlm::token("The"));
        EXPECT_FALSE(prefix.size() == 2ul and prefix[1] == ToyLlm::token("the"));
    }
    EXPECT_TRUE(sawPrefixWithCapitalizedThe);
}

TEST_F(Search, LlmSearchTest, LastWordVariantIsChosenTogetherWithSentenceEnd) {
    ToyLlm::capitalizedVariants = true;
    buildSearch();
    // Alone, "the" is cheaper than "The", but including the sentence end "The" is
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"<s>", "The"}, 2.0}, {{"the", "</s>"}, 5.0}, {{"The", "</s>"}, 0.5}};
    auto result         = decode({{{WORD_START "the", 0.0f}}});
    EXPECT_DOUBLE_EQ(result.lmScore, 2.0 + 0.5, 1e-4);
}
