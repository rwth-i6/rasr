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
 * Tests for `llm-timesync-beam-search` and its components, with a deterministic toy LLM: known texts are single
 * tokens, other texts are spelled by one token per character, and every token costs a configurable bigram cost.
 */

#include <Test/Lexicon.hh>
#include <Test/LlmHelpers.hh>
#include <Test/UnitTest.hh>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Nn/LabelScorer/NoOpLabelScorer.hh>
#include <Search/Llm/LlmHistoryTrie.hh>
#include <Search/Llm/LlmWordScorer.hh>
#include <Search/Llm/WordAssembler.hh>
#include <Search/LlmTimesyncBeamSearch/LlmTimesyncBeamSearch.hh>
#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>

using Test::defaultCost;
using Test::pieceLexicon;
using Test::registerToyLlm;
using Test::suppressed;
using Test::ToyLlm;

/*
 * =====================
 * === WordAssembler ===
 * =====================
 */

class WordAssemblerTest : public Test::ConfigurableFixture {
protected:
    std::unique_ptr<Search::WordAssembler> build(std::vector<std::string> const& labels) {
        auto assembler = std::make_unique<Search::WordAssembler>(config);
        assembler->setLexicon(Bliss::LexiconRef(pieceLexicon(labels).get()));
        return assembler;
    }
};

TEST_F(Search, WordAssemblerTest, WordStartMarked) {
    auto assembler = build({WORD_START "the", WORD_START "ca", "t"});

    auto first = assembler->extend(Search::WordAssembler::noWord, 0);
    EXPECT_EQ(first.finished, Search::WordAssembler::noWord);
    EXPECT_EQ(assembler->spelling(first.pending), std::string("the"));

    auto second = assembler->extend(first.pending, 1);
    EXPECT_EQ(assembler->spelling(second.finished), std::string("the"));
    auto third = assembler->extend(second.pending, 2);
    EXPECT_EQ(third.finished, Search::WordAssembler::noWord);
    EXPECT_EQ(assembler->spelling(third.pending), std::string("cat"));
}

TEST_F(Search, WordAssemblerTest, ContinuationMarked) {
    setParameter("*.word-piece-convention", "continuation-marked");
    auto assembler = build({"the", "ca@@", "t"});

    auto first = assembler->extend(Search::WordAssembler::noWord, 0);
    EXPECT_EQ(assembler->spelling(first.finished), std::string("the"));
    EXPECT_EQ(first.pending, Search::WordAssembler::noWord);

    auto second = assembler->extend(first.pending, 1);
    EXPECT_EQ(second.finished, Search::WordAssembler::noWord);
    auto third = assembler->extend(second.pending, 2);
    EXPECT_EQ(assembler->spelling(third.finished), std::string("cat"));
}

TEST_F(Search, WordAssemblerTest, EqualSpellingsHaveEqualIds) {
    auto assembler = build({WORD_START "cat", WORD_START "ca", "t"});
    auto whole     = assembler->extend(Search::WordAssembler::noWord, 0).pending;
    auto pieces    = assembler->extend(assembler->extend(Search::WordAssembler::noWord, 1).pending, 2).pending;
    EXPECT_EQ(whole, pieces);
}

/*
 * ======================
 * === LlmHistoryTrie ===
 * ======================
 */

TEST(Search, LlmHistoryTrie, HistoriesAreSharedPrefixes) {
    Search::LlmHistoryTrie trie;
    trie.reset({7});
    auto root = trie.initialHistory();
    EXPECT_EQ(trie.length(root), 0u);

    auto a  = trie.extend(root, 1);
    auto ab = trie.extend(a, 2);
    EXPECT_EQ(trie.extend(root, 1), a);
    EXPECT_EQ(trie.length(ab), 2u);
    EXPECT_FALSE(trie.hasCost(ab));
    trie.setCost(ab, 1.5);
    EXPECT_TRUE(trie.hasCost(ab));

    Search::LlmTokenSequence tokens;
    trie.tokens(ab, tokens);
    EXPECT_EQ(tokens.size(), 3ul);
    EXPECT_EQ(tokens[0], 7);
    EXPECT_EQ(tokens[2], 2);
}

/*
 * =====================
 * === LlmWordScorer ===
 * =====================
 */

class LlmWordScorerTest : public Test::ConfigurableFixture {
public:
    void setUp() {
        registerToyLlm();
        ToyLlm::clear();
        setParameter("*.type", "toy");
    }

protected:
    std::unique_ptr<Search::LlmWordScorer> scorer_;
    std::string                            the_ = "the";
    std::string                            cat_ = "cat";

    void build() {
        scorer_ = std::make_unique<Search::LlmWordScorer>(config);
        scorer_->reset();
    }

    Search::LlmWordScorer::Result score(Search::LlmHistory history, std::string const* word, bool sentenceEnd = false) {
        std::vector<Search::LlmWordScorer::Result> results;
        scorer_->score({{history, word, sentenceEnd}}, results);
        return results.front();
    }
};

TEST_F(Search, LlmWordScorerTest, WordsAreScoredOnceAndAfterTheSeparator) {
    build();
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"the", " cat"}, 2.0}};

    auto the = score(scorer_->initialHistory(), &the_);
    EXPECT_DOUBLE_EQ(the.cost, 1.0, 1e-6);
    auto cat = score(the.history, &cat_);
    EXPECT_DOUBLE_EQ(cat.cost, 2.0, 1e-6);
    EXPECT_EQ(ToyLlm::requests.size(), 2ul);

    // The request carries the history handle, its tokens and the handle after each token
    auto const& request = ToyLlm::requests.back();
    EXPECT_EQ(request.history, the.history);
    EXPECT_EQ(request.prefix.size(), 2ul);
    EXPECT_EQ(request.tokens.front(), ToyLlm::token(" cat"));
    EXPECT_EQ(request.tokenHistories.back(), cat.history);

    // Repeated words are answered from the cache
    EXPECT_EQ(score(the.history, &cat_).history, cat.history);
    EXPECT_EQ(ToyLlm::requests.size(), 2ul);
    EXPECT_TRUE(ToyLlm::tokenizedTexts == std::vector<std::string>({"the", " cat"}));
}

TEST_F(Search, LlmWordScorerTest, EqualRequestsInABatchAreSentOnce) {
    build();
    std::vector<Search::LlmWordScorer::Result> results;
    scorer_->score({{scorer_->initialHistory(), &the_, false}, {scorer_->initialHistory(), &the_, false}}, results);
    EXPECT_EQ(ToyLlm::requests.size(), 1ul);
    EXPECT_EQ(results[0].history, results[1].history);
}

TEST_F(Search, LlmWordScorerTest, CheapestVariantGivesCostAndHistory) {
    ToyLlm::capitalizedVariants = true;
    build();
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 2.0}, {{"<s>", "The"}, 0.5}};

    auto result = score(scorer_->initialHistory(), &the_);
    EXPECT_DOUBLE_EQ(result.cost, 0.5, 1e-6);
    EXPECT_EQ(ToyLlm::requests.size(), 2ul);
    EXPECT_EQ(ToyLlm::requests[1].tokenHistories.back(), result.history);
}

TEST_F(Search, LlmWordScorerTest, LastVariantIsChosenTogetherWithTheSentenceEnd) {
    ToyLlm::capitalizedVariants = true;
    build();
    // Alone "the" is cheaper, including the sentence end "The" is
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"<s>", "The"}, 2.0}, {{"the", "</s>"}, 5.0}, {{"The", "</s>"}, 0.5}};
    EXPECT_DOUBLE_EQ(score(scorer_->initialHistory(), &the_, true).cost, 2.5, 1e-6);
    EXPECT_DOUBLE_EQ(score(scorer_->initialHistory(), nullptr, true).cost, defaultCost, 1e-6);
}

TEST_F(Search, LlmWordScorerTest, ScoresAreForgottenAtSegmentStart) {
    build();
    score(scorer_->initialHistory(), &the_);
    scorer_->reset();
    score(scorer_->initialHistory(), &the_);
    EXPECT_EQ(ToyLlm::requests.size(), 2ul);
    EXPECT_EQ(ToyLlm::tokenizedTexts.size(), 1ul);
    EXPECT_EQ(ToyLlm::numResets, 2ul);
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
    labels_ = {"<blank>", WORD_START "the", WORD_START "cat", WORD_START "hat", WORD_START "ca", "t", WORD_START "a"};
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
    lexicon_ = pieceLexicon(labels_);
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
}

TEST_F(Search, LlmSearchTest, ContinuationMarkedPieces) {
    labels_ = {"<blank>", "the", "ca@@", "t"};
    setParameter("*.word-piece-convention", "continuation-marked");
    buildSearch();
    ToyLlm::bigramCosts = {{{"<s>", "the"}, 1.0}, {{"the", " cat"}, 0.5}};
    auto result         = decode({{{"the", 0.0f}}, {{"ca@@", 0.0f}}, {{"t", 0.0f}}});
    EXPECT_EQ(result.pieces, std::string("the ca@@ t"));
    EXPECT_DOUBLE_EQ(result.lmScore, 1.0 + 0.5 + defaultCost, 1e-4);
}

TEST_F(Search, LlmSearchTest, LlmIsNeverAskedTwiceInASegment) {
    buildSearch();
    decode({{{WORD_START "the", 0.0f}}, {{"<blank>", 0.0f}}, {{WORD_START "cat", 1.0f}, {WORD_START "hat", 1.0f}}, {{"<blank>", 0.0f}}});

    std::set<Search::LlmHistory> scored;
    for (auto const& request : ToyLlm::requests) {
        EXPECT_TRUE(scored.insert(request.tokenHistories.back()).second);
    }
    std::set<std::string> uniqueTexts(ToyLlm::tokenizedTexts.begin(), ToyLlm::tokenizedTexts.end());
    EXPECT_EQ(uniqueTexts.size(), ToyLlm::tokenizedTexts.size());
}

TEST_F(Search, LlmSearchTest, CleanupKeepsTheHistoriesOfTheBeam) {
    setParameter("*.cache-cleanup-interval", "1");
    buildSearch();
    decode({{{WORD_START "the", 0.0f}}, {{WORD_START "cat", 0.0f}}, {{"<blank>", 0.0f}}});

    EXPECT_EQ(ToyLlm::cleanups.size(), 3ul);
    // After "▁cat" the word "the" is finished, so the best hypothesis continues a history scored by a request
    auto const& active = ToyLlm::cleanups.back();
    bool        found  = false;
    for (auto const& request : ToyLlm::requests) {
        found = found or std::find(active.begin(), active.end(), request.tokenHistories.back()) != active.end();
    }
    EXPECT_TRUE(found);
}

TEST_F(Search, LlmSearchTest, ScaleAndWordPenalty) {
    setParameter("*.llm-scale", "2.0");
    setParameter("*.word-penalty", "-1.0");
    buildSearch();
    // Two words plus sentence end, all at the default cost
    auto result = decode({{{WORD_START "the", 0.0f}}, {{WORD_START "a", 0.0f}}});
    EXPECT_DOUBLE_EQ(result.lmScore, 2.0 * 3 * defaultCost - 2.0, 1e-4);
}
