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
 * Tests for `llm-rnnt-timesync-beam-search` with the toy LLM of `LlmHelpers.hh` and label scores that are given
 * per frame and only advance to the next frame on blank.
 */

#include <Test/LlmHelpers.hh>
#include <Test/UnitTest.hh>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Nn/LabelScorer/NoOpLabelScorer.hh>
#include <Search/LlmRnntTimesyncBeamSearch/LlmRnntTimesyncBeamSearch.hh>
#include <Speech/ModelCombination.hh>

using Test::defaultCost;
using Test::ToyLlm;

namespace {

// Scores given per frame, advancing to the next frame only on blank as an RNN-T does
class RnntNoOpLabelScorer : public Nn::StepwiseNoOpLabelScorer {
public:
    RnntNoOpLabelScorer(Core::Configuration const& config)
            : Core::Component(config), Nn::StepwiseNoOpLabelScorer(config) {}

    Nn::ScoringContextRef extendedScoringContext(Nn::ScoringContextRef scoringContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType) override {
        switch (transitionType) {
            case Nn::TransitionType::INITIAL_BLANK:
            case Nn::TransitionType::LABEL_TO_BLANK:
            case Nn::TransitionType::BLANK_LOOP:
                return StepwiseNoOpLabelScorer::extendedScoringContext(scoringContext, nextToken, transitionType);
            default:
                return scoringContext;
        }
    }
};

}  // namespace

class LlmRnntSearchTest : public Test::ConfigurableFixture {
public:
    void setUp();
    void tearDown();

protected:
    struct Result {
        std::string pieces;  // Labels along the traceback, joined by blanks
        double      amScore;
        double      lmScore;
    };

    void   buildSearch();
    Result decode(std::vector<std::map<std::string, f32>> const& frameScores);

    std::vector<std::string>                           labels_;
    Core::Ref<Test::Lexicon>                           lexicon_;
    Core::Ref<Speech::ModelCombination>                modelCombination_;
    std::unique_ptr<Search::LlmRnntTimesyncBeamSearch> search_;
};

void LlmRnntSearchTest::setUp() {
    Test::registerToyLlm();
    ToyLlm::clear();
    labels_ = {"<blank>", WORD_START "the", WORD_START "cat", WORD_START "hat", WORD_START "a"};
    setParameter("*.max-beam-size", "10");
    setParameter("*.llm.type", "toy");
}

void LlmRnntSearchTest::tearDown() {
    search_.reset();
    modelCombination_.reset();
    lexicon_.reset();
}

void LlmRnntSearchTest::buildSearch() {
    lexicon_ = Test::pieceLexicon(labels_);
    Bliss::LexiconRef const lexiconRef(lexicon_.get());
    modelCombination_ = Core::ref(new Speech::ModelCombination(config, lexiconRef, Core::Ref<Am::AcousticModel>(), Core::Ref<Lm::ScaledLanguageModel>()));
    modelCombination_->setLabelScorer(Core::ref(new RnntNoOpLabelScorer(select("label-scorer"))), 0ul);

    search_ = std::make_unique<Search::LlmRnntTimesyncBeamSearch>(config);
    search_->setModelCombination(*modelCombination_);
}

LlmRnntSearchTest::Result LlmRnntSearchTest::decode(std::vector<std::map<std::string, f32>> const& frameScores) {
    search_->enterSegment();
    for (auto const& scores : frameScores) {
        std::shared_ptr<f32[]> frame(new f32[labels_.size()]);
        std::fill(frame.get(), frame.get() + labels_.size(), Test::suppressed);
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

TEST_F(Search, LlmRnntSearchTest, LlmDecidesBetweenAcousticallyTiedWords) {
    buildSearch();
    std::vector<std::map<std::string, f32>> frames = {
            {{WORD_START "the", 0.0f}, {"<blank>", 0.5f}},
            {{WORD_START "cat", 1.0f}, {WORD_START "hat", 1.0f}, {"<blank>", 0.5f}}};

    ToyLlm::bigramCosts = {{{"<s>", "</s>"}, 10.0}, {{"<s>", "the"}, 1.0}, {{"the", " cat"}, 0.5}, {{"the", " hat"}, 3.0}, {{" cat", "</s>"}, 0.2}, {{" hat", "</s>"}, 0.2}};
    auto cat            = decode(frames);
    EXPECT_EQ(cat.pieces, std::string(WORD_START "the " WORD_START "cat"));
    EXPECT_DOUBLE_EQ(cat.lmScore, 1.0 + 0.5 + 0.2, 1e-4);
    EXPECT_DOUBLE_EQ(cat.amScore, 0.5 + 1.0 + 0.5, 1e-4);

    ToyLlm::bigramCosts[{"the", " cat"}] = 4.0;
    auto hat                             = decode(frames);
    EXPECT_EQ(hat.pieces, std::string(WORD_START "the " WORD_START "hat"));
    EXPECT_DOUBLE_EQ(hat.lmScore, 1.0 + 3.0 + 0.2, 1e-4);
}

TEST_F(Search, LlmRnntSearchTest, SeveralWordsInOneFrame) {
    buildSearch();
    ToyLlm::bigramCosts = {{{"<s>", "</s>"}, 10.0}, {{"<s>", "the"}, 1.0}, {{"the", " cat"}, 0.5}, {{" cat", "</s>"}, 0.2}};
    // Both orders cost the same acoustically; the LLM prefers "the cat"
    auto result = decode({{{WORD_START "the", 0.0f}, {WORD_START "cat", 0.0f}, {"<blank>", 0.0f}}});
    EXPECT_EQ(result.pieces, std::string(WORD_START "the " WORD_START "cat"));
    EXPECT_DOUBLE_EQ(result.lmScore, 1.0 + 0.5 + 0.2, 1e-4);
}

TEST_F(Search, LlmRnntSearchTest, MaxLabelsPerFrameLimitsTheWordsOfAFrame) {
    setParameter("*.max-labels-per-timeframe", "1");
    buildSearch();
    ToyLlm::bigramCosts = {{{"<s>", "</s>"}, 10.0}, {{"<s>", "the"}, 1.0}, {{"the", " cat"}, 0.5}, {{" cat", "</s>"}, 0.2}};
    auto result         = decode({{{WORD_START "the", 0.0f}, {WORD_START "cat", 0.0f}, {"<blank>", 0.0f}}});
    EXPECT_EQ(result.pieces, std::string(WORD_START "the"));
    EXPECT_DOUBLE_EQ(result.lmScore, 1.0 + defaultCost, 1e-4);
}

TEST_F(Search, LlmRnntSearchTest, CleanupKeepsTheHistoriesOfTheBeam) {
    setParameter("*.cache-cleanup-interval", "1");
    buildSearch();
    decode({{{WORD_START "the", 0.0f}, {"<blank>", 0.0f}}, {{WORD_START "cat", 0.0f}, {"<blank>", 0.0f}}, {{"<blank>", 0.0f}}});

    EXPECT_EQ(ToyLlm::cleanups.size(), 3ul);
    // After "▁cat" the word "the" is finished, so the best hypothesis continues a history scored by a request
    auto const& active = ToyLlm::cleanups.back();
    bool        found  = false;
    for (auto const& request : ToyLlm::requests) {
        found = found or std::find(active.begin(), active.end(), request.tokenHistories.back()) != active.end();
    }
    EXPECT_TRUE(found);
}

TEST_F(Search, LlmRnntSearchTest, ScaleAndWordPenalty) {
    setParameter("*.llm-scale", "2.0");
    setParameter("*.word-penalty", "-1.0");
    buildSearch();
    ToyLlm::bigramCosts = {{{"<s>", "</s>"}, 10.0}};
    // One word plus sentence end at the default cost
    auto result = decode({{{WORD_START "the", 0.0f}, {"<blank>", 0.0f}}});
    EXPECT_DOUBLE_EQ(result.lmScore, 2.0 * 2 * defaultCost - 1.0, 1e-4);
}
