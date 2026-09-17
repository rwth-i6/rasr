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
 * End-to-end word-boundary and word-LM accounting tests for the open-vocabulary
 * subword fallback of `tree-timesync-beam-search`.
 *
 * Each test drives the real search over a tiny artificial subword inventory and a
 * deterministic toy unigram word LM. The acoustic scores are handed to the search
 * directly through a `StepwiseNoOpLabelScorer`: the intended label of a frame gets
 * a score of 0 and every other label a large one, so the intended label sequence is
 * the only competitive path and the acoustic part of the total score is exactly 0.
 * The whole remaining score is then the word-LM contribution, which makes the number
 * of word events, the tokens they used and the unknown-word bias directly checkable.
 */

#include <Test/File.hh>
#include <Test/Lexicon.hh>
#include <Test/UnitTest.hh>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Am/AcousticModel.hh>
#include <Am/Module.hh>
#include <Lm/Module.hh>
#include <Nn/LabelScorer/NoOpLabelScorer.hh>
#include <Search/TreeTimesyncBeamSearch/TreeTimesyncBeamSearch.hh>
#include <Speech/ModelCombination.hh>

namespace {

// Cost of a log10 language model probability in RASR's internal convention
// (negative natural logarithm). Mirrors the conversion done by `Lm::ArpaLm`.
constexpr double ln10 = 2.30258509299404568402;

constexpr double costUnknown     = 0.60206 * ln10;
constexpr double costCat         = 0.30103 * ln10;
constexpr double costThe         = 1.00000 * ln10;
constexpr double costUnfamiliar  = 0.69897 * ln10;
constexpr double costKnownTwo    = 0.52288 * ln10;
constexpr double costSentenceEnd = 0.09691 * ln10;

// Score given to every label a frame is not supposed to emit. Large enough that no
// deviation from the intended label sequence can be bought back by the unknown-word
// bias the tests configure.
constexpr float suppressed = 1000.0f;

}  // namespace

class FallbackSearchFixture : public Test::ConfigurableFixture {
public:
    void tearDown();

protected:
    struct Result {
        // Lemma names along the traceback, without the final sentence-end item,
        // joined by single blanks so that a failing comparison prints them.
        std::string lemmas;
        // Accumulated acoustic and language model score of the full hypothesis.
        double amScore;
        double lmScore;
        bool   empty;
    };

    Bliss::Lemma* addLemma(std::string const&              orth,
                           std::vector<std::string> const& pronunciations,
                           std::string const&              special,
                           std::vector<std::string> const& syntacticTokens);

    void setCommonParameters();

    // Build acoustic model, search tree, LM and search algorithm from the lexicon
    // and the parameters set so far.
    void buildSearch();

    // Emission index of the single-phoneme "pronunciation" of one subword label.
    Nn::LabelIndex emissionIndexOf(std::string const& phoneme) const;

    // Decode one frame per entry of `labels`, forcing that label in that frame.
    Result decode(std::vector<std::string> const& labels);

    Core::Ref<Test::Lexicon>                        lexicon_;
    Core::Ref<Am::AcousticModel>                    acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>              languageModel_;
    Core::Ref<Speech::ModelCombination>             modelCombination_;
    std::unique_ptr<Search::TreeTimesyncBeamSearch> search_;
    size_t                                          numEmissions_;
};

void FallbackSearchFixture::tearDown() {
    search_.reset();
    modelCombination_.reset();
    languageModel_.reset();
    acousticModel_.reset();
    lexicon_.reset();
}

Bliss::Lemma* FallbackSearchFixture::addLemma(std::string const&              orth,
                                              std::vector<std::string> const& pronunciations,
                                              std::string const&              special,
                                              std::vector<std::string> const& syntacticTokens) {
    Bliss::Lemma* lemma = lexicon_->newLemma();
    if (!special.empty()) {
        lexicon_->defineSpecialLemma(special, lemma);
    }
    for (auto const& pronunciation : pronunciations) {
        Bliss::Pronunciation* pron   = nullptr;
        Core::Status          status = lexicon_->getPronunciation(pronunciation, pron);
        require(status.ok());
        lexicon_->addPronunciation(lemma, pron);
    }
    lexicon_->setOrthographicForms(lemma, {orth});
    lexicon_->setDefaultLemmaName(lemma);
    lexicon_->setSyntacticTokenSequence(lemma, syntacticTokens);
    lexicon_->setDefaultEvaluationToken(lemma);
    return lemma;
}

void FallbackSearchFixture::setCommonParameters() {
    setParameter("*.state-tying.type", "no-tying-dense");
    setParameter("*.hmm.states-per-phone", "1");
    setParameter("*.hmm.state-repetitions", "1");
    setParameter("*.hmm.across-word-model", "no");
    for (auto const* state : {"entry-m1", "entry-m2"}) {
        setParameter(std::string("*.tdp.") + state + ".loop", "infinity");
        setParameter(std::string("*.tdp.") + state + ".skip", "infinity");
    }
    for (auto const* state : {"silence", "state-0", "state-1"}) {
        setParameter(std::string("*.tdp.") + state + ".loop", "0");
        setParameter(std::string("*.tdp.") + state + ".skip", "infinity");
    }

    setParameter("*.lm.type", "ARPA");
    setParameter("*.lm.file", Test::dataFile("unknown_word_fallback/toy.arpa"));
    setParameter("*.lm.image", "");
    setParameter("*.lm.scale", "1.0");

    // A beam wide enough that nothing relevant is lost, so the tests observe the
    // search semantics rather than the pruning.
    setParameter("*.max-beam-size", "500");
    setParameter("*.max-word-end-beam-size", "500");
    setParameter("*.tree-builder-type", "ctc");
    setParameter("*.label-scorer.type", "no-op");
    setParameter("*.label-scorer.transition-preset", "ctc");
    // No cached tree image: every test builds its own topology.
    setParameter("*.search-network.cache-archive", "");
}

void FallbackSearchFixture::buildSearch() {
    acousticModel_ = Am::Module::instance().createAcousticModel(
            select("acoustic-model"), lexicon_, Am::AcousticModel::noEmissions);
    numEmissions_ = acousticModel_->nEmissions();

    Bliss::LexiconRef const lexiconRef(lexicon_.get());
    languageModel_ = Lm::Module::instance().createScaledLanguageModel(select("lm"), lexiconRef);

    modelCombination_ = Core::ref(new Speech::ModelCombination(config, lexiconRef, acousticModel_, languageModel_));
    modelCombination_->setLabelScorer(
            Core::ref(new Nn::StepwiseNoOpLabelScorer(select("label-scorer"))), 0ul);

    search_ = std::make_unique<Search::TreeTimesyncBeamSearch>(config);
    search_->setModelCombination(*modelCombination_);
}

Nn::LabelIndex FallbackSearchFixture::emissionIndexOf(std::string const& phoneme) const {
    // Built exactly like `CtcTreeBuilder::extendPronunciation` does, so the index
    // matches the one the search reads off the state descriptions.
    Bliss::Phoneme::Id const             id = lexicon_->phonemeInventory()->phoneme(phoneme)->id();
    Bliss::ContextPhonology::SemiContext history, future;
    Am::Allophone const*                 allophone = acousticModel_->allophoneAlphabet()->allophone(
            Am::Allophone(Bliss::ContextPhonology::PhonemeInContext(id, history, future),
                                          Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone));
    Am::AllophoneState alloState = acousticModel_->allophoneStateAlphabet()->allophoneState(allophone, 0);
    return static_cast<Nn::LabelIndex>(acousticModel_->emissionIndex(alloState));
}

FallbackSearchFixture::Result FallbackSearchFixture::decode(std::vector<std::string> const& labels) {
    search_->enterSegment();

    std::vector<std::shared_ptr<f32[]>> frames;  // keep the buffers alive until finishSegment
    for (auto const& label : labels) {
        std::shared_ptr<f32[]> frame(new f32[numEmissions_]);
        std::fill(frame.get(), frame.get() + numEmissions_, suppressed);
        require_lt(static_cast<size_t>(emissionIndexOf(label)), numEmissions_);
        frame[emissionIndexOf(label)] = 0.0f;
        frames.push_back(frame);
        search_->putFeature(Nn::DataView(std::shared_ptr<f32 const[]>(frame), numEmissions_));
    }
    search_->finishSegment();

    Result      result{std::string(), 0.0, 0.0, false};
    auto        traceback        = search_->getCurrentBestTraceback();
    auto const* sentenceEndLemma = lexicon_->specialLemma("sentence-end");

    for (auto const& item : *traceback) {
        result.amScore = item.score.acoustic;
        result.lmScore = item.score.lm;
        if (item.pronunciation == nullptr or item.pronunciation->lemma() == nullptr) {
            continue;
        }
        if (item.pronunciation->lemma() == sentenceEndLemma) {
            continue;
        }
        if (not result.lemmas.empty()) {
            result.lemmas += " ";
        }
        result.lemmas += item.pronunciation->lemma()->name().str();
    }
    result.empty = result.lemmas.empty();
    return result;
}

/*
 * ==========================================================================
 * === Word-start-marked (SentencePiece style) boundaries and exclusion   ===
 * ==========================================================================
 */

class WordStartFallbackSearchTest : public FallbackSearchFixture {
public:
    void setUp();
};

void WordStartFallbackSearchTest::setUp() {
    lexicon_ = Core::ref(new Test::Lexicon());
    for (auto const* phoneme : {"_cat", "_un", "_the", "_xy", "familiar", "s", "zz", "blank", "eos"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    // Ordinary in-vocabulary words.
    addLemma("cat", {"_cat"}, "", {"CAT"});
    addLemma("the", {"_the"}, "", {"THE"});
    addLemma("unfamiliar", {"_un familiar"}, "", {"UNFAMILIAR"});

    // Fallback inventory. "_xy" and "zz" spell no in-vocabulary word at all.
    addLemma("[UNKNOWN]", {}, "unknown", {"<UNK>"});
    addLemma("_cat", {"_cat"}, "unknown-word-start", {});
    addLemma("_un", {"_un"}, "unknown-word-start", {});
    addLemma("_the", {"_the"}, "unknown-word-start", {});
    addLemma("_xy", {"_xy"}, "unknown-word-start", {});
    addLemma("familiar", {"familiar"}, "unknown-word-internal", {});
    addLemma("s", {"s"}, "unknown-word-internal", {});
    addLemma("zz", {"zz"}, "unknown-word-internal", {});

    addLemma("[BLANK]", {"blank"}, "blank", {});
    addLemma("[SENTENCE-BEGIN]", {}, "sentence-begin", {"<s>"});
    addLemma("[SENTENCE-END]", {"eos"}, "sentence-end", {"</s>"});

    setCommonParameters();
    setParameter("*.unknown-word-tokenization", "word-start-marked");
    setParameter("*.unknown-word-fallback", "known-excluding");
}

TEST_F(Search, WordStartFallbackSearchTest, SegmentEndClosesOnePendingWordOnce) {
    buildSearch();
    // "_xy zz" is one unknown word of two pieces, not two unknown words.
    auto result = decode({"_xy", "zz"});

    EXPECT_EQ(result.lemmas, std::string("_xy"
                                         " "
                                         "zz"));
    EXPECT_DOUBLE_EQ(result.amScore, 0.0, 1e-4);
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, WordStartPieceClosesThePrecedingWord) {
    buildSearch();
    // The second "_xy" closes "_xy zz" and opens a new word, which the segment end
    // closes in turn: exactly two unknown events.
    auto result = decode({"_xy", "zz", "_xy", "zz"});

    EXPECT_EQ(result.lemmas, std::string("_xy"
                                         " "
                                         "zz"
                                         " "
                                         "_xy"
                                         " "
                                         "zz"));
    EXPECT_DOUBLE_EQ(result.lmScore, 2.0 * costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, ExactKnownPronunciationNeverBecomesUnknown) {
    // An extreme unknown reward must not be able to buy the unknown route for a
    // piece sequence which is an exact known pronunciation.
    setParameter("*.unknown-word-penalty", "-50.0");
    buildSearch();

    auto multiPiece = decode({"_un", "familiar"});
    EXPECT_DOUBLE_EQ(multiPiece.lmScore, costUnfamiliar + costSentenceEnd, 1e-4);

    auto singlePiece = decode({"_cat"});
    EXPECT_DOUBLE_EQ(singlePiece.lmScore, costCat + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, KnownPrefixStillExtendsIntoAnUnknownWord) {
    buildSearch();
    // "_un familiar" is a known word, "_un familiar s" is not; the longer word still
    // needs a fallback path.
    auto result = decode({"_un", "familiar", "s"});

    EXPECT_EQ(result.lemmas, std::string("_un"
                                         " "
                                         "familiar"
                                         " "
                                         "s"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, UnknownWordBiasIsChargedOncePerUnknownWord) {
    setParameter("*.unknown-word-penalty", "2.0");
    buildSearch();

    auto oneWord = decode({"_xy", "zz"});
    EXPECT_DOUBLE_EQ(oneWord.lmScore, costUnknown + 2.0 + costSentenceEnd, 1e-4);

    auto twoWords = decode({"_xy", "zz", "_xy", "zz"});
    EXPECT_DOUBLE_EQ(twoWords.lmScore, 2.0 * (costUnknown + 2.0) + costSentenceEnd, 1e-4);

    // A known word takes no unknown bias at all.
    auto knownWord = decode({"_cat"});
    EXPECT_DOUBLE_EQ(knownWord.lmScore, costCat + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, KnownAndUnknownWordsMixInOneSegment) {
    buildSearch();
    // "_cat" resolves to CAT, "_xy zz" stays unknown, "_the" resolves to THE.
    auto result = decode({"_cat", "_xy", "zz", "_the"});

    EXPECT_DOUBLE_EQ(result.lmScore, costCat + costUnknown + costThe + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, BlanksCreateNoWordBoundary) {
    buildSearch();
    // A blank inside the pending word and one before the segment end must not add
    // any word event.
    auto result = decode({"_xy", "blank", "zz", "blank"});

    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, EmptyPendingWordCreatesNoUnknownWord) {
    buildSearch();
    // Only a blank: no lexical word at all, so the word LM sees the sentence end only.
    auto result = decode({"blank"});

    EXPECT_DOUBLE_EQ(result.lmScore, costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, UnmarkedFirstPieceIsAccepted) {
    buildSearch();
    // A valid beginning-of-sequence piece without the word-start marker must not be
    // rejected just because ordinary pronunciations start with one.
    auto result = decode({"zz"});

    EXPECT_EQ(result.lemmas, std::string("zz"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

/*
 * ==========================================================================
 * === Continuation-marked (BPE style) boundaries and exclusion           ===
 * ==========================================================================
 */

class ContinuationMarkedFallbackSearchTest : public FallbackSearchFixture {
public:
    void setUp();
};

void ContinuationMarkedFallbackSearchTest::setUp() {
    lexicon_ = Core::ref(new Test::Lexicon());
    for (auto const* phoneme : {"kn@@", "own", "ra@@", "word", "blank", "eos"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    // "kn@@ own" is an in-vocabulary word which the fallback pieces can spell too.
    addLemma("knowntwo", {"kn@@ own"}, "", {"KNOWNTWO"});

    addLemma("[UNKNOWN]", {}, "unknown", {"<UNK>"});
    addLemma("ra@@", {"ra@@"}, "unknown-continuation", {});
    addLemma("kn@@", {"kn@@"}, "unknown-continuation", {});
    addLemma("word", {"word"}, "unknown-final", {"<UNK>"});
    addLemma("own", {"own"}, "unknown-final", {"<UNK>"});

    addLemma("[BLANK]", {"blank"}, "blank", {});
    addLemma("[SENTENCE-BEGIN]", {}, "sentence-begin", {"<s>"});
    addLemma("[SENTENCE-END]", {"eos"}, "sentence-end", {"</s>"});

    setCommonParameters();
    setParameter("*.unknown-word-tokenization", "continuation-marked");
}

TEST_F(Search, ContinuationMarkedFallbackSearchTest, FinalPieceClosesTheUnknownWord) {
    setParameter("*.unknown-word-fallback", "known-excluding");
    buildSearch();

    auto result = decode({"ra@@", "word"});
    EXPECT_EQ(result.lemmas, std::string("ra@@"
                                         " "
                                         "word"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);

    auto twoWords = decode({"ra@@", "word", "ra@@", "word"});
    EXPECT_DOUBLE_EQ(twoWords.lmScore, 2.0 * costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, ContinuationMarkedFallbackSearchTest, ExactKnownPronunciationNeverBecomesUnknown) {
    setParameter("*.unknown-word-fallback", "known-excluding");
    setParameter("*.unknown-word-penalty", "-50.0");
    buildSearch();

    // "kn@@ own" spells the known word even though both pieces are fallback pieces.
    auto result = decode({"kn@@", "own"});
    EXPECT_DOUBLE_EQ(result.lmScore, costKnownTwo + costSentenceEnd, 1e-4);
}

TEST_F(Search, ContinuationMarkedFallbackSearchTest, LegacyModeScoresKnownPieceSequenceAsUnknown) {
    // The same input in legacy mode: the word-LM event comes from the final piece's
    // syntactic token, so the known realization is not enforced. This is the
    // behavior the known-excluding mode is compared against.
    setParameter("*.unknown-word-fallback", "legacy");
    setParameter("*.unknown-word-penalty", "-50.0");
    buildSearch();

    auto result = decode({"kn@@", "own"});
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, ContinuationMarkedFallbackSearchTest, DanglingContinuationIsRejectedInStrictMode) {
    setParameter("*.unknown-word-fallback", "known-excluding");
    setParameter("*.sentence-end-fall-back", "false");
    buildSearch();

    // "ra@@" keeps the word open and no final piece follows. Even though emitting it
    // is acoustically free in this frame, the hypothesis does not end in a valid final
    // state, so the recognized output must not contain the dangling piece and has to
    // pay for a different label instead.
    auto result = decode({"ra@@"});
    EXPECT_TRUE(result.lemmas.find("ra@@") == std::string::npos);
    EXPECT_TRUE(result.amScore > 0.0);
}
