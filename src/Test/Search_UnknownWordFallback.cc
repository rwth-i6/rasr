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
#include <map>
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

    // Decode one frame per entry of `frameScores`, giving the listed labels their
    // listed score and every other label `suppressed`. Lets a test express that two
    // labels are acoustically close rather than mutually exclusive.
    Result decodeGraded(std::vector<std::map<std::string, float>> const& frameScores);

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
    // Boundary-independent tying, so one subword label has one emission index whether
    // it is a whole word or the first piece of a longer one. This is what a real
    // subword CTC setup must use, and without it the ordinary and the fallback
    // realization of the same piece would not even be the same acoustic label.
    setParameter("*.state-tying.type", "monophone");
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
    std::vector<std::map<std::string, float>> frameScores;
    for (auto const& label : labels) {
        frameScores.push_back({{label, 0.0f}});
    }
    return decodeGraded(frameScores);
}

FallbackSearchFixture::Result FallbackSearchFixture::decodeGraded(std::vector<std::map<std::string, float>> const& frameScores) {
    search_->enterSegment();

    std::vector<std::shared_ptr<f32[]>> frames;  // keep the buffers alive until finishSegment
    for (auto const& scores : frameScores) {
        std::shared_ptr<f32[]> frame(new f32[numEmissions_]);
        std::fill(frame.get(), frame.get() + numEmissions_, suppressed);
        for (auto const& [label, score] : scores) {
            require_lt(static_cast<size_t>(emissionIndexOf(label)), numEmissions_);
            frame[emissionIndexOf(label)] = score;
        }
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
    for (auto const* phoneme : {"_cat", "_un", "_the", "_xy", "_sep", "cat", "the", "familiar", "fam", "iliar", "s", "zz", "blank", "eos"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    // Ordinary in-vocabulary words.
    addLemma("cat", {"_cat"}, "", {"CAT"});
    addLemma("the", {"_the"}, "", {"THE"});
    addLemma("unfamiliar", {"_un familiar"}, "", {"UNFAMILIAR"});
    // Recognizable, but outside the word LM's vocabulary: scored with the unknown token.
    addLemma("unesses", {"_un s"}, "", {"<UNK>"});

    // Fallback inventory. "_xy" and "zz" spell no in-vocabulary word at all.
    addLemma("[UNKNOWN]", {}, "unknown", {"<UNK>"});
    addLemma("_cat", {"_cat"}, "unknown-word-start", {});
    addLemma("_un", {"_un"}, "unknown-word-start", {});
    addLemma("_the", {"_the"}, "unknown-word-start", {});
    addLemma("_xy", {"_xy"}, "unknown-word-start", {});
    addLemma("familiar", {"familiar"}, "unknown-word-internal", {});
    addLemma("s", {"s"}, "unknown-word-internal", {});
    addLemma("zz", {"zz"}, "unknown-word-internal", {});
    // An alternate tokenization of the known spelling "unfamiliar".
    addLemma("cat", {"cat"}, "unknown-word-internal", {});
    addLemma("the", {"the"}, "unknown-word-internal", {});
    addLemma("fam", {"fam"}, "unknown-word-internal", {});
    addLemma("iliar", {"iliar"}, "unknown-word-internal", {});
    // A standalone word-start marker: a boundary without lexical content. Declared by
    // additionally listing it as a "nonword".
    Bliss::Lemma* separator = addLemma("_sep", {"_sep"}, "unknown-word-start", {});
    lexicon_->defineSpecialLemma("nonword", separator);

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

TEST_F(Search, WordStartFallbackSearchTest, WithoutAPerPieceCostALongPendingWordSwallowsTheSentence) {
    buildSearch();

    // Both words are in the vocabulary and the word-start pieces are acoustically
    // preferred, but only by a little: the unmarked variant of the second word costs
    // 0.5 more. Keeping one word open over the whole utterance avoids one word-LM
    // event, and the cost of an unknown word does not depend on its length, so the
    // degenerate reading wins -- with an unlimited beam, i.e. this is a scoring
    // preference and not a pruning artifact.
    //
    //   correct   : THE + CAT + </s>        = 2.303 + 0.693 + 0.223 = 3.219, am 0.0
    //   degenerate: UNK("_the cat") + </s>  = 1.386         + 0.223 = 1.609, am 0.5
    auto result = decodeGraded({{{"_the", 0.0f}}, {{"_cat", 0.0f}, {"cat", 0.5f}}});

    EXPECT_EQ(result.lemmas, std::string("_the"
                                         " "
                                         "cat [1]"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
    EXPECT_DOUBLE_EQ(result.amScore, 0.5, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, PerPieceCostStopsTheRunawayPendingWord) {
    // One continuation piece at 2.0 outweighs the 1.11 the degenerate reading saves.
    setParameter("*.unknown-piece-penalty", "2.0");
    buildSearch();

    auto result = decodeGraded({{{"_the", 0.0f}}, {{"_cat", 0.0f}, {"cat", 0.5f}}});

    // Two word events with the words' own LM tokens rather than one unknown event.
    // Both words resolve to known pronunciations, so the provisional per-piece cost is
    // refunded and the result is scored exactly as on the ordinary route -- the two
    // routes tie, so which lemma identities the traceback shows is arbitrary and only
    // the score and the absence of the unmarked piece are asserted here.
    EXPECT_TRUE(result.lemmas.find("cat [1]") == std::string::npos);
    EXPECT_DOUBLE_EQ(result.lmScore, costThe + costCat + costSentenceEnd, 1e-4);
    EXPECT_DOUBLE_EQ(result.amScore, 0.0, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, UnknownWordCostGrowsWithItsLength) {
    setParameter("*.unknown-piece-penalty", "0.5");
    buildSearch();

    // The piece that opens a word is free; only continuing one costs, so an n-piece
    // unknown word costs beta + alpha * (n - 1).
    auto onePiece = decode({"zz"});
    EXPECT_DOUBLE_EQ(onePiece.lmScore, costUnknown + costSentenceEnd, 1e-4);

    auto twoPieces = decode({"_xy", "zz"});
    EXPECT_DOUBLE_EQ(twoPieces.lmScore, costUnknown + 0.5 + costSentenceEnd, 1e-4);

    auto threePieces = decode({"_un", "fam", "iliar"});
    EXPECT_DOUBLE_EQ(threePieces.lmScore, costUnknown + 2.0 * 0.5 + costSentenceEnd, 1e-4);

    // A separator carries no lexical content, so it neither opens nor continues a word.
    auto withSeparator = decode({"_xy", "zz", "_sep"});
    EXPECT_DOUBLE_EQ(withSeparator.lmScore, costUnknown + 0.5 + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, KnownWordOnTheFallbackRouteIsRefundedThePerPieceCost) {
    setParameter("*.unknown-piece-penalty", "5.0");
    buildSearch();

    // The unmarked first piece forces everything onto the fallback route, which cannot
    // be left again. "zz" is a one-piece unknown word, then "_un familiar" is closed by
    // the segment end and resolves to UNFAMILIAR: its continuation piece was charged
    // while the word was still open and has to be refunded, so that a known word costs
    // the same whichever route recognized it.
    auto result = decode({"zz", "_un", "familiar"});

    EXPECT_EQ(result.lemmas, std::string("zz _un familiar"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costUnfamiliar + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, OrdinaryLemmaWithTheUnknownTokenTakesTheUnknownPenalty) {
    // The "extended lexicon" pattern: a word which should be recognizable but is
    // outside the word LM's vocabulary is an ordinary lemma carrying the unknown token.
    // It needs no fallback sub-tree at all, but it is an unknown word and takes beta.
    setParameter("*.unknown-word-penalty", "3.0");
    setParameter("*.unknown-piece-penalty", "7.0");
    buildSearch();

    auto result = decode({"_un", "s"});

    // Its spelling is attested by the lexicon, so it pays beta but no per-piece cost,
    // even though it is two pieces long. The ordinary route and the fallback route
    // (which resolves the same pieces into this lemma) therefore score it identically
    // and tie, so only the score is asserted and not which identity the traceback shows.
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + 3.0 + costSentenceEnd, 1e-4);
    EXPECT_DOUBLE_EQ(result.amScore, 0.0, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, SeparatorPiecesCreateNoPhantomWord) {
    buildSearch();

    // A trailing separator closes the pending word and leaves nothing behind.
    auto trailing = decode({"_xy", "zz", "_sep"});
    EXPECT_DOUBLE_EQ(trailing.lmScore, costUnknown + costSentenceEnd, 1e-4);

    // Repeated separators likewise add no word.
    auto repeated = decode({"_xy", "zz", "_sep", "_sep"});
    EXPECT_DOUBLE_EQ(repeated.lmScore, costUnknown + costSentenceEnd, 1e-4);

    // A separator at the very beginning does not open one either.
    auto leading = decode({"_sep", "_xy", "zz"});
    EXPECT_DOUBLE_EQ(leading.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, KnownWordIsResolvedOnTheFallbackRoute) {
    buildSearch();
    // The unmarked first piece forces the hypothesis into the fallback sub-tree, which
    // it cannot leave again, so "_cat" is closed and resolved there rather than on the
    // ordinary route. It must still be scored with its own LM token.
    auto result = decode({"zz", "_cat"});

    EXPECT_EQ(result.lemmas, std::string("zz"
                                         " "
                                         "_cat"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costCat + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, AlternateTokenizationOfAKnownSpellingStaysUnknown) {
    buildSearch();
    // "_un fam iliar" detokenizes to the known spelling "unfamiliar", but it is not the
    // token sequence the lexicon lists. The exact-token-sequence rule of this prototype
    // deliberately reports it as an unknown word; recognizing it would need a declared
    // detokenized-spelling mapping.
    auto result = decode({"_un", "fam", "iliar"});

    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
}

TEST_F(Search, WordStartFallbackSearchTest, RepeatedPieceSeparatedByBlankStaysTwoPieces) {
    buildSearch();
    // Two identical pieces separated by a blank are two pieces of one pending word,
    // and the blank itself is not a boundary.
    auto result = decode({"_xy", "zz", "blank", "zz"});

    EXPECT_EQ(result.lemmas, std::string("_xy"
                                         " "
                                         "zz"
                                         " "
                                         "[BLANK]"
                                         " "
                                         "zz"));
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

TEST_F(Search, ContinuationMarkedFallbackSearchTest, SinglePieceUnknownWord) {
    setParameter("*.unknown-word-fallback", "known-excluding");
    buildSearch();

    // A final piece on its own is a complete one-piece unknown word.
    auto result = decode({"word"});
    EXPECT_EQ(result.lemmas, std::string("word"));
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown + costSentenceEnd, 1e-4);
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
    // behavior the known-excluding mode is compared against. The unknown-word penalty
    // applies here too, since it follows the unknown token rather than the route, so
    // both modes can be swept on the same axis.
    setParameter("*.unknown-word-fallback", "legacy");
    setParameter("*.unknown-word-penalty", "-50.0");
    buildSearch();

    auto result = decode({"kn@@", "own"});
    EXPECT_DOUBLE_EQ(result.lmScore, costUnknown - 50.0 + costSentenceEnd, 1e-4);
}

TEST_F(Search, ContinuationMarkedFallbackSearchTest, PendingFallbackWordDoesNotPruneAwayTheOrdinaryResult) {
    // A large unknown-word penalty with tight word-end pruning. The correct reading is
    // the in-vocabulary one, which the closed tree finds; adding the fallback sub-tree
    // must not destroy it.
    //
    // The competing fallback reading "ra@@ ra@@" never closes its word. If `beta` were
    // charged when a fallback word closes rather than when it opens, that hypothesis
    // would stay at its bare acoustic cost for the whole utterance, set the word-end
    // pruning threshold, and prune away the ordinary hypothesis which honestly paid the
    // LM cost of the word it completed. Since a pending word cannot end a segment under
    // this tokenization, the result would then be no hypothesis at all -- the fallback
    // making the search fail on input the closed tree handles.
    setParameter("*.unknown-word-fallback", "known-excluding");
    setParameter("*.unknown-word-penalty", "100.0");
    setParameter("*.score-threshold", "1.0");
    setParameter("*.word-end-score-threshold", "0.5");
    setParameter("*.sentence-end-fall-back", "false");
    buildSearch();

    auto result = decodeGraded({{{"kn@@", 0.0f}, {"ra@@", 0.0f}},
                                {{"own", 0.0f}, {"ra@@", 0.5f}}});

    EXPECT_FALSE(result.empty);
    EXPECT_DOUBLE_EQ(result.lmScore, costKnownTwo + costSentenceEnd, 1e-4);
    EXPECT_DOUBLE_EQ(result.amScore, 0.0, 1e-4);
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
