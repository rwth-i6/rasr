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

#include <Test/File.hh>
#include <Test/Lexicon.hh>
#include <Test/UnitTest.hh>

#include <deque>
#include <set>
#include <string>
#include <vector>

#include <Am/AcousticModel.hh>
#include <Am/Module.hh>
#include <Core/Application.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/TreeBuilder.hh>

/*
 * Shared helpers for the open-vocabulary fallback topology tests.
 */
class FallbackTreeFixture : public Test::ConfigurableFixture {
protected:
    static Bliss::Lemma* addLemma(Test::Lexicon&                  lexicon,
                                  std::string const&              orth,
                                  std::vector<std::string> const& pronunciations,
                                  std::string const&              special,
                                  std::vector<std::string> const& syntacticTokens);

    // Does any state that is a successor of `root` carry an exit for `lemma` with
    // pronunciation `pronunciation` leading to `transitState`?
    static bool hasExit(Search::PersistentStateTree const& network,
                        Bliss::Lexicon const&              lexicon,
                        Search::StateId                    root,
                        Bliss::Lemma const*                lemma,
                        std::string const&                 pronunciation,
                        Search::StateId                    transitState);

    // Is any state reachable from `root` in one label step a successor for `lemma`?
    static bool reaches(Search::PersistentStateTree const& network,
                        Bliss::Lexicon const&              lexicon,
                        Search::StateId                    root,
                        Bliss::Lemma const*                lemma);

    void setAcousticModelParameters();
    void buildTree();

    Core::Ref<Test::Lexicon>               lexicon_;
    Core::Ref<const Am::AcousticModel>     acousticModel_;
    Core::Ref<Search::PersistentStateTree> network_;
};

Bliss::Lemma* FallbackTreeFixture::addLemma(Test::Lexicon&                  lexicon,
                                            std::string const&              orth,
                                            std::vector<std::string> const& pronunciations,
                                            std::string const&              special,
                                            std::vector<std::string> const& syntacticTokens) {
    Bliss::Lemma* lemma = lexicon.newLemma();
    if (!special.empty()) {
        lexicon.defineSpecialLemma(special, lemma);
    }
    for (auto const& pronunciation : pronunciations) {
        Bliss::Pronunciation* pron   = nullptr;
        Core::Status          status = lexicon.getPronunciation(pronunciation, pron);
        require(status.ok());
        lexicon.addPronunciation(lemma, pron);
    }
    lexicon.setOrthographicForms(lemma, {orth});
    lexicon.setDefaultLemmaName(lemma);
    lexicon.setSyntacticTokenSequence(lemma, syntacticTokens);
    lexicon.setDefaultEvaluationToken(lemma);
    return lemma;
}

bool FallbackTreeFixture::hasExit(Search::PersistentStateTree const& network,
                                  Bliss::Lexicon const&              lexicon,
                                  Search::StateId                    root,
                                  Bliss::Lemma const*                lemma,
                                  std::string const&                 pronunciation,
                                  Search::StateId                    transitState) {
    for (auto successor = network.structure.successors(root); successor; ++successor) {
        if (successor.isLabel()) {
            continue;
        }
        for (auto target = network.structure.successors(*successor); target; ++target) {
            if (!target.isLabel()) {
                continue;
            }
            auto const& exit      = network.exits[target.label()];
            auto const* lemmaPron = lexicon.lemmaPronunciation(exit.pronunciation);
            if (exit.transitState == transitState && lemmaPron->lemma() == lemma &&
                lemmaPron->pronunciation()->format(lexicon.phonemeInventory()) == pronunciation) {
                return true;
            }
        }
    }
    return false;
}

bool FallbackTreeFixture::reaches(Search::PersistentStateTree const& network,
                                  Bliss::Lexicon const&              lexicon,
                                  Search::StateId                    root,
                                  Bliss::Lemma const*                lemma) {
    // Breadth-first over everything reachable from `root` without passing through
    // another root, so that words of more than one piece are found as well.
    std::set<Search::StateId>   visited;
    std::deque<Search::StateId> pending{root};
    while (!pending.empty()) {
        Search::StateId const state = pending.front();
        pending.pop_front();
        if (!visited.insert(state).second) {
            continue;
        }
        for (auto target = network.structure.successors(state); target; ++target) {
            if (target.isLabel()) {
                auto const& exit = network.exits[target.label()];
                if (lexicon.lemmaPronunciation(exit.pronunciation)->lemma() == lemma) {
                    return true;
                }
                continue;
            }
            if (*target != root && network.isRoot(*target)) {
                continue;  // a new word starts there, it is not part of this one
            }
            pending.push_back(*target);
        }
    }
    return false;
}

void FallbackTreeFixture::setAcousticModelParameters() {
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
}

void FallbackTreeFixture::buildTree() {
    acousticModel_ = Am::Module::instance().createAcousticModel(
            select("acoustic-model"), lexicon_, Am::AcousticModel::noEmissions);
    network_ = Core::ref(new Search::PersistentStateTree(config, acousticModel_, lexicon_, {}));
    CtcTreeBuilder builder(config, *lexicon_, *acousticModel_, *network_);
    builder.build();
}

/*
 * ================================================================
 * === Continuation-marked (BPE style) fallback topology         ===
 * ================================================================
 */

class ContinuationMarkedTreeBuilderTest : public FallbackTreeFixture {
public:
    void setUp();

protected:
    Bliss::Lemma const* continuationRa_;
    Bliss::Lemma const* continuationRe_;
    Bliss::Lemma const* continuationKn_;
    Bliss::Lemma const* finalWord_;
    Bliss::Lemma const* finalPiece_;
    Bliss::Lemma const* finalOwn_;
    Bliss::Lemma const* knownOnePiece_;
    Bliss::Lemma const* knownTwoPiece_;
};

void ContinuationMarkedTreeBuilderTest::setUp() {
    lexicon_ = Core::ref(new Test::Lexicon());
    for (auto const* phoneme : {"known", "kn@@", "own", "ra@@", "re@@", "word", "piece", "si", "blank"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    knownOnePiece_ = addLemma(*lexicon_, "KNOWN", {"known"}, "", {"KNOWN"});
    // A known word which the fallback pieces can spell as well; used to check that
    // the fallback sub-tree does not have to avoid known prefixes.
    knownTwoPiece_ = addLemma(*lexicon_, "KNOWNTWO", {"kn@@ own"}, "", {"KNOWNTWO"});

    addLemma(*lexicon_, "[UNKNOWN]", {}, "unknown", {"<UNK>"});
    finalWord_      = addLemma(*lexicon_, "word", {"word"}, "unknown-final", {"<UNK>"});
    finalPiece_     = addLemma(*lexicon_, "piece", {"piece"}, "unknown-final", {"<UNK>"});
    finalOwn_       = addLemma(*lexicon_, "own", {"own"}, "unknown-final", {"<UNK>"});
    continuationRa_ = addLemma(*lexicon_, "ra@@", {"ra@@"}, "unknown-continuation", {});
    continuationRe_ = addLemma(*lexicon_, "re@@", {"re@@"}, "unknown-continuation", {});
    continuationKn_ = addLemma(*lexicon_, "kn@@", {"kn@@"}, "unknown-continuation", {});
    addLemma(*lexicon_, "[SILENCE]", {"si"}, "silence", {});
    addLemma(*lexicon_, "[BLANK]", {"blank"}, "blank", {});

    setAcousticModelParameters();
}

TEST_F(Search, ContinuationMarkedTreeBuilderTest, LegacyConstrainsUnknownPieceSequences) {
    setParameter("*.unknown-word-fallback", "legacy");
    buildTree();

    Search::StateId const unknownWordRoot = network_->unknownWordRoot;
    EXPECT_NE(unknownWordRoot, Search::invalidTreeNodeIndex);
    EXPECT_TRUE(network_->otherRootStates.contains(unknownWordRoot));

    // A continuation-marked word must still receive its final piece, so the pending
    // root is deliberately not a valid segment-final state.
    EXPECT_FALSE(network_->finalStates.contains(unknownWordRoot));
    EXPECT_TRUE(network_->finalStates.contains(network_->rootState));

    Bliss::Lemma const* blankLemma = lexicon_->specialLemma("blank");

    // A fallback word may open at the ordinary root and continues in the pending root.
    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, continuationRa_, "ra@@", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, continuationRe_, "re@@", unknownWordRoot));
    // A final piece closes the word and returns to the ordinary root.
    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, finalWord_, "word", network_->rootState));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, finalPiece_, "piece", network_->rootState));
    // Blank stays available while a fallback word is pending.
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, blankLemma, "blank", unknownWordRoot));
    // No ordinary word can be started in the middle of an unfinished fallback word.
    EXPECT_FALSE(reaches(*network_, *lexicon_, unknownWordRoot, knownOnePiece_));
    EXPECT_FALSE(reaches(*network_, *lexicon_, unknownWordRoot, knownTwoPiece_));
}

TEST_F(Search, ContinuationMarkedTreeBuilderTest, KnownExcludingKeepsTheSameTopology) {
    // The known-word exclusion is a scoring decision of the search, not a topology
    // change: the fallback must still be able to spell a known piece sequence.
    setParameter("*.unknown-word-fallback", "known-excluding");
    buildTree();

    Search::StateId const unknownWordRoot = network_->unknownWordRoot;
    EXPECT_NE(unknownWordRoot, Search::invalidTreeNodeIndex);
    EXPECT_FALSE(network_->finalStates.contains(unknownWordRoot));

    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, continuationKn_, "kn@@", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, finalOwn_, "own", network_->rootState));
    // The ordinary realization of the same piece sequence is untouched.
    EXPECT_TRUE(reaches(*network_, *lexicon_, network_->rootState, knownTwoPiece_));
}

TEST_F(Search, ContinuationMarkedTreeBuilderTest, DisabledDropsTheFallbackSubTree) {
    setParameter("*.unknown-word-fallback", "disabled");
    buildTree();

    EXPECT_EQ(network_->unknownWordRoot, Search::invalidTreeNodeIndex);
    EXPECT_EQ(network_->unknownWordFallbackPolicy, u32(0));
    EXPECT_FALSE(reaches(*network_, *lexicon_, network_->rootState, continuationRa_));
    EXPECT_TRUE(reaches(*network_, *lexicon_, network_->rootState, knownOnePiece_));
}

TEST_F(Search, ContinuationMarkedTreeBuilderTest, PolicyKeyDistinguishesModes) {
    setParameter("*.unknown-word-fallback", "legacy");
    buildTree();
    u32 const legacyPolicy = network_->unknownWordFallbackPolicy;

    setParameter("*.unknown-word-fallback", "known-excluding");
    buildTree();
    u32 const knownExcludingPolicy = network_->unknownWordFallbackPolicy;

    // A cached tree image records this value, so an image built under a different
    // policy cannot be reused silently.
    EXPECT_NE(legacyPolicy, u32(0));
    EXPECT_NE(knownExcludingPolicy, u32(0));
    EXPECT_NE(legacyPolicy, knownExcludingPolicy);
}

/*
 * ================================================================
 * === Word-start-marked (SentencePiece style) fallback topology ===
 * ================================================================
 */

class WordStartMarkedTreeBuilderTest : public FallbackTreeFixture {
public:
    void setUp();

protected:
    Bliss::Lemma const* startCat_;
    Bliss::Lemma const* startUn_;
    Bliss::Lemma const* startThe_;
    Bliss::Lemma const* internalFamiliar_;
    Bliss::Lemma const* internalS_;
    Bliss::Lemma const* knownCat_;
    Bliss::Lemma const* knownUnfamiliar_;
};

void WordStartMarkedTreeBuilderTest::setUp() {
    lexicon_ = Core::ref(new Test::Lexicon());
    for (auto const* phoneme : {"_cat", "_un", "_the", "familiar", "s", "blank"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    knownCat_        = addLemma(*lexicon_, "cat", {"_cat"}, "", {"CAT"});
    knownUnfamiliar_ = addLemma(*lexicon_, "unfamiliar", {"_un familiar"}, "", {"UNFAMILIAR"});
    addLemma(*lexicon_, "the", {"_the"}, "", {"THE"});

    addLemma(*lexicon_, "[UNKNOWN]", {}, "unknown", {"<UNK>"});
    startCat_         = addLemma(*lexicon_, "_cat", {"_cat"}, "unknown-word-start", {});
    startUn_          = addLemma(*lexicon_, "_un", {"_un"}, "unknown-word-start", {});
    startThe_         = addLemma(*lexicon_, "_the", {"_the"}, "unknown-word-start", {});
    internalFamiliar_ = addLemma(*lexicon_, "familiar", {"familiar"}, "unknown-word-internal", {});
    internalS_        = addLemma(*lexicon_, "s", {"s"}, "unknown-word-internal", {});
    addLemma(*lexicon_, "[BLANK]", {"blank"}, "blank", {});

    setAcousticModelParameters();
    setParameter("*.unknown-word-tokenization", "word-start-marked");
    setParameter("*.unknown-word-fallback", "known-excluding");
}

TEST_F(Search, WordStartMarkedTreeBuilderTest, OnlyWordStartPiecesOpenAFallbackWord) {
    buildTree();

    Search::StateId const unknownWordRoot = network_->unknownWordRoot;
    EXPECT_NE(unknownWordRoot, Search::invalidTreeNodeIndex);

    // Every fallback piece keeps the word pending; the boundary is resolved by the
    // search when the next word-start piece arrives or the segment ends.
    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, startUn_, "_un", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, internalFamiliar_, "familiar", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, startCat_, "_cat", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, startThe_, "_the", unknownWordRoot));

    // A word-internal piece must not start a word at the ordinary root: otherwise
    // "_un familiar" could be segmented into two words and charged two unknown events.
    EXPECT_FALSE(hasExit(*network_, *lexicon_, network_->rootState, internalFamiliar_, "familiar", unknownWordRoot));
    EXPECT_FALSE(hasExit(*network_, *lexicon_, network_->rootState, internalS_, "s", unknownWordRoot));
}

TEST_F(Search, WordStartMarkedTreeBuilderTest, PendingRootIsFinalAndKeepsBlank) {
    buildTree();

    Search::StateId const unknownWordRoot = network_->unknownWordRoot;
    // The segment end closes a pending word, so unlike the continuation-marked case
    // the pending root is a valid final state.
    EXPECT_TRUE(network_->finalStates.contains(unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, lexicon_->specialLemma("blank"), "blank", unknownWordRoot));
}

TEST_F(Search, WordStartMarkedTreeBuilderTest, OrdinaryTreeIsUnchanged) {
    buildTree();

    // Ordinary words keep their own path and their own exit to the ordinary root.
    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, knownCat_, "_cat", network_->rootState));
    EXPECT_TRUE(reaches(*network_, *lexicon_, network_->rootState, knownUnfamiliar_));
    // No ordinary word can be started while a fallback word is pending.
    EXPECT_FALSE(reaches(*network_, *lexicon_, network_->unknownWordRoot, knownCat_));
}

/*
 * ================================================================
 * === Search-tree image identity                               ===
 * ================================================================
 */

class FallbackTreeCacheTest : public ContinuationMarkedTreeBuilderTest {
public:
    void tearDown();

protected:
    static constexpr char const* archiveName = "unit-test-tree-cache";

    // Point the persistent search-tree image at `file`. The archive itself lives in
    // the application, so it has to be (re-)registered there explicitly.
    void useCache(std::string const& file);

    // Build a fresh network under the parameters set so far and try to load the image.
    bool readTree();

    Core::Ref<Search::PersistentStateTree> reloaded_;
};

void FallbackTreeCacheTest::tearDown() {
    Core::Application::us()->updateCacheArchive(archiveName, config, true);
}

void FallbackTreeCacheTest::useCache(std::string const& file) {
    setParameter("*.search-network.cache-archive", archiveName);
    setParameter(std::string("*.") + archiveName + ".file", file);
    Core::Application::us()->updateCacheArchive(archiveName, config);
}

bool FallbackTreeCacheTest::readTree() {
    acousticModel_ = Am::Module::instance().createAcousticModel(
            select("acoustic-model"), lexicon_, Am::AcousticModel::noEmissions);
    reloaded_ = Core::ref(new Search::PersistentStateTree(config, acousticModel_, lexicon_, {}));
    return reloaded_->read();
}

TEST_F(Search, FallbackTreeCacheTest, ImageOfADifferentPolicyIsRejected) {
    Test::Directory tempDir;
    Test::File      cacheFile(tempDir, "tree-cache.bin");
    useCache(cacheFile.path());

    // Write an image built for the legacy policy ...
    setParameter("*.unknown-word-fallback", "legacy");
    buildTree();
    EXPECT_TRUE(network_->write(0));
    EXPECT_NE(network_->unknownWordFallbackPolicy, u32(0));

    // ... reading it back under the same policy works ...
    useCache(cacheFile.path());
    EXPECT_TRUE(readTree());

    // ... but under the known-excluding policy the image must be rejected instead of
    // silently decoding with a topology built for different semantics.
    setParameter("*.unknown-word-fallback", "known-excluding");
    useCache(cacheFile.path());
    EXPECT_FALSE(readTree());

    // The same holds for turning the fallback off entirely.
    setParameter("*.unknown-word-fallback", "disabled");
    useCache(cacheFile.path());
    EXPECT_FALSE(readTree());
}

TEST_F(Search, FallbackTreeCacheTest, PolicyAndRootSurviveSerialization) {
    Test::Directory tempDir;
    Test::File      cacheFile(tempDir, "tree-cache.bin");
    useCache(cacheFile.path());
    setParameter("*.unknown-word-fallback", "known-excluding");

    buildTree();
    EXPECT_TRUE(network_->write(0));
    Search::StateId const writtenRoot   = network_->unknownWordRoot;
    u32 const             writtenPolicy = network_->unknownWordFallbackPolicy;

    useCache(cacheFile.path());
    EXPECT_TRUE(readTree());
    EXPECT_EQ(reloaded_->unknownWordRoot, writtenRoot);
    EXPECT_EQ(reloaded_->unknownWordFallbackPolicy, writtenPolicy);
    EXPECT_TRUE(reloaded_->finalStates.contains(reloaded_->rootState));
    EXPECT_FALSE(reloaded_->finalStates.contains(reloaded_->unknownWordRoot));
}
