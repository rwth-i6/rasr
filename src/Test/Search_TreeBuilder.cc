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

#include <Test/Lexicon.hh>
#include <Test/UnitTest.hh>

#include <string>
#include <vector>

#include <Am/AcousticModel.hh>
#include <Am/Module.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/TreeBuilder.hh>

class UnknownWordTreeBuilderTest : public Test::ConfigurableFixture {
public:
    void setUp();

protected:
    static Bliss::Lemma* addLemma(Test::Lexicon&                  lexicon,
                                  std::string const&              orth,
                                  std::vector<std::string> const& pronunciations,
                                  std::string const&              special,
                                  std::vector<std::string> const& syntacticTokens);

    static bool hasExit(Search::PersistentStateTree const& network,
                        Bliss::Lexicon const&              lexicon,
                        Search::StateId                    root,
                        Bliss::Lemma const*                lemma,
                        std::string const&                 pronunciation,
                        Search::StateId                    transitState);

    Core::Ref<Test::Lexicon>               lexicon_;
    Core::Ref<const Am::AcousticModel>     acousticModel_;
    Core::Ref<Search::PersistentStateTree> network_;
    Bliss::Lemma const*                    continuationRa_;
    Bliss::Lemma const*                    continuationRe_;
    Bliss::Lemma const*                    finalWord_;
    Bliss::Lemma const*                    finalPiece_;
};

Bliss::Lemma* UnknownWordTreeBuilderTest::addLemma(Test::Lexicon&                  lexicon,
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

bool UnknownWordTreeBuilderTest::hasExit(Search::PersistentStateTree const& network,
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

void UnknownWordTreeBuilderTest::setUp() {
    lexicon_ = Core::ref(new Test::Lexicon());
    for (auto const* phoneme : {"known", "ra@@", "re@@", "word", "piece", "si", "blank"}) {
        lexicon_->addPhoneme(phoneme, false);
    }

    addLemma(*lexicon_, "KNOWN", {"known"}, "", {"KNOWN"});
    addLemma(*lexicon_, "[UNKNOWN]", {"word"}, "unknown", {"[UNKNOWN]"});
    finalWord_      = addLemma(*lexicon_, "word", {"word"}, "unknown-final", {"[UNKNOWN]"});
    finalPiece_     = addLemma(*lexicon_, "piece", {"piece"}, "unknown-final", {"[UNKNOWN]"});
    continuationRa_ = addLemma(*lexicon_, "ra@@", {"ra@@"}, "unknown-continuation", {});
    continuationRe_ = addLemma(*lexicon_, "re@@", {"re@@"}, "unknown-continuation", {});
    addLemma(*lexicon_, "[SILENCE]", {"si"}, "silence", {});
    addLemma(*lexicon_, "[BLANK]", {"blank"}, "blank", {});

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

    acousticModel_ = Am::Module::instance().createAcousticModel(
            select("acoustic-model"), lexicon_, Am::AcousticModel::noEmissions);
    network_ = Core::ref(new Search::PersistentStateTree(config, acousticModel_, lexicon_, {}));
    CtcTreeBuilder builder(config, *lexicon_, *acousticModel_, *network_);
    builder.build();
}

TEST_F(Search, UnknownWordTreeBuilderTest, ConstrainsUnknownPieceSequences) {
    EXPECT_EQ(network_->otherRootStates.size(), size_t(1));
    Search::StateId unknownWordRoot = *network_->otherRootStates.begin();
    EXPECT_FALSE(network_->finalStates.contains(unknownWordRoot));

    Bliss::Lemma const* blankLemma = lexicon_->specialLemma("blank");
    Bliss::Lemma const* knownLemma = lexicon_->lemma("KNOWN");

    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, continuationRa_, "ra@@", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, continuationRe_, "re@@", unknownWordRoot));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, network_->rootState, finalWord_, "word", network_->rootState));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, finalPiece_, "piece", network_->rootState));
    EXPECT_TRUE(hasExit(*network_, *lexicon_, unknownWordRoot, blankLemma, "blank", unknownWordRoot));
    EXPECT_FALSE(hasExit(*network_, *lexicon_, unknownWordRoot, knownLemma, "known", network_->rootState));
}
