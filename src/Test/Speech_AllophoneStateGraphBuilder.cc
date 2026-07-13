/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
/**
 * Test cases for Speech::CTCTopologyGraphBuilder.
 */

#include <Test/Lexicon.hh>
#include <Test/UnitTest.hh>

#include <deque>
#include <memory>
#include <set>
#include <utility>

#include <Am/AcousticModel.hh>
#include <Am/Module.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Input.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Static.hh>
#include <Speech/AllophoneStateGraphBuilder.hh>

/**
 * Fixture that sets up a minimal, file-free CTC acoustic model.
 *
 * The lexicon uses context-independent single-state phonemes, so every word
 * maps to exactly one label ("each word is its own token"). Blank is enabled
 * via a special "blank" lemma; the transition model is configured for the CTC
 * topology (forward:0, skip:inf, exit:0, loop:inf for entry states else 0).
 */
class CTCGraphBuilderTest : public Test::ConfigurableFixture {
public:
    void setUp();

protected:
    Core::Ref<const Bliss::Lexicon>                     lexicon_;
    Core::Ref<const Am::AcousticModel>                  acousticModel_;
    std::unique_ptr<Speech::AllophoneStateGraphBuilder> builder_;
    Fsa::LabelId                                        blankId_ = Fsa::InvalidLabelId;
};

void CTCGraphBuilderTest::setUp() {
    // --- lexicon: three single-token words A B C, plus silence and blank ---
    Core::Ref<Test::Lexicon> lexicon(new Test::Lexicon());
    lexicon->addPhoneme("a", false);  // context-independent
    lexicon->addPhoneme("b", false);
    lexicon->addPhoneme("c", false);
    lexicon->addPhoneme("si", false);
    lexicon->addPhoneme("blank", false);
    lexicon->addLemma("A", "a");
    lexicon->addLemma("B", "b");
    lexicon->addLemma("C", "c");
    lexicon->addLemma("[SILENCE]", "si", "silence");
    lexicon->addLemma("[BLANK]", "blank", "blank");
    lexicon_ = lexicon;

    // --- acoustic model configuration (no external files needed) ---
    // Resource names are matched against the fully-qualified parameter path
    // (e.g. UNNAMED.acoustic-model.state-tying.type), so use a "*." wildcard prefix.
    setParameter("*.state-tying.type", "no-tying-dense");
    setParameter("*.hmm.states-per-phone", "1");
    setParameter("*.hmm.state-repetitions", "1");
    setParameter("*.hmm.across-word-model", "no");

    // CTC transitions: forward/exit default to 0.0, so only loop and skip need setting.
    const char* entryStates[] = {"entry-m1", "entry-m2"};
    for (const char* st : entryStates) {
        setParameter(std::string("*.tdp.") + st + ".loop", "infinity");
        setParameter(std::string("*.tdp.") + st + ".skip", "infinity");
    }
    const char* labelStates[] = {"silence", "state-0", "state-1"};
    for (const char* st : labelStates) {
        setParameter(std::string("*.tdp.") + st + ".loop", "0");
        setParameter(std::string("*.tdp.") + st + ".skip", "infinity");
    }

    acousticModel_ = Am::Module::instance().createAcousticModel(
            select("acoustic-model"), lexicon_, Am::AcousticModel::noEmissions);
    EXPECT_TRUE(acousticModel_);
    blankId_ = acousticModel_->blankAllophoneStateIndex();
    EXPECT_NE(blankId_, Fsa::InvalidLabelId);

    // full (non-flat) model so that CTC label loops are added
    builder_ = std::make_unique<Speech::CTCTopologyGraphBuilder>(
            select("allophone-state-graph-builder"), lexicon_, acousticModel_, false);
}

// RASR has no built-in FSA equivalence (difference()/complement() are unreliable
// for these graphs), so test it directly: two deterministic acceptors accept the
// same language iff a synchronized traversal of their state pairs, completing
// missing transitions with a dead state, never disagrees on finality.
static bool equivalentAcceptors(Fsa::ConstAutomatonRef a, Fsa::ConstAutomatonRef b) {
    auto canonical = [](Fsa::ConstAutomatonRef f) {
        return Fsa::staticCopy(Fsa::determinize(Fsa::removeEpsilons(f)));
    };
    Core::Ref<Fsa::StaticAutomaton> sa = canonical(a);
    Core::Ref<Fsa::StaticAutomaton> sb = canonical(b);

    auto isFinal = [](const Core::Ref<Fsa::StaticAutomaton>& s, Fsa::StateId st) -> bool {
        return st != Fsa::InvalidStateId && s->fastState(st)->isFinal();
    };
    auto target = [](const Core::Ref<Fsa::StaticAutomaton>& s, Fsa::StateId st, Fsa::LabelId label) -> Fsa::StateId {
        if (st == Fsa::InvalidStateId) {
            return Fsa::InvalidStateId;  // dead state stays dead
        }
        const Fsa::State* state = s->fastState(st);
        for (u32 i = 0; i < state->nArcs(); ++i) {
            if (state->getArc(i)->input_ == label) {
                return state->getArc(i)->target_;  // deterministic: first match is the only one
            }
        }
        return Fsa::InvalidStateId;
    };
    auto labelsAt = [](const Core::Ref<Fsa::StaticAutomaton>& s, Fsa::StateId st, std::set<Fsa::LabelId>& out) {
        if (st == Fsa::InvalidStateId) {
            return;
        }
        const Fsa::State* state = s->fastState(st);
        for (u32 i = 0; i < state->nArcs(); ++i) {
            out.insert(state->getArc(i)->input_);
        }
    };

    std::set<std::pair<Fsa::StateId, Fsa::StateId>>   visited;
    std::deque<std::pair<Fsa::StateId, Fsa::StateId>> queue;
    queue.emplace_back(sa->initialStateId(), sb->initialStateId());
    while (!queue.empty()) {
        std::pair<Fsa::StateId, Fsa::StateId> pair = queue.front();
        queue.pop_front();
        if (visited.count(pair) > 0) {
            continue;
        }
        visited.insert(pair);
        if (isFinal(sa, pair.first) != isFinal(sb, pair.second)) {
            return false;
        }
        std::set<Fsa::LabelId> labels;
        labelsAt(sa, pair.first, labels);
        labelsAt(sb, pair.second, labels);
        for (Fsa::LabelId label : labels) {
            queue.emplace_back(target(sa, pair.first, label), target(sb, pair.second, label));
        }
    }
    return true;
}

TEST_F(Speech, CTCGraphBuilderTest, BuildSimpleSentence) {
    // orthography must be whitespace-normalized with a trailing blank
    Speech::AllophoneStateGraphRef graph = builder_->build(std::string("A B C "));
    EXPECT_FALSE(Fsa::isEmpty(graph));

    // The reference graph on disk was generated from this very builder
    // and manually checked for corectness.
    Fsa::ConstAutomatonRef reference = Fsa::read("data/allophone_state_graph_builder/ctc_abc.fsa.xml");
    EXPECT_TRUE(static_cast<bool>(reference));
    EXPECT_FALSE(Fsa::isEmpty(reference));

    EXPECT_TRUE(equivalentAcceptors(graph, reference));

    // sanity check that the comparison actually discriminates: a different
    // sentence must not match the stored "A B C" reference
    Speech::AllophoneStateGraphRef other = builder_->build(std::string("A B "));
    EXPECT_FALSE(equivalentAcceptors(other, reference));
}
