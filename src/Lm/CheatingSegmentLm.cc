#include "CheatingSegmentLm.hh"

#include <Core/StringUtilities.hh>
#include <Fsa/Output.hh>
#include <Fsa/Static.hh>

namespace Lm {

const Core::ParameterFloat CheatingSegmentLm::paramInfinityScore("infinity-score", "score to use for incorrect words", 1e9);

void CheatingSegmentLm::load() {
    // create Empty LM
    Fsa::StaticAutomaton* s = new Fsa::StaticAutomaton();
    s->setType(Fsa::TypeAcceptor);
    s->setInputAlphabet(lexicon_->syntacticTokenAlphabet());
    s->setSemiring(Fsa::TropicalSemiring);
    s->setDescription("empty-lm");
    s->setInitialStateId(s->newFinalState(Fsa::TropicalSemiring->one())->id());
    Fsa::ConstAutomatonRef f = Fsa::ConstAutomatonRef(s);
    this->setFsa(f);
}

void CheatingSegmentLm::setSegment(Bliss::SpeechSegment const* s) {
    std::string orth = s->orth();
    Core::normalizeWhitespace(orth);
    Core::enforceTrailingBlank(orth);

    Core::Ref<Bliss::LemmaAcceptor> orth_automaton = orthParser_->createLemmaAcceptor(orth);  // automaton using orth-alphabet
    auto                            alphabet       = orth_automaton->inputAlphabet();

    // build automaton using syntactic-token alphabet
    Fsa::StaticAutomaton* synt_automaton = new Fsa::StaticAutomaton();
    synt_automaton->setType(Fsa::TypeAcceptor);
    synt_automaton->setInputAlphabet(lexicon_->syntacticTokenAlphabet());
    synt_automaton->setSemiring(Fsa::TropicalSemiring);
    synt_automaton->setDescription(std::string("cheatingLm(") + s->fullName() + std::string(")"));

    std::vector<Fsa::StateId> id_map;
    id_map.reserve(orth_automaton->size());
    for (Fsa::StateId sid = 0u; sid <= orth_automaton->maxStateId(); sid++) {
        if (orth_automaton->hasState(sid)) {
            auto* orth_state = orth_automaton->fastState(sid);
            auto* synt_state = synt_automaton->newState(orth_state->tags(), orth_state->weight());
            id_map.push_back(synt_state->id());
        }
        else {
            id_map.push_back(0u);
        }
    }
    synt_automaton->setInitialStateId(id_map[orth_automaton->initialStateId()]);
    for (Fsa::StateId sid = 0u; sid <= orth_automaton->maxStateId(); sid++) {
        if (!orth_automaton->hasState(sid)) {
            continue;
        }
        auto* orth_state = orth_automaton->fastState(sid);
        for (auto arc = orth_state->begin(); arc != orth_state->end(); ++arc) {
            Bliss::Lemma const* lemma = lexicon_->lemma(alphabet->symbol(arc->input()));
            require(lemma != nullptr);
            if (lexicon_->specialLemma("silence") != lemma and lemma->hasSyntacticTokenSequence()) {
                auto& tokens = lemma->syntacticTokenSequence();
                if (tokens.size() > 0ul) {
                    Fsa::StateId source_id = id_map[sid];
                    for (size_t idx = 0ul; idx + 1 < tokens.size(); idx++) {
                        auto* source_state = synt_automaton->fastState(source_id);
                        auto* next_state   = synt_automaton->newState();
                        source_state->newArc(next_state->id(), Fsa::StaticAutomaton::Weight(), tokens[idx]->id());
                        source_id = next_state->id();
                    }
                    auto* synt_state = synt_automaton->fastState(source_id);
                    synt_state->newArc(id_map[arc->target()], arc->weight(), tokens[tokens.size() - 1ul]->id());
                }
            }
        }
    }
    auto synt_ref = Fsa::ConstAutomatonRef(synt_automaton);
    this->setFsa(synt_ref);
}

}  // namespace Lm
