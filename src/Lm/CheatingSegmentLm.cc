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
#include "CheatingSegmentLm.hh"

#include <Core/StringUtilities.hh>
#include <Fsa/Output.hh>
#include <Fsa/Static.hh>
#include <Fsa/tDraw.hh>

#include "HistoryManager.hh"

namespace Lm {

class CheatingSegmentLm::HistoryManager : public ReferenceCountingHistoryManager {
protected:
    virtual HistoryHash hashKey(HistoryHandle hd) const {
        CheatingHistory const* h = reinterpret_cast<CheatingHistory const*>(hd);
        return h->seq_idx << 32 xor h->fsa_state->id();
    }

    virtual bool isEquivalent(HistoryHandle hda, HistoryHandle hdb) const {
        CheatingHistory const* ha = reinterpret_cast<CheatingHistory const*>(hda);
        CheatingHistory const* hb = reinterpret_cast<CheatingHistory const*>(hdb);
        return ha->seq_idx == hb->seq_idx and ha->fsa_state->id() == hb->fsa_state->id();
    }

    virtual std::string format(HistoryHandle hd) const {
        CheatingHistory const* h = reinterpret_cast<CheatingHistory const*>(hd);
        return std::to_string(h->seq_idx) + "-" + std::to_string(h->fsa_state->id());
    }
};

const Core::ParameterFloat CheatingSegmentLm::paramInfinityScore("infinity-score", "score to use for incorrect words", 1e9);

CheatingSegmentLm::CheatingSegmentLm(Core::Configuration const& c, Bliss::LexiconRef l)
        : Core::Component(c),
          Precursor(c, l),
          segmentIdx_(0ul),
          lexicon_(l),
          orthParser_(Core::ref(new Bliss::OrthographicParser(config, lexicon_))) {
    infinityScore_ = paramInfinityScore(config);
    if (historyManager_) {
        delete historyManager_;
        historyManager_ = nullptr;
    }
    historyManager_ = new HistoryManager();
}

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
    segmentIdx_ += 1ul;
}

bool CheatingSegmentLm::setSegment(Bliss::SpeechSegment const* s) {
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
    segmentIdx_ += 1ul;

    return true;
}

History CheatingSegmentLm::startHistory() const {
    Core::Ref<CheatingHistory> ch(new CheatingHistory());
    ch->seq_idx   = segmentIdx_;
    ch->fsa_state = initialState();

    ch->acquireReference();
    return history(ch.get());
}

History CheatingSegmentLm::extendedHistory(History const& h, Token w) const {
    Core::Ref<const CheatingHistory> prev(descriptor<CheatingSegmentLm>(h));

    Core::Ref<CheatingHistory> ch(new CheatingHistory());
    ch->seq_idx   = segmentIdx_;
    ch->fsa_state = nextState(prev->fsa_state, w);

    ch->acquireReference();
    return history(ch.get());
}

Score CheatingSegmentLm::score(History const& h, Token w) const {
    Core::Ref<const CheatingHistory> hist(descriptor<CheatingSegmentLm>(h));
    return stateScore(hist->fsa_state, w);
}

Score CheatingSegmentLm::sentenceEndScore(History const& h) const {
    Core::Ref<const CheatingHistory> hist(descriptor<CheatingSegmentLm>(h));
    return stateSentenceEndScore(hist->fsa_state);
}

HistorySuccessors CheatingSegmentLm::getHistorySuccessors(History const& h) const {
    Core::Ref<const CheatingHistory> hist(descriptor<CheatingSegmentLm>(h));
    return getStateSuccessors(hist->fsa_state);
}

}  // namespace Lm
