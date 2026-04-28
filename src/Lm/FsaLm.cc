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
#include "FsaLm.hh"
#include <Bliss/Fsa.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Input.hh>
#include <Fsa/Sort.hh>

using namespace Lm;

const Core::ParameterString FsaLm::paramFilename("file", "name of fsa file file to load as language model");

const Core::ParameterBool FsaLm::paramGarbageLoopMode(
        "garbage-loop-mode",
        "accept garbage input (inf score) at any state towards the initial state, and allow looping over the fsa to accept multiple valid phrases in one utterance (final to initial)",
        false);

const Core::ParameterBool FsaLm::paramAcceptPartialRepeat(
        "accept-partial-repeat",
        "only under garbage-loop-mode: additionally accept repeating partial begin phrases",
        false);

Fsa::ConstStateRef FsaLm::invalidHistory(new Fsa::State(Fsa::InvalidStateId, Fsa::StateTagFinal));

class FsaLm::HistoryManager : public ReferenceCountingHistoryManager {
protected:
    virtual HistoryHash hashKey(HistoryHandle hd) const {
        const Fsa::State* state = (const Fsa::State*)hd;
        return state->id();
    }

    virtual bool isEquivalent(HistoryHandle hda, HistoryHandle hdb) const {
        const Fsa::State* sa = (const Fsa::State*)hda;
        const Fsa::State* sb = (const Fsa::State*)hdb;
        return sa->id() == sb->id();
    }

    virtual std::string format(HistoryHandle hd) const {
        const Fsa::State* state = (const Fsa::State*)hd;
        return std::to_string(state->id());
    }
};

FsaLm::FsaLm(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Core::Component(c),
          LanguageModel(c, lexicon),
          infinityScore_(1e9),
          garbageLoopMode_(paramGarbageLoopMode(c)),
          syntacticTokens_(lexicon->syntacticTokenAlphabet()) {
    historyManager_      = new HistoryManager;
    acceptPartialRepeat_ = garbageLoopMode_ && paramAcceptPartialRepeat(c);
    if (garbageLoopMode_) {
        log() << "accept garbage and loop over FSA mode";
    }
    if (acceptPartialRepeat_) {
        log() << "additionally accept repeating partial begin phrases";
    }
}

FsaLm::~FsaLm() {
    if (historyManager_) {
        delete historyManager_;
        historyManager_ = nullptr;
    }
}

void FsaLm::load() {
    const std::string filename(paramFilename(config));
    log("reading fsa as language model from file")
            << " \"" << filename << "\" ...";

    Fsa::StorageAutomaton* f = new Fsa::StaticAutomaton();
    if (!Fsa::read(f, filename)) {
        error("Failed to read FSA from file.");
        delete f;
        return;
    }
    setFsa(Fsa::ConstAutomatonRef(f));
}

void FsaLm::setFsa(Fsa::ConstAutomatonRef f) {
    fsa_ = Fsa::cache(Fsa::sort(Fsa::determinize(Fsa::mapInput(f, syntacticTokens_)), Fsa::SortTypeByInput));
}

History FsaLm::startHistory() const {
    Fsa::ConstStateRef sp = initialState();
    sp->acquireReference();
    return history(sp.get());
}

History FsaLm::extendedHistory(const History& h, Token w) const {
    Fsa::ConstStateRef sp(descriptor<Self>(h));
    if (sp != invalidHistory) {
        sp = nextState(sp, w);
    }

    sp->acquireReference();
    return history(sp.get());
}

Lm::Score FsaLm::score(const History& h, Token w) const {
    Fsa::ConstStateRef sp(descriptor<Self>(h));
    return stateScore(sp, w);
}

Lm::Score FsaLm::sentenceEndScore(const History& h) const {
    Fsa::ConstStateRef sp(descriptor<Self>(h));
    return stateSentenceEndScore(sp);
}

HistorySuccessors FsaLm::getHistorySuccessors(const History& h) const {
    Fsa::ConstStateRef sp(descriptor<Self>(h));
    return getStateSuccessors(sp);
}

Fsa::ConstStateRef FsaLm::initialState() const {
    Fsa::StateId initial = fsa_->initialStateId();
    if (initial == Fsa::InvalidStateId) {
        error("language model fsa does not have an initial state");
    }
    return fsa_->getState(initial);
}

Fsa::ConstStateRef FsaLm::nextState(Fsa::ConstStateRef sp, Token w) const {
    Fsa::ConstStateRef initial = initialState();
    bool               repeat  = acceptPartialRepeat_ && sp != initial;
    // fsa may contain direct EPS path from initial to final
    // reset completed path only once to avoid endless loop
    bool resetFinal = sp != initial;
    while (sp) {
        Fsa::Arc tmp;
        tmp.input_                   = w->id();
        Fsa::State::const_iterator a = sp->lower_bound(tmp, Fsa::byInput());
        if ((a == sp->end()) || (a->input() != w->id())) {
            a = sp->begin();
            if ((a == sp->end()) || (a->input() != Fsa::Epsilon)) {
                if (garbageLoopMode_) {
                    if ((sp->isFinal() && resetFinal) || repeat) {
                        sp         = initial;
                        repeat     = false;
                        resetFinal = false;
                        continue;
                    }
                    else  // garbage state: initial
                        return initial;
                }
                else {
                    return invalidHistory;
                }
            }
            sp = fsa_->getState(a->target());
        }
        else {
            return fsa_->getState(a->target());
        }
    }
    return sp;
}

Score FsaLm::stateScore(Fsa::ConstStateRef sp, Token w) const {
    if (w == sentenceEndToken()) {
        return stateSentenceEndScore(sp);
    }
    if (sp == invalidHistory) {
        return infinityScore();
    }

    Fsa::ConstStateRef initial    = fsa_->getState(fsa_->initialStateId());
    bool               repeat     = acceptPartialRepeat_ && sp != initial;
    bool               resetFinal = sp != initial;  // final to initial only once: avoid endless loop
    Score              score      = 0.0;
    while (sp) {
        Fsa::Arc tmp;
        tmp.input_ = w->id();
        // search successors for input w
        Fsa::State::const_iterator a = sp->lower_bound(tmp, Fsa::byInput());
        if ((a == sp->end()) || (a->input() != w->id())) {
            // not found: either none or still exist an EPS arc (first one)
            a = sp->begin();
            if ((a == sp->end()) || (a->input() != Fsa::Epsilon)) {
                // dead end: no matching successor nor EPS arc to check further
                if (garbageLoopMode_) {
                    // reset to initial for final state (complete and may start over)
                    if (sp->isFinal()) {
                        score += Score(sp->weight_);
                    }
                    if ((sp->isFinal() && resetFinal) || repeat) {
                        // either reset final or repeat partial: check again from initial
                        sp         = initial;
                        repeat     = false;
                        resetFinal = false;
                        continue;
                    }
                    else {  // otherwise infScore for dead end
                        return infinityScore();
                    }
                }
                else {  // infScore for dead end if not garbageLoopMode
                    return infinityScore();
                }
            }
            // proceed along the EPS arc to check further
            sp = fsa_->getState(a->target());
            score += Score(a->weight());
        }
        else {  // found successor: return score
            return score + Score(a->weight());
        }
    }
    return infinityScore();
}

Score FsaLm::stateSentenceEndScore(Fsa::ConstStateRef sp) const {
    if (sp == invalidHistory) {
        return infinityScore();
    }
    Score score = 0.0;
    while (sp) {
        if (sp->isFinal()) {
            return score + Score(sp->weight_);
        }
        Fsa::State::const_iterator a = sp->begin();
        if ((a == sp->end()) || (a->input() != Fsa::Epsilon)) {
            return infinityScore();
        }
        sp = fsa_->getState(a->target());
        score += Score(a->weight());
    }
    return infinityScore();
}

HistorySuccessors FsaLm::getStateSuccessors(Fsa::ConstStateRef sp) const {
    HistorySuccessors res;
    res.backOffScore = infinityScore();

    Fsa::ConstStateRef initial    = fsa_->getState(fsa_->initialStateId());
    bool               repeat     = acceptPartialRepeat_ && sp != initial;
    bool               resetFinal = sp != initial;  // final to initial only once: avoid endless loop
    Score              score      = 0.0;

    while (sp and sp != invalidHistory) {
        for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
            if (a->input() != Fsa::Epsilon) {
                res.emplace_back(a->input(), score + Score(a->weight()));
            }
        }

        Fsa::State::const_iterator a = sp->begin();
        if ((a == sp->end()) || (a->input() != Fsa::Epsilon)) {
            if (garbageLoopMode_) {
                if (sp->isFinal()) {  // complete and may start over
                    score += Score(sp->weight_);
                }
                if ((sp->isFinal() && resetFinal) || repeat) {
                    sp         = initial;
                    repeat     = false;
                    resetFinal = false;
                    continue;
                }
                else {
                    break;
                }
            }
            else {
                break;
            }
        }
        else {
            sp = fsa_->getState(a->target());
            score += Score(a->weight());
        }
    }

    return res;
}
