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
#include "SearchSpaceHelpers.hh"

namespace Search {
Instance::~Instance() {
    if (backOffInstance) {
        verify(backOffInstance->backOffParent == this);
        backOffInstance->backOffParent = 0;
    }
    if (backOffParent) {
        verify(backOffParent->backOffInstance == this);
        backOffParent->backOffInstance = 0;
    }
}

void Instance::enter(TraceManager &trace_manager, Core::Ref<Trace> trace, StateId entryNode, Score score) {
    rootStateHypotheses.push_back(StateHypothesis(entryNode, trace_manager.getTrace(TraceItem(trace, key.history, lookaheadHistory, scoreHistory)), score));
}

u32 Instance::backOffChainStates() const {
    const Instance* tree = this;
    while (tree->backOffParent)
        tree = tree->backOffParent;

    u32 states = 0;

    while (tree) {
        states += tree->states.size();
        tree = tree->backOffInstance;
    }
    return states;
}

void Instance::addLmScore(EarlyWordEndHypothesis&                         hyp,
                          Bliss::LemmaPronunciation::Id                   pron,
                          const Core::Ref<const Lm::ScaledLanguageModel>& lm,
                          const Bliss::LexiconRef&                        lexicon,
                          Score                                           wpScale) const {
    SimpleLMCache::const_iterator it = lmCache.find(pron);
    if (it == lmCache.end()) {
        Score oldLmScore = hyp.score.lm;
        if (pron != Bliss::LemmaPronunciation::invalidId) {
            Lm::addLemmaPronunciationScoreOmitExtension(lm, lexicon->lemmaPronunciation(pron), wpScale, lm->scale(), scoreHistory, hyp.score.lm);
        }
        lmCache.insert(std::make_pair(pron, hyp.score.lm - oldLmScore));
    }
    else {
        hyp.score.lm += (*it).second;
    }
}

void Instance::addLmScore(WordEndHypothesis&                              hyp,
                          Bliss::LemmaPronunciation::Id                   pron,
                          const Core::Ref<const Lm::ScaledLanguageModel>& lm,
                          const Bliss::LexiconRef&                        lexicon,
                          Score                                           wpScale) const {
    SimpleLMCache::const_iterator it = lmCache.find(pron);
    if (it == lmCache.end()) {
        Score oldLmScore = hyp.score.lm;
        if (pron != Bliss::LemmaPronunciation::invalidId) {
            Lm::addLemmaPronunciationScoreOmitExtension(lm, lexicon->lemmaPronunciation(pron), wpScale, lm->scale(), scoreHistory, hyp.score.lm);
        }
        lmCache.insert(std::make_pair(pron, hyp.score.lm - oldLmScore));
    }
    else {
        hyp.score.lm += (*it).second;
    }
}

int WordEndHypothesis::meshHistoryPhones = 1;
}  // namespace Search
