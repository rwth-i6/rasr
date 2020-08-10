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
#include "PrefixFilter.hh"
#include <Core/Configuration.hh>
#include <iostream>
#include <sstream>
#include <string>
#include "SearchSpace.hh"

namespace Search {
const Core::ParameterString paramPrefixWords(
        "prefix-words",
        "",
        "");

PrefixFilter::PrefixFilter(const PersistentStateTree& tree, Bliss::LexiconRef lexicon, Core::Configuration config)
        : tree_(tree),
          lexicon_(lexicon) {
    setPrefixWords(paramPrefixWords(config));
    prepareReachability();
}

void PrefixFilter::setPrefixWords(std::string prefixWords) {
    std::istringstream inwords(prefixWords);
    while (inwords) {
        std::string word;
        inwords >> word;
        if (word.size()) {
            std::cout << "prefix word '" << word << "'" << std::endl;
            if (lexicon_->lemma(word) == 0) {
                std::cerr << std::endl
                          << "Prefix word '" << word << "' not in lexicon!" << std::endl;
                verify(0);
            }
            else {
                prefixSequence_.push_back(const_cast<Bliss::Lemma*>(lexicon_->lemma(word)));
            }
        }
    }
}

void PrefixFilter::prepareReachability() {
    //Build list of non word lemmas
    for (u32 lemmaId = 0; lemmaId < lexicon_->nLemmas(); ++lemmaId) {
        const Bliss::Lemma* lemma = *(lexicon_->lemmas().first + lemmaId);
        if (!lemma->hasSyntacticTokenSequence() || lemma->syntacticTokenSequence().size() == 0) {
            nonWordLemmas_.insert(lemma);
            if (lemma->nOrthographicForms()) {
            }
        }
    }
    for (std::set<const Bliss::Lemma*>::const_iterator it = nonWordLemmas_.begin(); it != nonWordLemmas_.end(); ++it) {
        for (StateId node = 1; node < tree_.structure.stateCount(); ++node) {
            if (reachable(node, *it))
                nonWordLemmaNodes_.insert(node);
        }
        reachability_.clear();
    }

    for (u32 i = 0; i < prefixSequence_.size(); ++i) {
        std::set<StateId> lemmaNodes;
        for (StateId node = 1; node < tree_.structure.stateCount(); ++node) {
            if (reachable(node, prefixSequence_[i]))
                lemmaNodes.insert(node);
        }
        reachability_.clear();
        prefixReachability_.push_back(lemmaNodes);
    }
}

bool PrefixFilter::reachable(StateId state, const Bliss::Lemma* lemma) {
    if (state < reachability_.size() && reachability_[state] != -1)
        return (bool)reachability_[state];

    if (state >= reachability_.size())
        reachability_.resize(state + 1, -1);

    reachability_[state] = 0;

    for (HMMStateNetwork::SuccessorIterator target = tree_.structure.successors(state); target; ++target) {
        if (!target.isLabel()) {
            if (reachable(*target, lemma)) {
                reachability_[state] = 1;
                return true;
            }
        }
        else {
            if (lexicon_->lemmaPronunciation(tree_.exits[target.label()].pronunciation)->lemma() == lemma) {
                reachability_[state] = 1;
                return true;
            }
        }
    }
    return false;
}

bool PrefixFilter::prune(const TraceManager &trace_manager, const StateHypothesis& hyp) const {
    verify(hyp.trace != invalidTraceId);

    const Search::Trace&       traceItem(*trace_manager.traceItem(hyp.trace).trace);
    std::vector<Bliss::Lemma*> lemmaSequence(0);
    traceItem.getLemmaSequence(lemmaSequence);

    s32 position = 0;
    //Iterate over the lemma sequence of the current trace
    for (int i = 0; i < lemmaSequence.size(); i++) {
        //Skip non word lemmas
        if (nonWordLemmas_.count(lemmaSequence[i])) {
            continue;
        }
        if (position < prefixSequence_.size()) {
            //Trace is invalid if lemma sequence and prefix sequence do not match
            if (lemmaSequence[i] != prefixSequence_[position] && prefixSequence_[position] != 0) {
                return true;
            }
            else {
                position++;
            }
        }
        else {
            //Stop if end of prefix sequence is reached
            return false;
        }
    }

    if (position >= prefixSequence_.size())
        return false;

    //Check reachability of non word lemmas
    if (nonWordLemmaNodes_.count(hyp.state)) {
        return false;
    }

    verify(position < prefixReachability_.size());

    return !(bool)prefixReachability_[position].count(hyp.state);
}
}  // namespace Search
