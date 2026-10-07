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

#include "WordAssembler.hh"

namespace Search {

const Core::Choice WordAssembler::choiceConvention(
        "word-start-marked", WordStartMarked,
        "continuation-marked", ContinuationMarked,
        Core::Choice::endMark());

const Core::ParameterChoice WordAssembler::paramConvention(
        "word-piece-convention",
        &choiceConvention,
        "How word-piece labels mark word boundaries.",
        WordStartMarked);

const Core::ParameterString WordAssembler::paramMarker(
        "word-piece-marker",
        "Word boundary marker of the word pieces. Default: \"\xE2\x96\x81\" (U+2581) for word-start-marked, \"@@\" for continuation-marked.",
        "");

WordAssembler::WordAssembler(Core::Configuration const& config)
        : Core::Component(config),
          convention_(static_cast<Convention>(paramConvention(config))),
          marker_(paramMarker(config)) {
    if (marker_.empty()) {
        marker_ = convention_ == WordStartMarked ? "\xE2\x96\x81" : "@@";
    }
    intern("");
}

void WordAssembler::setLexicon(Bliss::LexiconRef lexicon) {
    pieces_.assign(lexicon->nLemmas(), Piece());
    steps_.clear();
    for (auto lemmas = lexicon->lemmas(); lemmas.first != lemmas.second; ++lemmas.first) {
        Bliss::Lemma const* lemma = *lemmas.first;
        if (lemma->nOrthographicForms() == 0) {
            continue;
        }
        Piece&      piece = pieces_[lemma->id()];
        std::string text  = lemma->preferredOrthographicForm().str();
        if (convention_ == WordStartMarked) {
            piece.marked = text.compare(0, marker_.size(), marker_) == 0;
            piece.text   = piece.marked ? text.substr(marker_.size()) : text;
        }
        else {
            piece.marked = text.size() >= marker_.size() and text.compare(text.size() - marker_.size(), marker_.size(), marker_) == 0;
            piece.text   = piece.marked ? text.substr(0, text.size() - marker_.size()) : text;
        }
    }
}

WordAssembler::Step WordAssembler::extend(WordId pending, Nn::LabelIndex label) {
    u64  key = (static_cast<u64>(pending) << 32) | label;
    auto it  = steps_.find(key);
    if (it == steps_.end()) {
        it = steps_.emplace(key, computeStep(pending, pieces_.at(label))).first;
    }
    return it->second;
}

WordAssembler::Step WordAssembler::computeStep(WordId pending, Piece const& piece) {
    if (convention_ == WordStartMarked) {
        if (piece.marked and pending != noWord) {
            return {pending, intern(piece.text)};
        }
        return {noWord, intern(spelling(pending) + piece.text)};
    }
    WordId word = intern(spelling(pending) + piece.text);
    return piece.marked ? Step{noWord, word} : Step{word, noWord};
}

WordAssembler::WordId WordAssembler::intern(std::string const& spelling) {
    auto [it, inserted] = wordIds_.try_emplace(spelling, static_cast<WordId>(spellings_.size()));
    if (inserted) {
        spellings_.push_back(&it->first);
    }
    return it->second;
}

}  // namespace Search
