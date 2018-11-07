/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "AbstractNNLanguageModel.hh"

#include <functional>
#include <numeric>

namespace Lm {

Core::ParameterBool   AbstractNNLanguageModel::paramCollectStatistics(
        "collect-statistics", "wether to collect runtime statistics", false);
Core::ParameterString AbstractNNLanguageModel::paramVocabularyFile(
        "vocab-file", "vocabulary file", "");
Core::ParameterString AbstractNNLanguageModel::paramVocabUnknownWord(
        "vocab-unknown-word", "the word from the provided vocabulary file that will serve as unknown token", "");

AbstractNNLanguageModel::AbstractNNLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
                        : Core::Component(c), Precursor(c, l),
                          collect_statistics_(paramCollectStatistics(c)), vocab_file_(paramVocabularyFile(c)), unknown_word_(paramVocabUnknownWord(config)),
                          lexicon_(l), num_outputs_(0ul), lexicon_mapping_(), usage_histogram_() {
    NNHistoryManager* hm = new NNHistoryManager();
    hm->setOnReleaseHandler(std::bind(&AbstractNNLanguageModel::onRelease, this, std::placeholders::_1));
    historyManager_ = hm;
}

AbstractNNLanguageModel::~AbstractNNLanguageModel() {
    if (collect_statistics_) {
        logStatistics();
    }
}

void AbstractNNLanguageModel::loadVocabulary() {
    std::unordered_map<std::string, size_t> vocab_map;

    std::ifstream input(vocab_file_, std::ios::in);
    std::string line;
    while (input.good()) {
        std::getline(input, line);
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string word;
        size_t idx;
        ss >> word;
        ss >> idx;
        vocab_map[word] = idx;
        num_outputs_ = std::max(num_outputs_, idx);
    }
    num_outputs_ += 1ul;  // largest id + 1

    size_t unknown_word_id = 0ul;
    auto unk = vocab_map.find(unknown_word_);
    if (unk != vocab_map.end()) {
        unknown_word_id = unk->second;
        log("unknown word: ") << unknown_word_ << " " << unknown_word_id;
    }
    else if (not unknown_word_.empty()) {
        warning("could not find unknown word ") << unknown_word_ << " in vocabulary";
    }

    lexicon_mapping_.resize(lexicon_->nSyntacticTokens());
    auto iters = lexicon_->syntacticTokens();
    for (; iters.first != iters.second; ++iters.first) {
        auto vm_iter = vocab_map.find((*iters.first)->symbol());
        if (vm_iter != vocab_map.end()) {
            lexicon_mapping_[(*iters.first)->id()] = vm_iter->second;
        }
        else {
            warning("did not find: ") << (*iters.first)->symbol() << " using output " << unknown_word_id;
            lexicon_mapping_[(*iters.first)->id()] = unknown_word_id;
        }
    }
}

void AbstractNNLanguageModel::onRelease(HistoryHandle handle) {
    NNCacheWithStats const* c = reinterpret_cast<NNCacheWithStats const*>(handle);
    if (not c->output_used.empty()) {
        unsigned used_outputs = std::accumulate(c->output_used.begin(), c->output_used.end(),
                                                0u, [](unsigned sum, bool used){ return sum + (used ? 1u : 0u); });
        size_t promille_used = static_cast<size_t>((1000.0 * used_outputs) / c->output_used.size());
        if (usage_histogram_.size() <= promille_used) {
            usage_histogram_.resize(promille_used + 1ul);
        }
        usage_histogram_[promille_used] += 1u;
    }
}

void AbstractNNLanguageModel::logStatistics() const {
    Core::XmlChannel out(config, "statistics");
    out << Core::XmlOpen("lm-usage-histogram")
           + Core::XmlAttribute("size", usage_histogram_.size());
    for (u32 h : usage_histogram_) {
        out << " " << h;
    }
    out << Core::XmlClose("lm-usage-histogram");
}

} // namespace Lm

