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
#include "FFNeuralNetworkLanguageModel.hh"

#include <fstream>
#include <string>

#include <Core/Assertions.hh>
#include <Core/Channel.hh>
#include <Math/Vector.hh>
#include "NNHistoryManager.hh"

namespace {
    struct ScoreCache : public Lm::NNCacheWithStats {
        Math::FastVector<Lm::Score> scores;
    };
}

namespace Lm {

Core::ParameterBool FFNeuralNetworkLanguageModel::paramExpandOneHot(
        "expand-one-hot", "wether to create a dense one-hot vector", false);
Core::ParameterBool FFNeuralNetworkLanguageModel::paramEagerForwarding(
        "eager-forwarding", "wether to forward histories eagerly in batches mode", true);
Core::ParameterInt FFNeuralNetworkLanguageModel::paramContextSize(
        "context-size", "context size (number of words passed to LM)", 0);
Core::ParameterInt FFNeuralNetworkLanguageModel::paramHistorySize(
        "history-size", "history size (length of history, has to be >= context-size)", 0);
Core::ParameterInt FFNeuralNetworkLanguageModel::paramBufferSize(
        "buffer-size", "buffer size", 32);

FFNeuralNetworkLanguageModel::FFNeuralNetworkLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
                            : Core::Component(c), FFNeuralNetworkLanguageModel::Precursor(c, l),
                              expand_one_hot_(paramExpandOneHot(c)), eager_forwarding_(paramEagerForwarding(c)), context_size_(paramContextSize(c)),
                              history_size_(std::max<int>(paramHistorySize(c), context_size_)), buffer_size_(paramBufferSize(c)),
                              nn_(select("nn")) {
}

FFNeuralNetworkLanguageModel::~FFNeuralNetworkLanguageModel() {
    delete historyManager_;
    nn_.finalize();
}

History FFNeuralNetworkLanguageModel::startHistory() const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    TokenIdSequence ts(history_size_, lexicon_mapping_[sentenceBeginToken()->id()]);
    return history(hm->get<ScoreCache>(ts));
}

History FFNeuralNetworkLanguageModel::extendedHistory(History const& hist, Token w) const {
    NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
    ScoreCache const* sc = reinterpret_cast<ScoreCache const*>(hist.handle());
    TokenIdSequence ts(history_size_);
    std::copy(sc->history->begin(), sc->history->end() - 1, ts.begin() + 1);
    ts.front() = lexicon_mapping_[w->id()];
    return history(hm->get<ScoreCache>(ts));
}

Score FFNeuralNetworkLanguageModel::score(History const& hist, Token w) const {
    ScoreCache const* sc = reinterpret_cast<ScoreCache const*>(hist.handle());
    size_t output_idx = lexicon_mapping_[w->id()];
    useOutput(*sc, output_idx);
    if (not sc->scores.empty()) {
        return sc->scores[output_idx];
    }

    std::vector<ScoreCache*> caches;
    if (eager_forwarding_) {
        NNHistoryManager* hm = dynamic_cast<NNHistoryManager*>(historyManager_);
        NNHistoryManager::NNCacheMap const& cm = hm->getNNCacheMap();
        for (auto const& kv : cm) {
            ScoreCache* c = const_cast<ScoreCache*>(static_cast<ScoreCache const*>(kv.second));
            if (c->scores.empty()) {
                caches.push_back(c);
            }
        }
    }
    else {
        caches.push_back(const_cast<ScoreCache*>(sc));
    }

    size_t vec_size = expand_one_hot_ ? context_size_ * num_outputs_ : context_size_;
    Nn::Types<f32>::NnMatrix input(vec_size, caches.size());
    input.setToZero();
    for (size_t c = 0ul; c < caches.size(); c++) {
        for (size_t t = 0ul; t < context_size_; t++) {
            if (expand_one_hot_) {
                input.at(t * num_outputs_ + caches[c]->history->at(t), c) = 1.0f;
            }
            else {
                input.at(t, c) = caches[c]->history->at(t);
            }
        }
    }
    nn_.forward(input);

    Nn::Types<f32>::NnMatrix& gpu_output = nn_.getTopLayerOutput();;
    gpu_output.finishComputation();
    Math::FastMatrix<f32>& cpu_output = gpu_output.asWritableCpuMatrix();
    require_eq(cpu_output.nRows(), num_outputs_);

    for (size_t c = 0ul; c < caches.size(); c++) {
        cpu_output.getColumn(c, caches[c]->scores);
        for (f32& s : caches[c]->scores) {
            s = -std::log(s);
        }
    }
    gpu_output.initComputation();
    require_eq(sc->scores.size(), num_outputs_);

    return sc->scores[output_idx];
}

void FFNeuralNetworkLanguageModel::load() {
    loadVocabulary();
    std::vector<u32> stream_sizes;
    if (expand_one_hot_) {
        stream_sizes.push_back(num_outputs_ * context_size_);
    }
    else {
        stream_sizes.push_back(context_size_);
    }
    nn_.initializeNetwork(buffer_size_, stream_sizes);
}

} // namespace Lm
