/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "OnnxStatelessLanguageModel.hh"

namespace Lm {

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "tokens",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -1}}},
        Onnx::IOSpecification{
                "lengths",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};

const Core::ParameterInt paramMaxBatchSize(
        "max-batch-size",
        "Maximum number of histories forwarded in one go",
        64, 1);

OnnxStatelessLm::OnnxStatelessLm(const Core::Configuration& c, Bliss::LexiconRef l)
        : Core::Component(c),
          Precursor(c, l),
          onnxModel_(select("onnx-model"), ioSpec),
          inputTokensName_(onnxModel_.mapping.getOnnxName("tokens")),
          inputLengthsName_(onnxModel_.mapping.getOnnxName("lengths")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          maxBatchSize_(paramMaxBatchSize(config)),
          batchQueue_(),
          batch_(),
          startHistory_() {
}

void OnnxStatelessLm::load() {
    loadVocabulary();
    startHistory_ = startHistory();
}

History OnnxStatelessLm::startHistory() const {
    // Assume that whenever a startHistory is requested, everything in the batchQueue is no longer needed
    batchQueue_.clear();

    if (startHistory_.isValid()) {
        return startHistory_;
    }

    auto            sentBeginId = lexicon_mapping_.at(sentenceBeginToken()->id());
    TokenIdSequence tokenSequence(1ul, sentBeginId);
    log() << "Initialize LM history with sentence begin token " << sentBeginId;

    auto historyManager = dynamic_cast<NNHistoryManager*>(historyManager_);
    auto handle         = historyManager->get<HistoryDescriptor>(tokenSequence);
    auto hist           = history(handle);
    batchQueue_.push_back(hist);
    return hist;
}

History OnnxStatelessLm::extendedHistory(History const& hist, Token nextToken) const {
    auto tokenId = lexicon_mapping_.at(nextToken->id());

    auto historyManager = dynamic_cast<NNHistoryManager*>(historyManager_);
    auto descriptor     = reinterpret_cast<HistoryDescriptor const*>(hist.handle());

    TokenIdSequence newTokens(*descriptor->history);
    newTokens.push_back(tokenId);

    auto extHandle = historyManager->get<HistoryDescriptor>(newTokens);

    auto extHist = history(extHandle);
    batchQueue_.push_back(extHist);
    return extHist;
}

Score OnnxStatelessLm::score(History const& hist, Token nextToken) const {
    size_t tokenId = lexicon_mapping_.at(nextToken->id());

    auto descriptor = static_cast<HistoryDescriptor const*>(hist.handle());

    if (descriptor->scores.empty()) {
        makeBatch(hist);
        scoreBatch();
        batch_.clear();
    }
    verify(not descriptor->scores.empty());
    return descriptor->scores[tokenId];
}

void OnnxStatelessLm::makeBatch(History const& hist) const {
    std::unordered_set<TokenIdSequence const*, TokenIdSequencePtrHash, TokenIdSequencePtrEq> seenHistories;

    batch_.push_back(hist);
    seenHistories.insert(static_cast<HistoryDescriptor const*>(hist.handle())->history.get());

    while (batch_.size() < maxBatchSize_ and not batchQueue_.empty()) {
        auto        queuedHistory    = batchQueue_.front();
        auto const* queuedDescriptor = static_cast<HistoryDescriptor const*>(queuedHistory.handle());
        auto const* queuedTokenSeq   = queuedDescriptor->history.get();
        batchQueue_.pop_front();

        if (seenHistories.find(queuedTokenSeq) == seenHistories.end() and queuedDescriptor->scores.empty()) {
            batch_.push_back(queuedHistory);
            seenHistories.insert(queuedTokenSeq);
        }
    }
}

void OnnxStatelessLm::scoreBatch() const {
    if (batch_.empty()) {
        return;
    }
    std::vector<HistoryDescriptor*> descriptors;
    descriptors.reserve(batch_.size());
    for (auto const& hist : batch_) {
        descriptors.push_back(const_cast<HistoryDescriptor*>(static_cast<HistoryDescriptor const*>(hist.handle())));
    }

    size_t maxLength = 0ul;
    for (auto* descriptor : descriptors) {
        maxLength = std::max(maxLength, descriptor->history->size());
    }

    Math::FastMatrix<s32> tokenMat(maxLength, batch_.size());
    Math::FastVector<s32> lengthVec(batch_.size());

    u32 b = 0ul;
    for (auto* descriptor : descriptors) {
        lengthVec[b] = descriptor->history->size();
        for (u32 n = 0; n < descriptor->history->size(); ++n) {
            tokenMat.at(n, b) = descriptor->history->at(n);
        }
        // zero padding
        for (u32 n = descriptor->history->size(); n < maxLength; ++n) {
            tokenMat.at(n, b) = 0;
        }
        ++b;
    }

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(inputTokensName_, Onnx::Value::create(tokenMat, true));
    sessionInputs.emplace_back(inputLengthsName_, Onnx::Value::create(lengthVec));

    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    Onnx::Value scoreOutput(std::move(sessionOutputs.front()));  // Only one session output

    b = 0ul;
    for (auto* descriptor : descriptors) {
        scoreOutput.get(b, descriptor->scores);
        ++b;
    }
}

}  // namespace Lm
