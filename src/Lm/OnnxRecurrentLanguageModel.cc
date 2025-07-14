#ifndef _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HH
#define _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HH

#include "OnnxRecurrentLanguageModel.hh"

#include "OnnxTransformerStateManager.hh"

#include "OnnxNceSoftmaxAdapter.hh"
#include "OnnxPassthroughSoftmaxAdapter.hh"

namespace {

std::vector<Onnx::IOSpecification> getIOSpec(int64_t num_classes) {
    return std::vector<Onnx::IOSpecification>({
            Onnx::IOSpecification{"word", Onnx::IODirection::INPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::INT32}, {{-1, -1}}},
            Onnx::IOSpecification{"word-length", Onnx::IODirection::INPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::INT32}, {{-1}}},
            Onnx::IOSpecification{"nn-output", Onnx::IODirection::OUTPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::FLOAT}, {{-1, -1, num_classes}}},
    });
}

}  // namespace

namespace Lm {

enum OnnxStateManagerType {
    OnnxTransformerStateManagerType,
    OnnxTransformerStateManager16BitType,
    OnnxTransformerStateManager8BitType,
};

const Core::Choice stateManagerTypeChoice(
        "transformer", OnnxTransformerStateManagerType,
        "transformer-16bit", OnnxTransformerStateManager16BitType,
        "transformer-8bit", OnnxTransformerStateManager8BitType,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice,
        "type of the state manager",
        OnnxTransformerStateManagerType);

std::unique_ptr<OnnxStateManager> createOnnxStateManager(Core::Configuration const& config) {
    OnnxStateManager* res = nullptr;

    switch (stateManagerTypeParam(config)) {
        case OnnxTransformerStateManagerType: res = new Lm::OnnxTransformerStateManager<float>(config); break;
        case OnnxTransformerStateManager16BitType: res = new Lm::OnnxTransformerStateManager<int16_t>(config); break;
        case OnnxTransformerStateManager8BitType: res = new Lm::OnnxTransformerStateManager<int8_t>(config); break;
        default: defect();
    }
    return std::unique_ptr<OnnxStateManager>(res);
}

enum OnnxSoftmaxAdapterType {
    OnnxPassthroughSoftmaxAdapterType,
    OnnxNceSoftmaxAdapterType,
};

const Core::Choice softmaxAdapterTypeChoice(
        "passthrough", OnnxPassthroughSoftmaxAdapterType,
        "nce", OnnxNceSoftmaxAdapterType,
        Core::Choice::endMark());

const Core::ParameterChoice softmaxAdapterTypeParam(
        "type", &softmaxAdapterTypeChoice,
        "type of the softmax adapter",
        OnnxPassthroughSoftmaxAdapterType);

std::unique_ptr<OnnxSoftmaxAdapter> createOnnxSoftmaxAdapter(Core::Configuration const& config) {
    switch (softmaxAdapterTypeParam(config)) {
        case OnnxPassthroughSoftmaxAdapterType: return std::unique_ptr<OnnxSoftmaxAdapter>(new Lm::OnnxPassthroughSoftmaxAdapter(config));
        case OnnxNceSoftmaxAdapterType: return std::unique_ptr<OnnxSoftmaxAdapter>(new Lm::OnnxNceSoftmaxAdapter(config));
        default: defect();
    }
}

OnnxRecurrentLanguageModel::OnnxRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
        : Core::Component(c),
          Precursor(c, l, createOnnxStateManager(select("state-manager"))),
          session_(select("session")),
          ioSpec_(getIOSpec(-2)),
          mapping_(select("io-map"), ioSpec_),
          validator_(select("validator")),
          softmax_adapter_(createOnnxSoftmaxAdapter(select("softmax-adapter"))) {
    state_variables_ = session_.getStateVariablesMetadata();

    setEmptyHistory();

    softmax_adapter_->init(session_, mapping_);
    validator_.validate(ioSpec_, mapping_, session_);
}

void OnnxRecurrentLanguageModel::setState(std::vector<std::pair<std::string, Onnx::Value>> const& inputs, std::vector<std::string> const& targets) const {
}

void OnnxRecurrentLanguageModel::extendInputs(std::vector<std::pair<std::string, Onnx::Value>>& inputs, Math::FastMatrix<s32> const& words, Math::FastVector<s32> const& word_lengths, std::vector<s32> const& state_lengths) const {
    inputs.emplace_back(mapping_.getOnnxName("word"), Onnx::Value::create(words));
    inputs.emplace_back(mapping_.getOnnxName("word-length"), Onnx::Value::create(word_lengths));
}

void OnnxRecurrentLanguageModel::extendTargets(std::vector<std::string>& targets) const {
    targets.emplace(targets.begin(), mapping_.getOnnxName("nn-output"));
}

void OnnxRecurrentLanguageModel::getOutputs(std::vector<std::pair<std::string, Onnx::Value>>& inputs, std::vector<Onnx::Value>& outputs, std::vector<std::string> const& targets) const {
    session_.run(std::move(inputs), targets, outputs);
}

std::vector<Onnx::Value> OnnxRecurrentLanguageModel::fetchStates(std::vector<Onnx::Value>& outputs) const {
    std::vector<Onnx::Value> state_vars(std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
    return state_vars;
}

Score OnnxRecurrentLanguageModel::transformOutput(Lm::CompressedVectorPtr<float> const& nn_output, size_t index) const {
    return softmax_adapter_->get_score(nn_output, index);
}

}  // namespace Lm

#endif  // _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HH
