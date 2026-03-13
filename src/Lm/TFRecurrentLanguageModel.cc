#include "TFRecurrentLanguageModel.hh"

#include "TFBlasNceSoftmaxAdapter.hh"
#include "TFLstmStateManager.hh"
#include "TFNceSoftmaxAdapter.hh"
#include "TFPassthroughSoftmaxAdapter.hh"
#include "TFQuantizedBlasNceSoftmaxAdapter.hh"
#include "TFSoftmaxAdapter.hh"
#include "TFTransformerStateManager.hh"

namespace Lm {

enum StateManagerType {
    LstmStateManagerType,
    TransformerStateManagerType,
    TransformerStateManager16BitType,
    TransformerStateManager8BitType,
    TransformerStateManagerWithCommonPrefixType,
    TransformerStateManagerWithCommonPrefix16BitType,
    TransformerStateManagerWithCommonPrefix8BitType,
};

const Core::Choice stateManagerTypeChoice(
        "lstm", LstmStateManagerType,
        "transformer", TransformerStateManagerType,
        "transformer-16bit", TransformerStateManager16BitType,
        "transformer-8bit", TransformerStateManager8BitType,
        "transformer-with-common-prefix", TransformerStateManagerWithCommonPrefixType,
        "transformer-with-common-prefix-16bit", TransformerStateManagerWithCommonPrefix16BitType,
        "transformer-with-common-prefix-8bit", TransformerStateManagerWithCommonPrefix8BitType,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice,
        "type of the state manager",
        LstmStateManagerType);

std::unique_ptr<TFStateManager> createStateManager(Core::Configuration const& config) {
    TFStateManager* res = nullptr;
    switch (stateManagerTypeParam(config)) {
        case LstmStateManagerType: res = new Lm::TFLstmStateManager(config); break;
        case TransformerStateManagerType: res = new Lm::TFTransformerStateManager<float>(config); break;
        case TransformerStateManager16BitType: res = new Lm::TFTransformerStateManager<int16_t>(config); break;
        case TransformerStateManager8BitType: res = new Lm::TFTransformerStateManager<int8_t>(config); break;
        case TransformerStateManagerWithCommonPrefixType: res = new Lm::TFTransformerStateManagerWithCommonPrefix<float>(config); break;
        case TransformerStateManagerWithCommonPrefix16BitType: res = new Lm::TFTransformerStateManagerWithCommonPrefix<int16_t>(config); break;
        case TransformerStateManagerWithCommonPrefix8BitType: res = new Lm::TFTransformerStateManagerWithCommonPrefix<int8_t>(config); break;
        default: defect();
    }
    return std::unique_ptr<TFStateManager>(res);
}

enum SoftmaxAdapterType {
    TFBlasNceSoftmaxAdapterType,
    TFNceSoftmaxAdapterType,
    TFPassthroughSoftmaxAdapterType,
    TFQuantizedBlasNceSoftmaxAdapter16BitType
};

const Core::Choice softmaxAdapterTypeChoice(
        "blas_nce", TFBlasNceSoftmaxAdapterType,  // included for backward compatibility
        "blas-nce", TFBlasNceSoftmaxAdapterType,  // more consistent with RASR conventions
        "nce", TFNceSoftmaxAdapterType,
        "passthrough", TFPassthroughSoftmaxAdapterType,
        "quantized-blas-nce-16bit", TFQuantizedBlasNceSoftmaxAdapter16BitType,
        Core::Choice::endMark());

const Core::ParameterChoice softmaxAdapterTypeParam(
        "type", &softmaxAdapterTypeChoice,
        "type of the softmax adapter",
        TFPassthroughSoftmaxAdapterType);

std::unique_ptr<TFSoftmaxAdapter> createSoftmaxAdapter(Core::Configuration const& config) {
    switch (softmaxAdapterTypeParam(config)) {
        case TFBlasNceSoftmaxAdapterType: return std::unique_ptr<TFSoftmaxAdapter>(new Lm::TFBlasNceSoftmaxAdapter(config));
        case TFNceSoftmaxAdapterType: return std::unique_ptr<TFSoftmaxAdapter>(new Lm::TFNceSoftmaxAdapter(config));
        case TFPassthroughSoftmaxAdapterType: return std::unique_ptr<TFSoftmaxAdapter>(new Lm::TFPassthroughSoftmaxAdapter(config));
        case TFQuantizedBlasNceSoftmaxAdapter16BitType: return std::unique_ptr<TFSoftmaxAdapter>(new Lm::TFQuantizedBlasNceSoftmaxAdapter16Bit(config));
        default: defect();
    }
}

TFRecurrentLanguageModel::TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
        : Core::Component(c),
          Precursor(c, l, createStateManager(select("state-manager"))),
          session_(select("session")),
          loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
          graph_(loader_->load_graph()),
          tensor_input_map_(select("input-map")),
          tensor_output_map_(select("output-map")),
          softmax_adapter_(createSoftmaxAdapter(select("softmax-adapter"))) {
    session_.addGraph(*graph_);
    loader_->initialize(session_);

    auto const& softmax_info = tensor_output_map_.get_info("softmax");
    output_tensor_names_.push_back(softmax_info.tensor_name());
    state_variables_.reserve(graph_->state_vars().size());
    for (std::string const& s : graph_->state_vars()) {
        auto const& var = graph_->variables().find(s)->second;
        state_variables_.emplace_back(var);
        initializer_tensor_names_.push_back(var.initializer_name);
        read_vars_tensor_names_.push_back(var.snapshot_name);
    }

    if (state_variables_.empty()) {
        error("No recurrent state variables found in tensorflow graph of recurrent language model.");
    }

    setEmptyHistory();

    softmax_adapter_->init(session_, tensor_input_map_, tensor_output_map_);
}

void TFRecurrentLanguageModel::setState(std::vector<std::pair<std::string, Tensorflow::Tensor>> const& inputs, std::vector<std::string> const& targets) const {
    session_.run(inputs, targets);
}

void TFRecurrentLanguageModel::extendInputs(std::vector<std::pair<std::string, Tensorflow::Tensor>>& inputs, Math::FastMatrix<s32> const& words, Math::FastVector<s32> const& word_lengths, std::vector<s32> const& state_lengths) const {
    inputs.clear();
    auto const& word_info = tensor_input_map_.get_info("word");
    inputs.emplace_back(std::make_pair(word_info.tensor_name(), Tensorflow::Tensor::create(words)));
    if (not word_info.seq_length_tensor_name().empty()) {
        inputs.emplace_back(std::make_pair(word_info.seq_length_tensor_name(), Tensorflow::Tensor::create(word_lengths)));
    }
    if (tensor_input_map_.has_info("state-lengths")) {
        auto const& state_lengths_info = tensor_input_map_.get_info("state-lengths");
        inputs.emplace_back(std::make_pair(state_lengths_info.tensor_name(), Tensorflow::Tensor::create(state_lengths)));
    }
}

void TFRecurrentLanguageModel::extendTargets(std::vector<std::string>& targets) const {
}

void TFRecurrentLanguageModel::getOutputs(std::vector<std::pair<std::string, Tensorflow::Tensor>>& inputs, std::vector<Tensorflow::Tensor>& outputs, std::vector<std::string> const& targets) const {
    session_.run(inputs, output_tensor_names_, graph_->update_ops(), outputs);
}

std::vector<Tensorflow::Tensor> TFRecurrentLanguageModel::fetchStates(std::vector<Tensorflow::Tensor>& outputs) const {
    session_.run({}, read_vars_tensor_names_, {}, outputs);
    return outputs;
}

Score TFRecurrentLanguageModel::transformOutput(Lm::CompressedVectorPtr<float> const& nn_output, size_t index) const {
    return softmax_adapter_->get_score(nn_output, index);
}

}  // namespace Lm
