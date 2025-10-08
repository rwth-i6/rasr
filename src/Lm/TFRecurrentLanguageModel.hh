#ifndef _LM_TF_RECURRENT_LANGUAGE_MODEL_HH
#define _LM_TF_RECURRENT_LANGUAGE_MODEL_HH

#include <Tensorflow/Graph.hh>
#include <Tensorflow/GraphLoader.hh>
#include <Tensorflow/Module.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/Tensor.hh>
#include <Tensorflow/TensorMap.hh>

#include "RecurrentLanguageModel.hh"
#include "TFSoftmaxAdapter.hh"

namespace Lm {

class TFRecurrentLanguageModel : public RecurrentLanguageModel<Tensorflow::Tensor, Tensorflow::Variable> {
public:
    using Precursor = RecurrentLanguageModel<Tensorflow::Tensor, Tensorflow::Variable>;

    TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~TFRecurrentLanguageModel() {}

protected:
    virtual void                            setState(std::vector<std::pair<std::string, Tensorflow::Tensor>> const& inputs, std::vector<std::string> const& targets) const;
    virtual void                            extendInputs(std::vector<std::pair<std::string, Tensorflow::Tensor>>& inputs, Math::FastMatrix<s32> const& words, Math::FastVector<s32> const& word_lengths, std::vector<s32> const& state_lengths) const;
    virtual void                            extendTargets(std::vector<std::string>& targets) const;
    virtual void                            getOutputs(std::vector<std::pair<std::string, Tensorflow::Tensor>>& inputs, std::vector<Tensorflow::Tensor>& outputs, std::vector<std::string> const& targets) const;
    virtual std::vector<Tensorflow::Tensor> fetchStates(std::vector<Tensorflow::Tensor>& outputs) const;

    virtual Score transformOutput(Lm::CompressedVectorPtr<float> const& nn_output, size_t index) const;

private:
    mutable Tensorflow::Session              session_;
    std::unique_ptr<Tensorflow::GraphLoader> loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;
    Tensorflow::TensorInputMap               tensor_input_map_;
    Tensorflow::TensorOutputMap              tensor_output_map_;

    std::unique_ptr<TFSoftmaxAdapter> softmax_adapter_;

    std::vector<std::string> initializer_tensor_names_;
    std::vector<std::string> output_tensor_names_;
    std::vector<std::string> read_vars_tensor_names_;
};

}  // namespace Lm

#endif  // _LM_TF_RECURRENT_LANGUAGE_MODEL_HH
