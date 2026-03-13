#ifndef _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HXX
#define _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HXX

#include <Onnx/IOSpecification.hh>
#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>

#include "OnnxSoftmaxAdapter.hh"
#include "RecurrentLanguageModel.hh"

namespace Lm {

class OnnxRecurrentLanguageModel : public RecurrentLanguageModel<Onnx::Value, Onnx::OnnxStateVariable> {
public:
    using Precursor = RecurrentLanguageModel<Onnx::Value, Onnx::OnnxStateVariable>;

    OnnxRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~OnnxRecurrentLanguageModel() {}

protected:
    virtual void                     setState(std::vector<std::pair<std::string, Onnx::Value>> const& inputs, std::vector<std::string> const& targets) const;
    virtual void                     extendInputs(std::vector<std::pair<std::string, Onnx::Value>>& inputs, Math::FastMatrix<s32> const& words, Math::FastVector<s32> const& word_lengths, std::vector<s32> const& state_lengths) const;
    virtual void                     extendTargets(std::vector<std::string>& targets) const;
    virtual void                     getOutputs(std::vector<std::pair<std::string, Onnx::Value>>& inputs, std::vector<Onnx::Value>& outputs, std::vector<std::string> const& targets) const;
    virtual std::vector<Onnx::Value> fetchStates(std::vector<Onnx::Value>& outputs) const;

    virtual Score transformOutput(Lm::CompressedVectorPtr<float> const& nn_output, size_t index) const;

private:
    mutable Onnx::Session              session_;
    std::vector<Onnx::IOSpecification> ioSpec_;
    Onnx::IOMapping                    mapping_;
    Onnx::IOValidator                  validator_;

    std::unique_ptr<OnnxSoftmaxAdapter> softmax_adapter_;
};

}  // namespace Lm

#endif  // _LM_ONNX_RECURRENT_LANGUAGE_MODEL_HXX
