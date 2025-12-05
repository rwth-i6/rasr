#ifndef _LM_ONNX_STATE_VARIABLE_HH
#define _LM_ONNX_STATE_VARIABLE_HH

#include <Core/Types.hh>

#include <string>

namespace Lm {

struct OnnxStateVariable {
    std::string      input_state_key;
    std::string      output_state_key;
    std::vector<s64> shape;
};

}  // namespace Lm

#endif  // _LM_ONNX_STATE_VARIABLE_HH