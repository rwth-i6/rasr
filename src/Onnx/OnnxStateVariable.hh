#ifndef _ONNX_ONNX_STATE_VARIABLE_HH
#define _ONNX_ONNX_STATE_VARIABLE_HH

#include <string>

#include <Core/Types.hh>

namespace Onnx {

struct OnnxStateVariable {
    std::string      input_state_key;
    std::string      output_state_key;
    std::vector<s64> shape;
};

}  // namespace Onnx

#endif  // _ONNX_ONNX_STATE_VARIABLE_HH
