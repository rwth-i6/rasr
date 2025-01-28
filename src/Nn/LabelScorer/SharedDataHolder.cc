#include "SharedDataHolder.hh"

namespace Nn {

SharedDataHolder::SharedDataHolder(std::shared_ptr<const f32[]> const& ptr, size_t offset)
        : dataPtr_(ptr, ptr.get() + offset) {
}

SharedDataHolder::SharedDataHolder(std::vector<f32> const& vec, size_t offset) {
    auto inputWrapper = std::make_shared<std::vector<f32>>(vec);
    dataPtr_          = std::shared_ptr<const f32[]>(
            inputWrapper->data() + offset,
            [inputWrapper](const f32*) mutable {});
}

#ifdef MODULE_ONNX
SharedDataHolder::SharedDataHolder(Onnx::Value const& value, size_t offset) {
    auto inputWrapper = std::make_shared<Onnx::Value>(value);
    dataPtr_          = std::shared_ptr<const f32[]>(
            inputWrapper->data<f32>() + offset,
            [inputWrapper](const f32*) mutable {});
}
#endif

}  // namespace Nn
