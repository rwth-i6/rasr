#include "SharedDataHolder.hh"

namespace Nn {

SharedDataHolder::SharedDataHolder(SharedDataHolder const& data, size_t offset)
        : dataPtr_(static_cast<std::shared_ptr<const f32[]>>(data), data.get() + offset) {
}

SharedDataHolder::SharedDataHolder(std::shared_ptr<const f32[]> const& ptr, size_t offset)
        : dataPtr_(ptr, ptr.get() + offset) {
}

SharedDataHolder::SharedDataHolder(Core::Ref<const Mm::Feature::Vector> vec, size_t offset) {
    dataPtr_ = std::shared_ptr<const f32[]>(
            vec->data() + offset,
            [vecCopy = vec](const f32[]) mutable {});
}

#ifdef MODULE_ONNX
SharedDataHolder::SharedDataHolder(Onnx::Value&& value, size_t offset) {
    auto valueWrapper = std::make_shared<Onnx::Value>(std::move(value));
    dataPtr_          = std::shared_ptr<const f32[]>(
            valueWrapper->data<f32>() + offset,
            [valueWrapper](const f32[]) mutable {});
}
#endif

}  // namespace Nn
