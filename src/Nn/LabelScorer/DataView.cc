#include "DataView.hh"

namespace Nn {

DataView::DataView(DataView const& dataView)
        : dataPtr_(dataView.dataPtr_), size_(dataView.size_) {
}

DataView::DataView(DataView const& dataView, size_t size, size_t offset)
        : dataPtr_(dataView.dataPtr_, dataView.data() + offset), size_(size) {
}

DataView::DataView(std::shared_ptr<f32 const[]> const& ptr, size_t size, size_t offset)
        : size_(size) {
    // Use aliasing constructor to create sub-shared_ptr that shares ownership with the original one but points to offset memory location
    dataPtr_ = std::shared_ptr<f32 const[]>(ptr, ptr.get() + offset);
}

DataView::DataView(Core::Ref<Mm::Feature::Vector const> const& featureVectorRef) : size_(featureVectorRef->size()) {
    // Copy Ref in custom deleter to keep it alive
    dataPtr_ = std::shared_ptr<f32 const[]>(
            featureVectorRef->data(),
            [featureVectorRef](f32 const[]) mutable {});
}

#ifdef MODULE_ONNX
DataView::DataView(Onnx::Value&& value) {
    // Move Onnx value into a shared_ptr to enable ref counting without requiring a copy
    auto valuePtr = std::make_shared<Onnx::Value>(std::move(value));

    // Create f32 shared_ptr based on Onnx value shared_ptr
    dataPtr_ = std::shared_ptr<f32 const[]>(
            valuePtr->data<f32>(),
            [valuePtr](f32 const[]) mutable {});

    size_ = 1ul;
    for (int d = 0ul; d < valuePtr->numDims(); ++d) {
        size_ *= valuePtr->dimSize(d);
    }
}
#endif

#ifdef MODULE_PYTHON
DataView::DataView(pybind11::array_t<f32> const& array, size_t size, size_t offset) : size_(size) {
    // Copy array (increasing its ref counter) in custom deleter to keep it alive
    dataPtr_ = std::shared_ptr<f32 const[]>(
            array.data() + offset,
            [array](f32 const[]) mutable {});
}
#endif

}  // namespace Nn
