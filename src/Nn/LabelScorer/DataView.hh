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

#ifndef DATA_VIEW_HH
#define DATA_VIEW_HH

#include <Mm/Feature.hh>

#ifdef MODULE_ONNX
#include <Onnx/Value.hh>
#endif

#ifdef MODULE_PYTHON
#pragma push_macro("ensure")  // Macro duplication in numpy.h
#undef ensure
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#pragma pop_macro("ensure")
#endif

namespace Nn {

/*
 * This class encapsulates a std::shared_ptr and a size. The internal shared_ptr is tied
 * to the lifetime of the original data container in order to make sure
 * it stays valid as long as the view is alive.
 *
 * It can be initialized using various data containers such as a
 * Core::TsRef<Mm::Feature::Vector>, another shared_ptr, an Onnx::Value
 * or a pybind11::array_t.
 */
class DataView {
public:
    DataView(DataView const& dataView);
    DataView(DataView const& dataView, size_t size, size_t offset = 0ul);
    DataView(Core::TsRef<Mm::Feature::Vector const> const& featureVectorRef);
    DataView(std::shared_ptr<f32 const[]> const& ptr, size_t size, size_t offset = 0ul);

#ifdef MODULE_ONNX
    DataView(Onnx::Value&& value);
#endif

#ifdef MODULE_PYTHON
    DataView(pybind11::array_t<f32> const& array, size_t size, size_t offset = 0ul);
#endif

    f32 const* data() const {
        return dataPtr_.get();
    }

    f32 operator[](size_t idx) const {
        verify(idx < size_);
        return dataPtr_[idx];
    }

    size_t size() const {
        return size_;
    }

private:
    std::shared_ptr<f32 const[]> dataPtr_;
    size_t                       size_;
};

}  // namespace Nn

#endif  // DATA_VIEW_HH
