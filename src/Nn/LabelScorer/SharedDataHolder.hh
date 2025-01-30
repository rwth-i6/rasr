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

#ifndef SHARED_DATA_HOLDER_HH
#define SHARED_DATA_HOLDER_HH

#include <Mm/Feature.hh>

#ifdef MODULE_ONNX
#include <Onnx/Value.hh>
#endif

namespace Nn {

/*
 * Wraps the data of various data structures in a `std::shared_ptr`
 * without copying while making sure that the data is not invalidated.
 * This is achieved via custom deleters in the shared_ptr's which tie
 * the lifetime of the original datastructure to the shared_ptr.
 */
class SharedDataHolder {
public:
    SharedDataHolder(SharedDataHolder const& data, size_t offset = 0ul);
    SharedDataHolder(Core::Ref<const Mm::Feature::Vector> vec, size_t offset = 0ul);
    SharedDataHolder(std::shared_ptr<const f32[]> const& ptr, size_t offset = 0ul);

#ifdef MODULE_ONNX
    SharedDataHolder(Onnx::Value&& value, size_t offset = 0ul);
#endif

    operator std::shared_ptr<const f32[]>() const {
        return dataPtr_;
    }

    const f32* get() const {
        return dataPtr_.get();
    }

private:
    std::shared_ptr<const f32[]> dataPtr_;
};

}  // namespace Nn

#endif  // SHARED_DATA_HOLDER_HH
