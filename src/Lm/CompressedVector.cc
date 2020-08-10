/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#include "CompressedVector.hh"

namespace Lm {

ContiguousBlockInfo::ContiguousBlockInfo(std::gslice const& slice)
        : start_(slice.start()),
          sizes_(slice.size()),
          strides_(slice.stride()),
          totalSize_(0ul),
          numBlocks_(1ul),
          blockSize_(1ul),
          firstIdxDim_(0ul) {
    size_t i                 = sizes_.size();
    size_t contiguous_stride = 1ul;
    while (i > 0ul) {
        i -= 1ul;
        bool contiguous = (contiguous_stride == strides_[i]);
        if (contiguous) {
            firstIdxDim_ = i;
        }
        numBlocks_ *= contiguous ? 1ul : sizes_[i];
        blockSize_ *= contiguous ? sizes_[i] : 1ul;
        contiguous_stride *= sizes_[i];
    }

    if (sizes_.size() > 0ul) {
        totalSize_ = 1ul;
        for (size_t i = 0ul; i < sizes_.size(); i++) {
            totalSize_ *= sizes_[i];
        }
    }
}

size_t ContiguousBlockInfo::blockOffset(size_t idx) const {
    size_t res = start_;
    for (int i = firstIdxDim_; i >= 0; i--) {
        if (sizes_[i] > 0) {
            res += strides_[i] * (idx % sizes_[i]);
            idx /= sizes_[i];
        }
    }

    return res;
}

}  // namespace Lm
