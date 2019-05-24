/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#ifndef SEARCH_LINEARPREDICTION_HH
#define SEARCH_LINEARPREDICTION_HH

#include <Core/Types.hh>
#include <iostream>
#include <vector>
#include "TreeStructure.hh"

namespace Search {
class LinearPrediction {
public:
    LinearPrediction(u32 bins, u32 maxKey)
            : maxKey_(maxKey) {
        recorded.resize(bins, Stat(0, 0));
    }

    void add(u32 key, u32 value) {
        verify(key < maxKey_);
        u32 pos = (key * recorded.size()) / maxKey_;
        recorded[pos].count += 1;
        recorded[pos].sum += value;
    }

    u32 totalCount() const {
        u32 ret = 0;
        for (u32 bin = 0; bin < recorded.size(); ++bin)
            ret += recorded[bin].count;
        return ret;
    }

    u32 predict(u32 key) const {
        // Get the closest 2 filled bins, and interpolate
        verify(key < maxKey_);
        u32 pos    = (key * recorded.size()) / maxKey_;
        int lower  = pos;
        int higher = pos;

        while (lower > 0 && recorded[lower].count == 0)
            --lower;

        while (higher < recorded.size() - 1 && recorded[higher].count == 0)
            ++higher;

        if (recorded[higher].count != 0 && recorded[lower].count != 0 && higher != lower) {
            // interpolate
            return ((recorded[higher].sum / recorded[higher].count) * (pos - lower) + (recorded[lower].sum / recorded[lower].count) * (higher - pos)) / (higher - lower);
        }
        else if (recorded[lower].count) {
            return recorded[lower].sum / recorded[lower].count;
        }
        else if (recorded[higher].count) {
            return recorded[higher].sum / recorded[higher].count;
        }
        else {
            return 0;
        }
    }

    bool read(Core::MappedArchiveReader& reader) {
        u32               maxKey;
        std::vector<Stat> inRecorded;

        reader >> maxKey >> inRecorded;
        if (maxKey != maxKey_ || inRecorded.size() != recorded.size())
            return false;

        recorded.swap(inRecorded);
        return true;
    }

    void write(Core::MappedArchiveWriter& file) {
        file << maxKey_ << recorded;
    }

private:
    struct Stat {
        Stat(u32 c = 0, u32 s = 0)
                : count(c), sum(s) {
        }

        u32 count;
        u32 sum;
    };

    u32               maxKey_;
    std::vector<Stat> recorded;
};
}  // namespace Search

#endif
