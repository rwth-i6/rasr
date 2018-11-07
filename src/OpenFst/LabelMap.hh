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
#ifndef _OPENFST_LABEL_MAP_HH
#define _OPENFST_LABEL_MAP_HH

#include <cstdlib>
#include <vector>
#include <Core/CompressedStream.hh>
#include <Core/StringUtilities.hh>
#include <OpenFst/Types.hh>

namespace OpenFst {

/**
 * label to label mapping.
 * File format:
 *   <to-label>\t<from-label>\n
 */
class LabelMap {
public:
    LabelMap() {}

    bool load(const std::string &filename) {
        Core::CompressedInputStream in(filename);
        Label from, to;
        if (!in) return false;
        while (in) {
            std::string line;
            std::getline(in, line);
            Core::stripWhitespace(line);
            std::vector<std::string> fields = Core::split(line, "\t");
            if (fields.size() != 2) continue;
            to = std::atol(fields[0].c_str());
            from = std::atol(fields[1].c_str());
            if (from >= map_.size())
                map_.resize(from + 1, 0);
            map_[from] = to;
        }
        return true;
    }

    Label mapLabel(OpenFst::Label from) const {
        return map_[from];
    }
    bool empty() const { return map_.empty(); }

private:
    std::vector<OpenFst::Label> map_;
};

}

#endif  // _OPENFST_LABEL_MAP_HH
