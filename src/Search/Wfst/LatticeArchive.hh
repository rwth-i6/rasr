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
#ifndef _SEARCH_WFST_LATTICE_ARCHIVE_HH
#define _SEARCH_WFST_LATTICE_ARCHIVE_HH

#include <Core/Component.hh>

namespace Core {
class Archive;
}

namespace Search {
namespace Wfst {

class Lattice;

class LatticeArchive : public Core::Component {
    static const Core::ParameterString paramPath;

public:
    LatticeArchive(const Core::Configuration& c)
            : Core::Component(c), archive_(0), path_(paramPath(c)) {}
    LatticeArchive(const Core::Configuration& c, const std::string& path)
            : Core::Component(c), archive_(0), path_(path) {}
    virtual ~LatticeArchive();
    bool                   write(const std::string& id, const Search::Wfst::Lattice* l);
    Search::Wfst::Lattice* read(const std::string& id, bool readBoundaries = true);

private:
    bool               openArchive(bool write);
    static const char *boundariesSuffix, *fstSuffix;
    Core::Archive*     archive_;
    std::string        path_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_LATTICE_ARCHIVE_HH
