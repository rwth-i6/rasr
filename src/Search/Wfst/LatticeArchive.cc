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
#include <Core/Archive.hh>
#include <Core/BinaryStream.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/LatticeArchive.hh>
#include <fst/fst.h>

namespace Search {
namespace Wfst {

const Core::ParameterString LatticeArchive::paramPath(
        "path", "lattice archive path", "");

const char* LatticeArchive::fstSuffix        = ".fst";
const char* LatticeArchive::boundariesSuffix = ".wb";

LatticeArchive::~LatticeArchive() {
    delete archive_;
}

bool LatticeArchive::write(const std::string& id, const Search::Wfst::Lattice* l) {
    if (!openArchive(true))
        return false;
    {
        Core::ArchiveWriter writer(*archive_, id + fstSuffix);
        if (!l->Write(writer, FstLib::FstWriteOptions()))
            return false;
    }
    {
        Core::ArchiveWriter      writer(*archive_, id + boundariesSuffix);
        Core::BinaryOutputStream os(writer);
        os << l->wordBoundaries();
    }
    return true;
}

Lattice* LatticeArchive::read(const std::string& id, bool readBoundaries) {
    Lattice* l = 0;
    if (!openArchive(false) || !archive_->hasFile(id + fstSuffix))
        return l;
    {
        Core::ArchiveReader reader(*archive_, id + fstSuffix);
        l = Lattice::Read(reader, FstLib::FstReadOptions());
    }
    if (l && readBoundaries && archive_->hasFile(id + boundariesSuffix)) {
        Core::ArchiveReader     reader(*archive_, id + boundariesSuffix);
        Core::BinaryInputStream is(reader);
        is >> l->wordBoundaries();
    }
    return l;
}

bool LatticeArchive::openArchive(bool write) {
    Core::Archive::AccessMode mode = (write ? Core::Archive::AccessModeWrite : Core::Archive::AccessModeRead);
    if (!archive_ || !archive_->hasAccess(mode)) {
        delete archive_;
        archive_ = Core::Archive::create(config, path_, mode);
    }
    if (archive_->hasFatalErrors()) {
        delete archive_;
        archive_ = 0;
    }
    return archive_;
}

}  // namespace Wfst
}  // namespace Search
