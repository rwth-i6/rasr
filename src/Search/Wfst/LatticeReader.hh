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
#ifndef _SEARCH_WFST_LATTICE_READER_HH
#define _SEARCH_WFST_LATTICE_READER_HH

#include <Flf/Archive.hh>
#include <Search/Wfst/LatticeArchive.hh>

namespace Search { namespace Wfst {

class FlfConverter
{
public:
    Flf::ConstLatticeRef convert(const Search::Wfst::Lattice  &lattice,
                                 Fsa::ConstAlphabetRef inputAlphabet,
                                 Fsa::ConstAlphabetRef outputAlphabet);
protected:
    void createSemiring();
    Flf::ConstSemiringRef semiring_;
};


class LatticeArchiveReader : public Flf::LatticeArchiveReader
{
    static const Core::ParameterString paramInputAlphabet;
    static const Core::ParameterString paramOutputAlphabet;
public:
    LatticeArchiveReader(const Core::Configuration&,
                  const std::string &pathname);
    virtual ~LatticeArchiveReader() { delete archive_; }
    Flf::ConstLatticeRef get(const std::string &id);
protected:
protected:
    std::string defaultSuffix() const { return ".fst"; }
    Search::Wfst::LatticeArchive *archive_;
    Fsa::ConstAlphabetRef inputAlphabet_, outputAlphabet_;
    FlfConverter converter_;
};

} // namespace Wfst
} // namespace Search

#endif // _SEARCH_WFST_LATTICE_READER_HH
