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
#ifndef _SEARCH_WFST_LATTICE_ADAPTOR_HH
#define _SEARCH_WFST_LATTICE_ADAPTOR_HH

#include <Search/LatticeAdaptor.hh>

namespace Search {
namespace Wfst {

class Lattice;

class WfstLatticeAdaptor : public Search::LatticeAdaptor {
public:
    typedef Search::Wfst::Lattice          Lattice;
    typedef ::Lattice::ConstWordLatticeRef WordLatticeRef;

    WfstLatticeAdaptor()
            : l_(0) {}
    WfstLatticeAdaptor(const Lattice* l)
            : l_(l) {}
    virtual ~WfstLatticeAdaptor();

    bool write(const std::string& id, Search::LatticeHandler* handler) const;

    const Lattice* get() const {
        return l_;
    }

    WordLatticeRef wordLattice(const Search::LatticeHandler* handler) const;

    bool empty() const {
        return !l_;
    }

protected:
    const Lattice* l_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_LATTICE_ADAPTOR_HH
