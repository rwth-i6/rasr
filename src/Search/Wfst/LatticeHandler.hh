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
#ifndef _SEARCH_WFST_LATTICE_HANDLER_HH
#define _SEARCH_WFST_LATTICE_HANDLER_HH

#include <Search/LatticeHandler.hh>
#include <Search/Wfst/LatticeArchive.hh>

namespace Search {
namespace Wfst {

class Lattice;
class LatticeArchive;

class LatticeHandler : public Search::LatticeHandler {
public:
    LatticeHandler(const Core::Configuration& c, Search::LatticeHandler* parent)
            : Search::LatticeHandler(c),
              parent_(parent),
              archive_(c) {}
    virtual ~LatticeHandler();

    bool write(const std::string& id, const WordLatticeAdaptor& l) {
        return parent_->write(id, l);
    }
    bool write(const std::string& id, const FlfLatticeAdaptor& l) {
        return parent_->write(id, l);
    }
    bool write(const std::string& id, const WfstLatticeAdaptor& l);

    Core::Ref<LatticeAdaptor> read(const std::string& id, const std::string& name);

    void setLexicon(Core::Ref<const Bliss::Lexicon> lexicon) {
        parent_->setLexicon(lexicon);
    }
    Core::Ref<const Bliss::Lexicon> lexicon() const {
        return parent_->lexicon();
    }

    ConstWordLatticeRef convert(const WordLatticeAdaptor& l) const {
        return parent_->convert(l);
    }
    ConstWordLatticeRef convert(const FlfLatticeAdaptor& l) const {
        return parent_->convert(l);
    }
    ConstWordLatticeRef convert(const WfstLatticeAdaptor& l) const;

private:
    Search::LatticeHandler* parent_;
    LatticeArchive          archive_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_LATTICE_HANDLER_HH
