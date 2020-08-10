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
#ifndef _SEARCH_LATTICE_NETWORK_HH
#define _SEARCH_LATTICE_NETWORK_HH

#include <Search/Wfst/Network.hh>
#include <fst/fst-decl.h>
#include <fst/matcher-fst.h>

namespace Search {
namespace Wfst {

class AbstractGrammarFst;
class AbstractLexicalFst;
class Lattice;
class LatticeArchive;

class LatticeNetwork : public StaticNetwork {
    static const Core::ParameterString paramLmFst;
    static const Core::ParameterString paramLexiconFst;
    static const Core::ParameterBool   paramRemoveLexiconWeights;
    typedef Wfst::Lattice              Lattice;
    typedef Wfst::LatticeArchive       LatticeArchive;

    typedef FstLib::LabelLookAheadMatcher<
            FstLib::SortedMatcher<FstLib::ConstFst<FstLib::StdArc>>,
            FstLib::olabel_lookahead_flags,
            FstLib::DefaultAccumulator<FstLib::StdArc>>
            Matcher;

    typedef FstLib::MatcherFst<
            FstLib::ConstFst<FstLib::StdArc>,
            Matcher,
            FstLib::olabel_lookahead_fst_type,
            FstLib::LabelLookAheadRelabeler<FstLib::StdArc>>
            LexiconFst;

public:
    LatticeNetwork(const Core::Configuration& c)
            : StaticNetwork(c), l_(0), g_(0), archive_(0) {}
    virtual ~LatticeNetwork();
    virtual bool init();
    virtual void reset() {}
    virtual void setSegment(const std::string& name) {
        loadLattice(name);
    }
    virtual u32 nStates() const {
        return f_->NumStates();
    }
    static bool hasGrammarState() {
        return false;
    }

private:
    bool                loadLexicon(const std::string& file);
    bool                loadGrammar(const std::string& file);
    bool                loadLattice(const std::string& name);
    FstLib::StdFst*     getLmLattice(const Lattice& lattice) const;
    void                createNetwork(const FstLib::StdFst& lmLattice);
    AbstractLexicalFst* l_;
    AbstractGrammarFst* g_;
    LatticeArchive*     archive_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_LATTICE_NETWORK_HH
