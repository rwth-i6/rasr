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
#include <Lattice/LatticeAdaptor.hh>
#include <OpenFst/Input.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/LatticeAdaptor.hh>
#include <Search/Wfst/LatticeHandler.hh>

namespace Search {
namespace Wfst {

LatticeHandler::~LatticeHandler() {
    delete parent_;
}

bool LatticeHandler::write(const std::string& id, const WfstLatticeAdaptor& l) {
    if (format_ != formatOpenFst)
        return parent_->write(id, ::Lattice::WordLatticeAdaptor(l.wordLattice(this)));
    else {
        verify(!l.empty());
        return archive_.write(id, l.get());
    }
}

Core::Ref<LatticeAdaptor> LatticeHandler::read(const std::string& id, const std::string& name) {
    if (format_ != formatOpenFst)
        return parent_->read(id, name);
    Lattice* l = archive_.read(id);
    return Core::ref(new WfstLatticeAdaptor(l));
}

LatticeHandler::ConstWordLatticeRef LatticeHandler::convert(const WfstLatticeAdaptor& l) const {
    require(lexicon());
    if (l.empty())
        return ::Lattice::ConstWordLatticeRef();
    ::Lattice::WordLattice*         lattice = new ::Lattice::WordLattice();
    LmScoreLattice                  lmScores(*l.get(), LatticeLmScoreMapper());
    AmScoreLattice                  amScores(*l.get(), LatticeAmScoreMapper());
    Core::Ref<Fsa::StaticAutomaton> lm = OpenFst::convertToFsa(lmScores);
    Core::Ref<Fsa::StaticAutomaton> am = OpenFst::convertToFsa(amScores);
    Fsa::ConstAlphabetRef           alphabet;
    switch (l.get()->outputType()) {
        case OutputLemma:
            alphabet = lexicon()->lemmaAlphabet();
            break;
        case OutputLemmaPronunciation:
            alphabet = lexicon()->lemmaPronunciationAlphabet();
            break;
        case OutputSyntacticToken:
            alphabet = lexicon()->syntacticTokenAlphabet();
            break;
        default:
            defect();
            break;
    }
    if (lm->type() == Fsa::TypeAcceptor)
        lm->setInputAlphabet(alphabet);
    else
        lm->setOutputAlphabet(alphabet);
    lattice->setFsa(lm, ::Lattice::WordLattice::lmFsa);
    lattice->setFsa(am, ::Lattice::WordLattice::acousticFsa);
    const Lattice::WordBoundaries& wb         = l.get()->wordBoundaries();
    ::Lattice::WordBoundaries*     boundaries = new ::Lattice::WordBoundaries;
    for (int s = 0; s < wb.size(); ++s)
        boundaries->set(s, ::Lattice::WordBoundary(wb[s]));
    lattice->setWordBoundaries(Core::ref(boundaries));
    return ::Lattice::ConstWordLatticeRef(lattice);
}

}  // namespace Wfst
}  // namespace Search
