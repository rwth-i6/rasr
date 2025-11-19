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
#include <Flf/Copy.hh>
#include <Flf/FlfCore/Lattice.hh>
#include <Flf/Lexicon.hh>
#include <OpenFst/FstMapper.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/LatticeReader.hh>

namespace Search {
namespace Wfst {

template<class, class>
class FlfWeightConverter {
public:
    FlfWeightConverter(Flf::ConstSemiringRef semiring)
            : semiring_(semiring) {}
    Flf::Semiring::Weight operator()(const LatticeWeight& w) const {
        Flf::Semiring::Weight r = semiring_->create();
        r->set(0, w.AmScore());
        r->set(1, w.LmScore());
        return r;
    }

private:
    Flf::ConstSemiringRef semiring_;
};

const Core::ParameterString LatticeArchiveReader::paramInputAlphabet(
        "input-alphabet", "input alphabet", "lemma");
const Core::ParameterString LatticeArchiveReader::paramOutputAlphabet(
        "output-alphabet", "output alphabet", "lemma");

LatticeArchiveReader::LatticeArchiveReader(const Core::Configuration& config,
                                           const std::string&         pathname)
        : Flf::LatticeArchiveReader(config, pathname) {
    archive_       = new Search::Wfst::LatticeArchive(config, pathname);
    inputAlphabet_ = Flf::Lexicon::us()->alphabet(
            Flf::Lexicon::us()->alphabetId(paramInputAlphabet(config)));
    outputAlphabet_ = Flf::Lexicon::us()->alphabet(
            Flf::Lexicon::us()->alphabetId(paramOutputAlphabet(config)));
}

Flf::ConstLatticeRef LatticeArchiveReader::get(const std::string& id) {
    Search::Wfst::Lattice* lattice = archive_->read(id, true);
    Flf::ConstLatticeRef   result;
    if (lattice) {
        result = converter_.convert(*lattice, inputAlphabet_, outputAlphabet_);
        delete lattice;
    }
    return result;
}

void FlfConverter::createSemiring() {
    semiring_ = Flf::ConstSemiringRef(new Flf::TropicalSemiring(2));
    semiring_->setKey(0, "am");
    semiring_->setKey(1, "lm");
}

Flf::ConstLatticeRef FlfConverter::convert(const Search::Wfst::Lattice& lattice,
                                           Fsa::ConstAlphabetRef        inputAlphabet,
                                           Fsa::ConstAlphabetRef        outputAlphabet) {
    typedef Search::Wfst::Lattice::WordBoundaries                                                    WordBoundaries;
    typedef OpenFst::FstMapperAutomaton<Flf::Semiring, LatticeArc, FlfWeightConverter, Flf::Lattice> Mapper;

    createSemiring();

    Mapper* mapper = new Mapper(&lattice, semiring_, FlfWeightConverter<LatticeWeight, Flf::Semiring::Weight>(semiring_));
    mapper->setInputAlphabet(inputAlphabet);
    mapper->setOutputAlphabet(outputAlphabet);
    Flf::StaticLattice* flfLattice = new Flf::StaticLattice();

    Flf::ConstLatticeRef mapperRef(mapper);
    Flf::deepCopy(mapperRef, flfLattice, 0);

    if (!lattice.wordBoundaries().empty()) {
        Flf::StaticBoundaries* flfBoundaries = new Flf::StaticBoundaries();
        const WordBoundaries&  boundaries    = lattice.wordBoundaries();
        for (WordBoundaries::const_iterator b = boundaries.begin(); b != boundaries.end(); ++b)
            flfBoundaries->push_back(Flf::Boundary(*b));
        flfLattice->setBoundaries(Flf::ConstBoundariesRef(flfBoundaries));
    }
    return Flf::ConstLatticeRef(flfLattice);
}

}  // namespace Wfst
}  // namespace Search
