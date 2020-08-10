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
#ifndef _SEARCH_WFST_LATTICE_HH_
#define _SEARCH_WFST_LATTICE_HH_

#include <OpenFst/Types.hh>
#include <Search/Types.hh>
#include <Search/Wfst/ExpandingFsaSearch.hh>
#include <fst/arc-map.h>
#include <fst/float-weight.h>
#include <fst/pair-weight.h>
#include <fst/prune.h>
#include <fst/vector-fst.h>

namespace Search {
namespace Wfst {

class LatticeWeight : public FstLib::PairWeight<FstLib::TropicalWeight, FstLib::TropicalWeight> {
    typedef FstLib::TropicalWeight                             TropicalWeight;
    typedef FstLib::PairWeight<TropicalWeight, TropicalWeight> PairWeight;

public:
    using PairWeight::One;
    using PairWeight::Quantize;
    using PairWeight::Reverse;
    using PairWeight::Zero;

    typedef LatticeWeight ReverseWeight;

    LatticeWeight() {}
    LatticeWeight(const LatticeWeight& w)
            : PairWeight(w) {}
    LatticeWeight(const PairWeight& w)
            : PairWeight(w) {}
    LatticeWeight(TropicalWeight w1, TropicalWeight w2)
            : PairWeight(w1, w2) {}

    static const LatticeWeight& Zero() {
        static const LatticeWeight zero(PairWeight::Zero());
        return zero;
    }

    static const LatticeWeight& One() {
        static const LatticeWeight one(PairWeight::One());
        return one;
    }

    static const string& Type() {
        static const string type = "lattice-" + TropicalWeight::Type();
        return type;
    }

    static constexpr uint64 Properties() {
        /*! @todo: check properties */
        return TropicalWeight::Properties();
        ;
    }

    LatticeWeight Quantize(float delta = FstLib::kDelta) const {
        return PairWeight::Quantize(delta);
    }

    LatticeWeight Reverse() const {
        return PairWeight::Reverse();
    }

    TropicalWeight Combined() const {
        return FstLib::Times(Value1(), Value2());
    }

    float AmScore() const {
        return Value1().Value();
    }

    float LmScore() const {
        return Value2().Value();
    }

    size_t Hash() const {
        return Value1().Hash() + Value2().Hash();
    }
};

inline int Compare(const LatticeWeight& w, const LatticeWeight& v) {
    f32 f1 = w.AmScore() + w.LmScore(),
        f2 = v.AmScore() + v.LmScore();
    if (f1 < f2)
        return 1;
    else if (f1 > f1)
        return -1;
    else if (w.LmScore() < v.LmScore())
        return 1;
    else if (w.LmScore() > v.LmScore())
        return -1;
    else
        return 0;
}

inline LatticeWeight Plus(const LatticeWeight& w,
                          const LatticeWeight& v) {
    return (Compare(w, v) >= 0 ? w : v);
}

inline LatticeWeight Times(const LatticeWeight& w,
                           const LatticeWeight& v) {
    return LatticeWeight(w.AmScore() + v.AmScore(), w.LmScore() + v.LmScore());
}

inline LatticeWeight Divide(const LatticeWeight& w,
                            const LatticeWeight& v,
                            FstLib::DivideType   typ = FstLib::DIVIDE_ANY) {
    return LatticeWeight(Divide(w.Value1(), v.Value1(), typ),
                         Divide(w.Value2(), v.Value2(), typ));
}

typedef FstLib::ArcTpl<LatticeWeight> LatticeArc;

class Lattice : public FstLib::VectorFst<LatticeArc> {
    typedef FstLib::VectorFst<LatticeArc> VectorFst;
    typedef VectorFst::StateId            StateId;

public:
    typedef std::vector<TimeframeIndex> WordBoundaries;

    Lattice()
            : outputType_(DefaultOutput) {}
    explicit Lattice(const Lattice& o)
            : VectorFst(o), wordBoundaries_(o.wordBoundaries_), outputType_(DefaultOutput) {}
    explicit Lattice(const VectorFst& o)
            : VectorFst(o), outputType_(DefaultOutput) {}
    explicit Lattice(const FstLib::Fst<LatticeArc>& o)
            : VectorFst(o), outputType_(DefaultOutput) {}
    virtual ~Lattice() {}

    Lattice* Copy(bool safe = false) const {
        return new Lattice(*this);
    }

    /**
     * Call VectorFst::DeleteStates and handle wordBoundaries_
     */
    virtual void DeleteStates(const std::vector<StateId>& dstates) {
        if (!wordBoundaries_.empty()) {
            std::vector<StateId> sorted = dstates;
            std::sort(sorted.begin(), sorted.end());
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
            std::vector<StateId>::const_iterator d       = sorted.begin();
            StateId                              nstates = 0;
            for (StateId s = 0; s < wordBoundaries_.size(); ++s) {
                if (d != sorted.end() && s == *d) {
                    ++d;
                }
                else {
                    if (s != nstates)
                        wordBoundaries_[nstates] = wordBoundaries_[s];
                    ++nstates;
                }
            }
            wordBoundaries_.resize(nstates);
        }
        VectorFst::DeleteStates(dstates);
    }

    virtual void DeleteStates() {
        wordBoundaries_.clear();
        VectorFst::DeleteStates();
    }

    const WordBoundaries& wordBoundaries() const {
        return wordBoundaries_;
    }

    WordBoundaries& wordBoundaries() {
        return wordBoundaries_;
    }

    void setWordBoundary(StateId s, TimeframeIndex t) {
        if (s >= wordBoundaries_.size())
            wordBoundaries_.resize(s + 1, InvalidTimeframeIndex);
        wordBoundaries_[s] = t;
    }

    void prune(f32 threshold) {
        FstLib::Prune(this, Weight(0, threshold));
    }

    static Lattice* Read(const std::string& file) {
        VectorFst* f = VectorFst::Read(file);
        Lattice*   l = 0;
        if (f)
            l = new Lattice(*f);
        delete f;
        return l;
    }
    static Lattice* Read(std::istream& is, FstLib::FstReadOptions opt) {
        VectorFst* f = VectorFst::Read(is, opt);
        Lattice*   l = 0;
        if (f)
            l = new Lattice(*f);
        delete f;
        return l;
    }

    OutputType outputType() const {
        return outputType_;
    }
    void setOutputType(OutputType t) {
        outputType_ = t;
    }

private:
    WordBoundaries          wordBoundaries_;
    static const OutputType DefaultOutput = OutputLemmaPronunciation;
    OutputType              outputType_;
};

template<bool LmScore>
class WeightMapper {
public:
    typedef Lattice::Arc             FromArc;
    typedef OpenFst::Arc             ToArc;
    typedef typename FromArc::Weight FromWeight;
    typedef typename ToArc::Weight   ToWeight;

    ToArc operator()(const FromArc& arc) const {
        ToWeight w = (LmScore ? arc.weight.LmScore() : arc.weight.AmScore());
        return ToArc(arc.ilabel, arc.olabel, w, arc.nextstate);
    }

    FstLib::MapFinalAction FinalAction() const {
        return FstLib::MAP_NO_SUPERFINAL;
    }

    FstLib::MapSymbolsAction InputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }

    FstLib::MapSymbolsAction OutputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }

    uint64 Properties(uint64 props) const {
        return (props & FstLib::kWeightInvariantProperties);
    }
};

typedef WeightMapper<true>                                 LatticeLmScoreMapper;
typedef WeightMapper<false>                                LatticeAmScoreMapper;
typedef FstLib::RmWeightMapper<Lattice::Arc, OpenFst::Arc> LatticeRmScoreMapper;

typedef FstLib::ArcMapFst<LatticeArc, OpenFst::Arc, LatticeLmScoreMapper> LmScoreLattice;
typedef FstLib::ArcMapFst<LatticeArc, OpenFst::Arc, LatticeAmScoreMapper> AmScoreLattice;
typedef FstLib::ArcMapFst<LatticeArc, OpenFst::Arc, LatticeRmScoreMapper> RmScoreLattice;

}  // namespace Wfst
}  // namespace Search

namespace fst {

/**
 * Specialization for efficient comparison.
 */
template<>
class NaturalLess<Search::Wfst::LatticeWeight> {
public:
    typedef Search::Wfst::LatticeWeight Weight;
    NaturalLess() {}
    bool operator()(const Weight& w1, const Weight& w2) const {
        return (Search::Wfst::Compare(w1, w2) == 1);
    }
};

}  // namespace fst

#endif  // _SEARCH_WFST_LATTICE_HH_
