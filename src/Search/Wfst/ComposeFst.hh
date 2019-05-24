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
#ifndef _SEARCH_WFST_COMPOSEFST_HH
#define _SEARCH_WFST_COMPOSEFST_HH

#include <Core/Debug.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/GrammarFst.hh>
#include <Search/Wfst/Types.hh>
#include <fst/compose.h>
#include <fst/state-table.h>

namespace Search {
namespace Wfst {

/**
 * Interface for state tables.
 * FstLib::StateTable is templated on the composition filter state
 * without a common base class. The type of the composition filter state
 * is chosen at runtime by the type of lookahead used (see AbstractLexicalFst).
 */
class AbstractStateTable {
public:
    virtual ~AbstractStateTable() {}
    virtual OpenFst::StateId leftState(OpenFst::StateId s) const  = 0;
    virtual OpenFst::StateId rightState(OpenFst::StateId s) const = 0;
    virtual size_t           size() const                         = 0;
};

/**
 * Concrete state table for the given compose filter state
 */
template<class FilterState>
class StateTable : public AbstractStateTable {
public:
    typedef FstLib::GenericComposeStateTable<FstLib::StdArc, FilterState> Table;
    StateTable(const Table* table)
            : table_(table) {}
    virtual ~StateTable() {
        // table_ is deleted by the ComposeFst
    }
    OpenFst::StateId leftState(OpenFst::StateId s) const {
        return table_->Tuple(s).StateId1();
    }
    OpenFst::StateId rightState(OpenFst::StateId s) const {
        return table_->Tuple(s).StateId2();
    }
    size_t size() const {
        return table_->Size();
    }

protected:
    const Table* table_;
};

/**
 * Creates a ComposeFst (CL o G).
 *
 * Selects the correct composition filter and ComposeFstOptions based
 * on the class of the lexicon transducer and the type of the GrammarFst.
 * See ComposeFstFactory::create()
 */
class ComposeFstFactory {
private:
    typedef FstLib::StdFst                  StdFst;
    typedef StdFst::Arc                     Arc;
    typedef FstLib::ComposeFst<Arc>         ComposeFst;
    typedef AbstractGrammarFst::GrammarType GrammarType;
    typedef FstLib::SortedMatcher<StdFst>   SortedMatcher;
    typedef FailArcGrammarFst::Matcher      FailArcMatcher;

    template<LookAheadType t, class M1, class M2>
    struct ComposeOptions {};

    /**
     * Filter for Composition without look-ahead.
     */
    template<class M1, class M2>
    struct ComposeOptions<NoLookAhead, M1, M2> {
        typedef SortedMatcher                                           FstMatcher1;
        typedef FstMatcher1                                             FstMatcher2;
        typedef FstLib::SequenceComposeFilter<FstMatcher1, FstMatcher2> ComposeFilter;
    };

    /**
     * Filter for composition with phi filter (fail arcs in G).
     * In this case matching is performed on the arcs in G.
     */
    template<class M1>
    struct ComposeOptions<NoLookAhead, M1, FailArcMatcher> {
        typedef FailArcMatcher                                                FstMatcher1;
        typedef FailArcMatcher                                                FstMatcher2;
        typedef FstLib::SequenceComposeFilter<FailArcMatcher, FailArcMatcher> ComposeFilter;
    };

    /**
     * ComposeFilter used for the ComposeFst providing label look-ahead.
     * In the default case we can use the abstract LookAheadMatcher which
     * chooses the correct Matcher implementation from the MatcherFst.
     */
    template<class M1, class M2>
    struct ComposeOptions<LabelLookAhead, M1, M2> {
        typedef FstLib::LookAheadMatcher<FstLib::StdFst>                   FstMatcher1;
        typedef M2                                                         FstMatcher2;
        typedef FstLib::AltSequenceComposeFilter<FstMatcher1, FstMatcher2> SF;
        typedef FstLib::LookAheadComposeFilter<SF, FstMatcher1, FstMatcher2,
                                               FstLib::MATCH_OUTPUT>
                ComposeFilter;
    };

    /**
     * Specialization for DynamicLmFst.
     *
     * The specialized ArcIterator for DynamicLmFst relies on correct arc iterator
     * flags. The default ArcIterator does not forward calls to SetFlags() to
     * its base_ member.
     *
     * If the LookAheadComposeFilter should use the specialized ArcIterator for
     * DynamicLmFst (in LabelLookAheadMatcher::LookAheadFst() called by
     * LookAheadComposeFilter::LookAheadFilterArc()), both Matchers have to be
     * specified. Furthermore, we cannot use LookAheadMatcher, as
     * LookAheadMatcher::LookAheadFst(Fst&) drops the type information of its
     * argument before forwarding the call to the actual Matcher.
     *
     * The constructor of ComposeFst using ComposeFstImplOptions requires M1::FST
     * and M2::FST as parameters.
     * LabelLookAheadMatcher< SortedMatcher<ConstFst> >::FST = ConstFst
     * MatcherFst<ConstFst> is derived from ExpandedFst
     * Therefore we cannot use MatcherSelector::Matcher and
     * MatcherSelector::MatcherFst, but have to define the LabelLookAheadMatcher
     * with ExpandedFst as template Parameter.
     *
     * MATCH_OUTPUT has to be specified as well (instead of the default MATCH_BOTH),
     * to avoid problems in the constructor of LabelLookAheadFilter.
     */
    template<class M1>
    struct ComposeOptions<LabelLookAhead, M1, DynamicLmFstMatcher> {
        typedef M1                                                FstMatcher1;
        typedef DynamicLmFstMatcher                               FstMatcher2;
        typedef FstLib::AltSequenceComposeFilter<M1, FstMatcher2> SF;
        typedef FstLib::LookAheadComposeFilter<SF, M1, FstMatcher2,
                                               FstLib::MATCH_OUTPUT>
                ComposeFilter;
    };

    /**
     * ComposeFilter for composition with weight pushing (and label look-ahead).
     */
    template<class M1, class M2>
    struct ComposeOptions<PushWeights, M1, M2> {
        typedef ComposeOptions<LabelLookAhead, M1, M2>                                               LookAheadOptions;
        typedef typename LookAheadOptions::FstMatcher1                                               FstMatcher1;
        typedef typename LookAheadOptions::FstMatcher2                                               FstMatcher2;
        typedef typename LookAheadOptions::ComposeFilter                                             LF;
        typedef FstLib::PushWeightsComposeFilter<LF, FstMatcher1, FstMatcher2, FstLib::MATCH_OUTPUT> ComposeFilter;
    };

    /**
     * ComposeFilter for composition with label and weight pushing
     * (and label look-ahead).
     */
    template<class M1, class M2>
    struct ComposeOptions<PushLabels, M1, M2> {
        typedef ComposeOptions<PushWeights, M1, M2>                                                 PushWeightOptions;
        typedef typename PushWeightOptions::FstMatcher1                                             FstMatcher1;
        typedef typename PushWeightOptions::FstMatcher2                                             FstMatcher2;
        typedef typename PushWeightOptions::ComposeFilter                                           WF;
        typedef FstLib::PushLabelsComposeFilter<WF, FstMatcher1, FstMatcher2, FstLib::MATCH_OUTPUT> ComposeFilter;
    };

    /**
     * ComposeFilter for composition with label pushing
     * (and label look-ahead).
     */
    template<class M1, class M2>
    struct ComposeOptions<PushLabelsOnly, M1, M2> {
        typedef ComposeOptions<LabelLookAhead, M1, M2>                                              LookAheadOptions;
        typedef typename LookAheadOptions::FstMatcher1                                              FstMatcher1;
        typedef typename LookAheadOptions::FstMatcher2                                              FstMatcher2;
        typedef typename LookAheadOptions::ComposeFilter                                            LF;
        typedef FstLib::PushLabelsComposeFilter<LF, FstMatcher1, FstMatcher2, FstLib::MATCH_OUTPUT> ComposeFilter;
    };

    /**
     * ComposeFilter for composition with arc look-ahead.
     */
    template<class M1, class M2>
    struct ComposeOptions<ArcLookAhead, M1, M2> {
        typedef ComposeOptions<LabelLookAhead, M1, M2>   LookAheadOptions;
        typedef typename LookAheadOptions::FstMatcher1   FstMatcher1;
        typedef typename LookAheadOptions::FstMatcher2   FstMatcher2;
        typedef typename LookAheadOptions::ComposeFilter ComposeFilter;
    };

    /**
     * Construction helper.
     * Generates ComposeFstImplOptions and call the ComposeFst constructor.
     * create() is overloaded in order to get the correct Fst for
     * DynamicLmFst (see ComposeOption above).
     */
    template<class O>
    struct Compose {
        typedef StateTable<typename O::ComposeFilter::FilterState> ComposeStateTable;
        typedef FstLib::ComposeFstImplOptions<
                typename O::FstMatcher1,
                typename O::FstMatcher2,
                typename O::ComposeFilter,
                typename ComposeStateTable::Table>
                ComposeFstOptions;
        typedef typename O::FstMatcher1 Matcher1;
        typedef typename O::FstMatcher2 Matcher2;
        /**
         * general case
         */
        template<class L, class R>
        ComposeFst* create(const L& l, const R& r, size_t cacheSize,
                           AbstractStateTable** stateTable,
                           Matcher1* matcher1, Matcher2* matcher2) const;
        /**
         * look-ahead on L with DynamicLmFst as G
         */
        template<class L>
        ComposeFst* create(const L& l, const DynamicLmFst& r,
                           size_t cacheSize, AbstractStateTable** stateTable,
                           Matcher1* matcher1, Matcher2* matcher2) const;

        /**
         * no look-ahead on L, but DynamicLmFst as G
         */
        ComposeFst* create(const FstLib::StdVectorFst& l, const DynamicLmFst& r,
                           size_t cacheSize, AbstractStateTable** stateTable,
                           Matcher1* matcher1, Matcher2* matcher2) const;

        /**
         * verifies we have the correct types and we choose the correct
         * constructor of ComposeFst
         */
        ComposeFst* get(const typename O::FstMatcher1::FST&,
                        const typename O::FstMatcher2::FST&,
                        size_t, AbstractStateTable**,
                        Matcher1* matcher1, Matcher2* matcher2) const;
    };

    /**
     * Delegate construction to Compose<O>
     */
    template<class F1, class F2, class O>
    static ComposeFst* create3(const F1& l, const F2& g, size_t cacheSize,
                               AbstractStateTable**     stateTable,
                               typename O::FstMatcher1* matcher1 = 0,
                               typename O::FstMatcher2* matcher2 = 0) {
        return Compose<O>().create(l, g, cacheSize, stateTable, matcher1, matcher2);
    }

    // Selection based on type of g.
    template<class F, LookAheadType t, class M>
    static ComposeFst* create2(const F& l, const AbstractGrammarFst& g, size_t cacheSize,
                               AbstractStateTable** stateTable) {
        GrammarType gtype = g.type();
        switch (gtype) {
            case AbstractGrammarFst::TypeDynamic: {
                const DynamicLmFst* rg       = static_cast<const DynamicLmFst*>(g.getFst());
                M*                  matcher1 = static_cast<M*>(l.InitMatcher(FstLib::MATCH_OUTPUT));
                return create3<F, DynamicLmFst, ComposeOptions<t, M, DynamicLmFstMatcher>>(
                        l, *rg, cacheSize, stateTable, matcher1);
                break;
            }
            case AbstractGrammarFst::TypeFailArc: {
                FailArcMatcher* matcher1 = new FailArcMatcher(l, FstLib::MATCH_NONE, FstLib::kNoLabel);
                FailArcMatcher* matcher2 = new FailArcMatcher(*g.getFst(), FstLib::MATCH_INPUT, FailArcGrammarFst::FailLabel, false, FstLib::MATCHER_REWRITE_NEVER);
                verify_eq(t, NoLookAhead);
                return create3<F, StdFst, ComposeOptions<NoLookAhead, FailArcMatcher, FailArcMatcher>>(
                        l, *g.getFst(), cacheSize, stateTable, matcher1, matcher2);
                break;
            }
            default:
                return create3<F, StdFst, ComposeOptions<t, M, SortedMatcher>>(
                        l, *g.getFst(), cacheSize, stateTable);
                break;
        }
    }

public:
    /**
     * Creates a ComposeFst for the given L and G transducers.
     * The actual composition filter is deduced from the type
     * of L (i.e. L::FilterType) and G (i.e. g.type()).
     */
    template<class L>
    static ComposeFst* create(const L& l, const AbstractGrammarFst& g, size_t cacheSize,
                              AbstractStateTable** stateTable) {
        typedef typename L::Fst     MatcherFst;
        typedef typename L::Matcher Matcher;
        const MatcherFst&           matcherFst = l.getFst();
        return create2<MatcherFst, L::FilterType, Matcher>(matcherFst, g, cacheSize, stateTable);
    }
};

template<class O>
template<class L, class R>
inline ComposeFstFactory::ComposeFst* ComposeFstFactory::Compose<O>::create(const L& l, const R& r, size_t cacheSize,
                                                                            AbstractStateTable** stateTable,
                                                                            Matcher1* matcher1, Matcher2* matcher2) const {
    const typename O::FstMatcher1::FST& f1 = l;
    const typename O::FstMatcher2::FST& f2 = r;
    return get(f1, f2, cacheSize, stateTable, matcher1, matcher2);
}

template<class O>
template<class L>
inline ComposeFstFactory::ComposeFst* ComposeFstFactory::Compose<O>::create(const L& l, const DynamicLmFst& r, size_t cacheSize,
                                                                            AbstractStateTable** stateTable,
                                                                            Matcher1* matcher1, Matcher2* matcher2) const {
    const typename O::FstMatcher1::FST& f1 = l.GetFst();
    const typename O::FstMatcher2::FST& f2 = r;
    return get(f1, f2, cacheSize, stateTable, matcher1, matcher2);
}

template<class O>
inline ComposeFstFactory::ComposeFst* ComposeFstFactory::Compose<O>::create(const FstLib::StdVectorFst& l, const DynamicLmFst& r, size_t cacheSize,
                                                                            AbstractStateTable** stateTable,
                                                                            Matcher1* matcher1, Matcher2* matcher2) const {
    const typename O::FstMatcher1::FST& f1 = l;
    const typename O::FstMatcher2::FST& f2 = r;
    return get(f1, f2, cacheSize, stateTable, matcher1, matcher2);
}

template<class O>
inline ComposeFstFactory::ComposeFst* ComposeFstFactory::Compose<O>::get(const typename O::FstMatcher1::FST& f1,
                                                                         const typename O::FstMatcher2::FST& f2,
                                                                         size_t                              cacheSize,
                                                                         AbstractStateTable**                stateTable,
                                                                         Matcher1* matcher1, Matcher2* matcher2) const {
    ComposeFstOptions options;
    options.matcher1         = matcher1;
    options.matcher2         = matcher2;
    options.gc_limit         = cacheSize;
    options.state_table      = new typename ComposeStateTable::Table(f1, f2);
    *stateTable              = new ComposeStateTable(options.state_table);
    FLAGS_fst_compat_symbols = false;
    return new ComposeFst(f1, f2, options);
}

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_COMPOSEFST_HH
