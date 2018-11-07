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
#ifndef SEARCH_WFST_LEXICONFST_HH
#define SEARCH_WFST_LEXICONFST_HH

#include <OpenFst/Types.hh>
#include <Search/Wfst/ComposeFst.hh>
#include <Search/Wfst/DynamicLmFst.hh>
#include <Search/Wfst/GrammarFst.hh>
#include <Search/Wfst/Types.hh>
#include <fst/compose-filter.h>
#include <fst/fst.h>
#include <fst/lookahead-filter.h>
#include <fst/lookahead-matcher.h>
#include <fst/matcher.h>
#include <fst/matcher-fst.h>

namespace Search { namespace Wfst {

class AbstractLexicalFst;

/**
 * Loads and creates (expanded) lexicon transducers (C o L)
 */
class LexicalFstFactory : public Core::Component
{
    static const Core::ParameterChoice paramAccumulatorType_;
    static const Core::ParameterChoice paramLookAheadType_;
    static const Core::Choice choiceAccumulatorType_;
    static const Core::Choice choiceLookAheadType_;
    static const Core::ParameterBool paramMatcherFst_;
    static const Core::ParameterFloat paramScale_;

    typedef AbstractGrammarFst::GrammarType GrammarType;
    typedef FstLib::StdFst StdFst;
    typedef StdFst::Arc Arc;

public:
    enum AccumulatorType { DefaultAccumulator, LogAccumulator, FastLogAccumulator };

    struct Options {
        AccumulatorType accumulatorType;
        LookAheadType lookAhead;
    };

public:
    LexicalFstFactory(const Core::Configuration &c) : Core::Component(c) {}

    /**
     * creates LexicalFst from the given filename using parameter values.
     * relabels g if necessary.
     */
    AbstractLexicalFst* load(const std::string &filename, GrammarType gType,
                             AbstractGrammarFst *g) const;

    /**
     * creates LexicalFst from the given filename using parameter values,
     * with default values from the given options.
     * relabels g if necessary.
     */
    AbstractLexicalFst* load(const std::string &filename, const Options &o,
                             AbstractGrammarFst *g) const;

    AbstractLexicalFst* convert(OpenFst::VectorFst *base, GrammarType gType,
                                AbstractGrammarFst *g) const;

    AbstractLexicalFst* convert(OpenFst::VectorFst *base, const Options &o,
                                AbstractGrammarFst *g) const;


    /**
     * Factory function to create LexicalFst objects.
     */
    static AbstractLexicalFst* create(const Options &options);
private:
    Options parseOptions() const;
    Options parseOptions(const Options &defaultValues) const;
    void logOptions(const Options &options) const;
    AbstractLexicalFst* read(const std::string &filename, const Options &o,
                             bool isMatcherFst, f32 scale, AbstractGrammarFst *g) const;
    void convert(OpenFst::VectorFst *base, f32 scale,
                 AbstractLexicalFst *result, AbstractGrammarFst *g) const;


    /**
     * Internal helper function for create(options).
     * Selects accumulator
     */
    template<template <class> class N>
    static AbstractLexicalFst* createFst(AccumulatorType t);
};

// ===============================================================

/**
 * Interface for LexicalFstImpl.
 * See LexicalFstImpl.
 */
class AbstractLexicalFstImpl
{
public:
    virtual ~AbstractLexicalFstImpl() {}
    virtual bool load(const std::string &filename) = 0;
    virtual bool create(const OpenFst::VectorFst &src) = 0;
};

/**
 * Manages loading, converting, and deleting of the underlying fst.
 * The LexicalFstImpl object is stored in the AbstractLexicalFst,
 * which forwards the calls to load() and create().
 * fst_ is a pointer to the parent's member, such that the parent
 * can use the correct pointer type without casting.
 */
template<class M>
class LexicalFstImpl : public AbstractLexicalFstImpl
{
public:
    typedef M Fst;

    LexicalFstImpl(M **fst) : fst_(fst)  {}
    virtual ~LexicalFstImpl() { delete *fst_; }
    bool load(const std::string &filename) {
        *fst_ = M::Read(filename);
        return *fst_;
    }
    bool create(const OpenFst::VectorFst &src) {
        *fst_ = new M(src);
        return *fst_;
    }
protected:
    M **fst_;
};

// ===============================================================

class AbstractStateTable;

/**
 * Interface for C o L transducer.
 * The actual type of the underlying transducer depends on the type of
 * label/weight look-ahead and the accumulator used for weight pushing.
 * All of the parameters are template arguments in OpenFst, therefore a
 * polymorphic wrapper is required.
 *
 * The derived classes take care of loading or creating a MatcherFst,
 * relabeling the G transducer (if required) and
 * creating the ComposeFst (using ComposeFstFactory)
 */
class AbstractLexicalFst
{
protected:
    typedef FstLib::StdFst StdFst;
    typedef StdFst::Arc Arc;
public:
    typedef FstLib::ComposeFst<Arc> ComposeFst;

    AbstractLexicalFst() : impl_(0) {}
    virtual ~AbstractLexicalFst() {
        delete impl_;
    }
    virtual bool load(const std::string &filename) {
        return impl_->load(filename);
    }
    virtual bool create(const OpenFst::VectorFst &src) {
        return impl_->create(src);
    }
    virtual void relabel(AbstractGrammarFst *g) const = 0;
    virtual ComposeFst* compose(
        const AbstractGrammarFst &g, size_t cacheSize,
        AbstractStateTable **stateTable) const = 0;

protected:
    void SetImpl(AbstractLexicalFstImpl *impl) { impl_ = impl; }
    AbstractLexicalFstImpl *impl_;
};

// ===============================================================

/**
 * standard fst without any look-ahead
 */
class StandardLexicalFst : public AbstractLexicalFst
{
    typedef AbstractLexicalFst::StdFst StdFst;
    typedef FstLib::StdVectorFst VectorFst;
    typedef LexicalFstImpl<VectorFst> Impl;

public:
    typedef FstLib::SortedMatcher<StdFst> Matcher;
    typedef Impl::Fst Fst;

    static const LookAheadType FilterType = NoLookAhead;

    const Fst &getFst() const { return *fst_; }

    StandardLexicalFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {}

    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(
                *this, g, cacheSize, stateTable);
    }

protected:
    Fst *fst_;
};

/**
 * Provides the type of the MatcherFst for the given Accumulator A
 * and the matcher flags.
 * The MatcherFst is used for the L transducer in all LexicalFst's using
 * a label-lookahead. Depending on whether weight pushing is used, the
 * matcher flags have to be changed.
 */
template<class A, uint32 flags = FstLib::olabel_lookahead_flags>
struct MatcherFstSelector
{
    typedef A Accumulator;
    typedef typename Accumulator::Arc Arc;
    typedef FstLib::ConstFst<Arc> F;
    typedef FstLib::SortedMatcher<F> M;
    typedef FstLib::LabelLookAheadMatcher<M, flags, Accumulator> Matcher;
    typedef FstLib::LabelLookAheadRelabeler<Arc> Relabeler;
    typedef FstLib::MatcherFst<
            F, Matcher,
            FstLib::olabel_lookahead_fst_type,
            Relabeler> MatcherFst;
};


/**
 * Produces a ComposeFst using a label look-ahead composition filter.
 */
// template<class R>
class LookAheadLexicalFst : public AbstractLexicalFst
{
    typedef AbstractLexicalFst::StdFst StdFst;
    typedef StdFst::Arc Arc;
    typedef FstLib::DefaultAccumulator<Arc> Accumulator;

    static const uint32 LookAheadFlags = FstLib::olabel_lookahead_flags &
            ~(FstLib::kLookAheadWeight | FstLib::kLookAheadPrefix);
    typedef MatcherFstSelector<Accumulator, LookAheadFlags> Selector;
    typedef Selector::MatcherFst MatcherFst;
    typedef LexicalFstImpl<MatcherFst> Impl;
public:
    typedef Selector::Matcher Matcher;
    typedef Impl::Fst Fst;

    static const LookAheadType FilterType = LabelLookAhead;

    const Fst &getFst() const { return *fst_; }

    LookAheadLexicalFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {
        g->relabel(GrammarRelabeler<MatcherFst>(*fst_));
    }
    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(*this, g, cacheSize, stateTable);
    }

protected:
    Fst *fst_;
};


/**
 * Produces a ComposeFst using a weight pushing (label look-ahead)
 * composition filter.
 */
template<class A>
class PushWeightsLexicalFst : public AbstractLexicalFst
{
    typedef A Accumulator;

    static const uint32 LookAheadFlags = FstLib::olabel_lookahead_flags &
            (~FstLib::kLookAheadPrefix);
    typedef MatcherFstSelector<Accumulator, LookAheadFlags> Selector;
    typedef typename Selector::MatcherFst MatcherFst;
    typedef LexicalFstImpl<MatcherFst> Impl;
public:
    typedef typename Selector::Matcher Matcher;
    typedef typename Impl::Fst Fst;

    static const LookAheadType FilterType = PushWeights;

    const Fst &getFst() const { return *fst_; }

    PushWeightsLexicalFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {
        g->relabel(GrammarRelabeler<MatcherFst>(*fst_));
    }
    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(*this, g, cacheSize, stateTable);
    }

protected:
    Fst *fst_;
};


/**
 * Produces a ComposeFst using a weight and label pushing (label look-ahead)
 * composition filter.
 */
template<class A>
class PushLabelsLexicalFst : public AbstractLexicalFst
{
    typedef A Accumulator;

    static const uint32 LookAheadFlags = FstLib::olabel_lookahead_flags;
    typedef MatcherFstSelector<Accumulator, LookAheadFlags> Selector;
    typedef typename Selector::MatcherFst MatcherFst;
    typedef LexicalFstImpl<MatcherFst> Impl;
public:
    typedef typename Selector::Matcher Matcher;
    typedef typename Impl::Fst Fst;

    static const LookAheadType FilterType = PushLabels;

    const Fst &getFst() const { return *fst_; }

    PushLabelsLexicalFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {
        g->relabel(GrammarRelabeler<MatcherFst>(*fst_));
    }
    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(*this, g, cacheSize, stateTable);
    }
protected:
    Fst *fst_;
};

/**
 * Produces a ComposeFst using a label pushing (label look-ahead)
 * composition filter (without weight pushing).
 */
class PushLabelsOnlyLexicalFst : public AbstractLexicalFst
{
    typedef FstLib::DefaultAccumulator<FstLib::StdArc> Accumulator;

    static const uint32 LookAheadFlags = FstLib::olabel_lookahead_flags
            & (~FstLib::kLookAheadWeight);
    typedef MatcherFstSelector<Accumulator, LookAheadFlags> Selector;
    typedef Selector::MatcherFst MatcherFst;
    typedef LexicalFstImpl<MatcherFst> Impl;
public:
    typedef Selector::Matcher Matcher;
    typedef Impl::Fst Fst;

    static const LookAheadType FilterType = PushLabelsOnly;

    const Fst &getFst() const { return *fst_; }

    PushLabelsOnlyLexicalFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {
        g->relabel(GrammarRelabeler<MatcherFst>(*fst_));
    }
    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(*this, g, cacheSize, stateTable);
    }
protected:
    Fst *fst_;
};


class ArcLookAheadFst : public AbstractLexicalFst
{
protected:
    typedef AbstractLexicalFst::StdFst StdFst;
    typedef StdFst::Arc Arc;

    typedef FstLib::StdArcLookAheadFst MatcherFst;
    typedef LexicalFstImpl<MatcherFst> Impl;
public:
    typedef MatcherFst::FstMatcher Matcher;
    typedef Impl::Fst Fst;

    static const LookAheadType FilterType = ArcLookAhead;

    const Fst &getFst() const { return *fst_; }

    ArcLookAheadFst() { SetImpl(new Impl(&fst_)); }
    void relabel(AbstractGrammarFst *g) const {}
    ComposeFst* compose(const AbstractGrammarFst &g, size_t cacheSize,
                        AbstractStateTable **stateTable) const {
        return ComposeFstFactory::create(*this, g, cacheSize, stateTable);
    }

protected:
    Fst *fst_;
};

} // namespace Wfst
} // namespace Search

#endif  // SEARCH_WFST_LEXICONFST_HH
