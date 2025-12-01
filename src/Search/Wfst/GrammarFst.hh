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
#ifndef _SEARCH_COMBINED_GRAMMAR_HH
#define _SEARCH_COMBINED_GRAMMAR_HH

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/DynamicLmFst.hh>
#include <Search/Wfst/Network.hh>
#include <fst/compact-fst.h>
#include <fst/compose-filter.h>
#include <fst/matcher-fst.h>
#include <fst/matcher.h>
#include <fst/project.h>
#include <fst/relabel.h>
#include <fst/replace.h>

namespace OpenFst {
template<class A>
class CompactReplaceFst;
}

namespace Search {
namespace Wfst {

/**
 * Interface for GrammarRelabeler classes
 */
class GrammarRelabelerBase {
public:
    typedef std::vector<std::pair<OpenFst::Label, OpenFst::Label>> LabelMap;
    virtual void                                                   apply(FstLib::StdMutableFst*, bool = true) const = 0;
    virtual void                                                   getMap(LabelMap* map) const                      = 0;
};

/**
 * Functor to apply the relabeling of G for L with output label lookahead.
 */
template<class L>
class GrammarRelabeler : public GrammarRelabelerBase {
public:
    GrammarRelabeler(const L& l)
            : l_(l) {}
    void apply(FstLib::StdMutableFst* g, bool relabelInput = true) const {
        FstLib::LabelLookAheadRelabeler<OpenFst::Arc>::Relabel(g, l_, relabelInput);
    }
    void getMap(LabelMap* map) const {
        map->clear();
        FstLib::LabelLookAheadRelabeler<OpenFst::Arc>::RelabelPairs(l_, map);
    }

protected:
    const L& l_;
};

/**
 * Interface for G transducers used in ComposedNetwork
 */
class AbstractGrammarFst {
public:
    enum GrammarType {
        TypeAny,
        TypeVector,
        TypeConst,
        TypeCompact,
        TypeCombine,
        TypeCompose,
        TypeDynamic,
        TypeFailArc
    };

    AbstractGrammarFst() {}
    virtual ~AbstractGrammarFst() {}
    virtual void                  setLexicon(Bliss::LexiconRef lexicon) {}
    virtual bool                  load(const std::string& main) = 0;
    virtual void                  reset() {}
    virtual const FstLib::StdFst* getFst() const = 0;
    virtual void                  relabel(const GrammarRelabelerBase& relabeler) {}
    virtual GrammarType           type() const = 0;
    /**
     * factory function
     */
    static AbstractGrammarFst* create(GrammarType type, const Core::Configuration& c);
};

/**
 * Template for grammar fsts using some constant OpenFst class as underlying type.
 */
template<class F, AbstractGrammarFst::GrammarType t>
class GrammarFstTpl : public AbstractGrammarFst {
public:
    GrammarFstTpl()
            : fst_(0) {}
    GrammarFstTpl(const F& other)
            : fst_(other.Copy()) {}
    virtual ~GrammarFstTpl() {
        delete fst_;
    }
    bool load(const std::string& filename) {
        fst_ = F::Read(filename);
        return fst_;
    }
    const FstLib::StdFst* getFst() const {
        return fst_;
    }
    GrammarType type() const {
        return t;
    }

protected:
    F* fst_;
};

/**
 * G transducer using StdVectorFst
 */
class GrammarFst : public GrammarFstTpl<FstLib::StdVectorFst, AbstractGrammarFst::TypeVector> {
    typedef GrammarFstTpl<FstLib::StdVectorFst, AbstractGrammarFst::TypeVector> Parent;

public:
    GrammarFst() {}
    GrammarFst(const FstLib::StdVectorFst& f)
            : Parent(f) {}
    void relabel(const GrammarRelabelerBase& relabeler);
};

class FailArcGrammarFst : public GrammarFst {
    typedef FstLib::SortedMatcher<FstLib::Fst<FstLib::StdArc>> M;

public:
    typedef FstLib::PhiMatcher<M> Matcher;

    FailArcGrammarFst() {}
    virtual ~FailArcGrammarFst() {};
    bool        load(const std::string& filename);
    GrammarType type() const {
        return TypeFailArc;
    }
    static const OpenFst::Label FailLabel;
};

typedef GrammarFstTpl<FstLib::StdCompactAcceptorFst,
                      AbstractGrammarFst::TypeCompact>
        CompactGrammarFst;

typedef GrammarFstTpl<FstLib::StdConstFst,
                      AbstractGrammarFst::TypeConst>
        ConstGrammarFst;

/**
 * G transducer consisting of two or more transducers which are combined using
 * a ReplaceFst.
 * Arcs with special labels in the main G transducer (e.g. [UNKNOWN])
 * are replaced on the fly by a separate LM transducer.
 */
class CombinedGrammarFst : public AbstractGrammarFst, public Core::Component {
public:
    CombinedGrammarFst(const Core::Configuration& c)
            : Core::Component(c),
              fst_(0),
              rootFst_(0) {}
    virtual ~CombinedGrammarFst();

    bool                  load(const std::string& root);
    void                  relabel(const GrammarRelabelerBase& relabeler);
    const FstLib::StdFst* getFst() const;
    void                  reset();

    GrammarType type() const {
        return TypeCombine;
    }

protected:
    static const Core::ParameterInt             paramCacheSize;
    static const Core::ParameterStringVector    paramReplaceLabels;
    static const Core::ParameterIntVector       paramReplaceIds;
    static const Core::ParameterFloatVector     paramAddOnScales;
    static const Core::ParameterStringVector    paramAddOnFiles;
    void                                        replaceArcLabels();
    OpenFst::CompactReplaceFst<FstLib::StdArc>* fst_;
    OpenFst::VectorFst*                         rootFst_;
    std::vector<OpenFst::VectorFst*>            addOnFsts_;
    std::vector<OpenFst::Label>                 replaceLabels_;
};

class ComposedGrammarFst : public AbstractGrammarFst, public Core::Component {
    typedef FstLib::StdProjectFst                                                 ProjectFst;
    typedef FstLib::StdComposeFst                                                 ComposeFst;
    typedef FstLib::RelabelFst<FstLib::StdArc>                                    RelabelFst;
    typedef FstLib::Matcher<FstLib::StdFst>                                       Matcher;
    typedef FstLib::AltSequenceComposeFilter<Matcher>                             Filter;
    typedef FstLib::GenericComposeStateTable<FstLib::StdArc, Filter::FilterState> StateTable;

public:
    ComposedGrammarFst(const Core::Configuration& c)
            : Core::Component(c),
              cfst_(0),
              pfst_(0),
              rfst_(0),
              table_(0),
              rootFst_(0),
              addOnFst_(0) {}
    virtual ~ComposedGrammarFst();
    bool                  load(const std::string& root);
    void                  relabel(const GrammarRelabelerBase& relabeler);
    const FstLib::StdFst* getFst() const {
        return rfst_;
    }
    void        reset();
    GrammarType type() const {
        return TypeCompose;
    }

protected:
    static const Core::ParameterInt    paramCacheSize;
    static const Core::ParameterString paramAddOnFile;
    static const Core::ParameterFloat  paramAddOnScale;
    GrammarRelabelerBase::LabelMap     iLabelMap_, oLabelMap_;
    ComposeFst*                        cfst_;
    ProjectFst*                        pfst_;
    RelabelFst*                        rfst_;
    StateTable*                        table_;
    OpenFst::VectorFst *               rootFst_, *addOnFst_;
};

class DynamicGrammarFst : public AbstractGrammarFst, public Core::Component {
public:
    DynamicGrammarFst(const Core::Configuration& c)
            : Core::Component(c),
              fst_(0) {}
    virtual ~DynamicGrammarFst();
    void setLexicon(Bliss::LexiconRef lexicon) {
        lexicon_ = lexicon;
    }
    bool                  load(const std::string&);
    void                  relabel(const GrammarRelabelerBase& relabeler);
    const FstLib::StdFst* getFst() const {
        return fst_;
    }
    void        reset();
    GrammarType type() const {
        return TypeDynamic;
    }

protected:
    static const Core::ParameterBool  paramLemma;
    static const Core::ParameterFloat paramPronunciationScale;
    DynamicLmFst*                     fst_;
    Bliss::LexiconRef                 lexicon_;
    Core::Ref<Lm::LanguageModel>      lm_;
    GrammarRelabelerBase::LabelMap    labelMap_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_COMBINED_GRAMMAR_HH
