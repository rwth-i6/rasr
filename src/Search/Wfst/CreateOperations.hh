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
#ifndef _SEARCH_CREATE_OPERATIONS_HH
#define _SEARCH_CREATE_OPERATIONS_HH

#include <unordered_set>

#include <Search/Wfst/Builder.hh>
#include <Search/Wfst/StateSequence.hh>

namespace Search { namespace Wfst { namespace Builder {

/**
 * build the language model transducer
 */
class BuildGrammar : public Operation, public OutputTypeDependent
{
protected:
    static const Core::ParameterBool paramAddEmptySyntacticTokens;
    static const Core::ParameterBool paramAddSentenceBoundaries;
    static const Core::ParameterBool paramAddSentenceBegin;
    static const Core::ParameterBool paramAddSentenceEnd;
public:
    BuildGrammar(const Core::Configuration &c, Resources &r) :
        Operation(c, r), OutputTypeDependent(c) {}
protected:
    virtual AutomatonRef process();
    void addSentenceBoundaries(Core::Ref<Fsa::StaticAutomaton> g,
                               Fsa::LabelId sb, Fsa::LabelId se,
                               bool addSb, bool addSe) const;
    AutomatonRef mapSymbols(AutomatonRef g) const;
public:
    static std::string name() { return "build-g"; }
};

/**
 * build a transducer for the lexicon (phoneme to outputType() mapping).
 */
class BuildLexicon : public DisambiguatorDependentOperation, public OutputTypeDependent
{
protected:
    static const Core::ParameterBool paramCloseLexicon;
public:
    static const Core::ParameterBool paramCloseWithSilence;
public:
    BuildLexicon(const Core::Configuration &c, Resources &r) :
        Operation(c, r), DisambiguatorDependentOperation(c, r), OutputTypeDependent(c) {}
    virtual bool consumeInput() const { return false; }

    static const char* attrInitialPhoneOffset;
    static const char* attrWordLabelOffset;
    static const char* attrDisambiguatorOffset;
protected:
    virtual AutomatonRef process();
    void mapOutputSymbols(OpenFst::VectorFst *l) const;
public:
    static std::string name() { return "build-l"; }
};

/**
 * lexicon construction using the old Bliss code.
 * Does not support input encoded word labels.
 */
class BuildOldLexicon : public BuildLexicon
{
public:
    BuildOldLexicon(const Core::Configuration &c, Resources &r) :
        Operation(c, r), BuildLexicon(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "build-old-l"; }
};

/**
 * build the closure of the lexicon transducer
 */
class CloseLexicon : public SleeveOperation
{
public:
    CloseLexicon(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "close-l"; }
};

class LemmaMapping : public DisambiguatorDependentOperation
{
protected:
    enum MapType { LemmaPronunciationToLemma, LemmaToSyntacticToken };
    static const Core::Choice mapChoice;
    static const Core::ParameterChoice paramMapType;
    static const Core::ParameterFloat paramScale;
public:
    LemmaMapping(const Core::Configuration &c, Resources &r) :
        Operation(c, r), DisambiguatorDependentOperation(c, r) {}
    virtual u32 nInputAutomata() const { return 0; }
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "map-lemma"; }
};

class AddPronunciationWeight : public DisambiguatorDependentOperation
{
protected:
    static const Core::ParameterFloat paramScale;
public:
    AddPronunciationWeight(const Core::Configuration &c, Resources &r) :
        Operation(c, r), DisambiguatorDependentOperation(c, r) {}
    virtual u32 nInputAutomata() const { return 0; }
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "add-pron-weight"; }
};

/**
 * restore output labels encoded as input symbols
 */
class RestoreOutputSymbols : public SleeveOperation
{
public:
    RestoreOutputSymbols(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "restore-output"; }
};

/**
 * Replace disambiguator symbols with epsilon
 */
class RemovePhoneDisambiguators :
    public DisambiguatorDependentOperation, public SleeveOperation
{
public:
    RemovePhoneDisambiguators(const Core::Configuration &c, Resources &r) :
        Operation(c, r), DisambiguatorDependentOperation(c, r),
        SleeveOperation(c, r) {}
protected:
    virtual u32 nInputAutomata() const { return SleeveOperation::nInputAutomata(); }
    virtual bool addInput(AutomatonRef f);
    virtual AutomatonRef process();
public:
    static std::string name() { return "remove-disambiguators"; }
};


/**
 * move eps:<word> labels such that the output occurs on the
 * next non-epsilon arc.
 */
class PushOutputLabels : public SleeveOperation
{
public:
    PushOutputLabels(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "push-output"; }
};

class CheckLabels : public SleeveOperation
{
    static const Core::ParameterString paramStateSequences;
public:
    CheckLabels(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "check-labels"; }
};


/**
 * adds non-word tokens (silence/noise) to the grammar transducer.
 * loop arcs are added either to all states (parmAllStates == true),
 * to the initial state, to the final state (assuming only one final state),
 * or to the unigram state.
 */
class AddNonWordTokens : public OutputTypeDependent, public SleeveOperation
{
    static const Core::ParameterFloat paramWeight;
    static const Core::ParameterStringVector paramNonWordLemmas;
    static const Core::ParameterBool paramAllStates;
    static const Core::ParameterBool paramInitialState;
    static const Core::ParameterBool paramFinalState;
    static const Core::ParameterBool paramUnigramState;
    static const Core::ParameterBool paramRenormalize;
public:
    AddNonWordTokens(const Core::Configuration &c, Resources &r) :
        Operation(c, r), OutputTypeDependent(c), SleeveOperation(c, r),
        renormalize_(paramRenormalize(Operation::config)) {}
    virtual u32 nInputAutomata() const { return SleeveOperation::nInputAutomata(); }
protected:
    virtual AutomatonRef process();
private:
    void addArcs(OpenFst::StateId s, f32 weight, const std::vector<OpenFst::Label> &labels);
    void getLabels(const std::vector<std::string> &lemmas,
                   std::vector<OpenFst::Label> &labels) const;
    OpenFst::Label getLabel(const Bliss::Lemma *lemma) const;
    OpenFst::StateId getFinalState() const;
    OpenFst::StateId getUnigramState() const;
    void renormalizeWeights(OpenFst::StateId s);
    bool renormalize_;
public:
    static std::string name() { return "add-non-word-tokens"; }
};

/**
 * Modify the G transducer such that at only paths with a least
 * non-epsilon label are successful.
 */
class RemoveEmptyPath : public SleeveOperation
{
public:
    RemoveEmptyPath(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "remove-empty-path"; }
};

/**
 * Modify a subword LM transducer such that it is composable
 * with a regular (fullword) G transducer.
 * Adds loop transitions to a new initial state for all non subword tokens.
 */
class CreateSubwordGrammar : public OutputTypeDependent, public SleeveOperation
{
    static const Core::ParameterString paramSubwordList;
    static const Core::ParameterString paramTransitionSymbol;
public:
    CreateSubwordGrammar(const Core::Configuration &c, Resources &r) :
        Operation(c, r), OutputTypeDependent(c), SleeveOperation(c, r) {}
protected:
    virtual AutomatonRef process();
    bool readSubwordList(const std::string &filename);
    bool addLemma(Fsa::LabelId synt);
    bool addLemmaPronunciation(Fsa::LabelId lemma);
    std::unordered_set<Fsa::LabelId> subwordTokens_;
public:
    static std::string name() { return "create-subword-g"; }
};

/**
 * create allophone (triphone) to phoneme mapping
 */
class ContextBuilder : public SleeveOperation
{
public:
    ContextBuilder(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r) {}
    bool consumeInput() const { return false; }

protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "build-c"; }
};

/**
 * build allophone to allophone state sequence mapping.
 * not required for ExpandingFsaSearch
 */
class HmmBuilder : public DisambiguatorDependentOperation
{
public:
    HmmBuilder(const Core::Configuration &c, Resources &r) :
        Operation(c, r), DisambiguatorDependentOperation(c, r) {}
    virtual bool consumeInput() const { return false; }
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "build-h"; }
};

/**
 * replace the allophone index labels by an index of
 * its (tied) allophones state sequence.
 */
class CreateStateSequences : public SleeveOperation
{
protected:
    static const Core::ParameterString paramFilename;
public:
    CreateStateSequences(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r), filename_(paramFilename(config)) {}
protected:
    virtual bool precondition() const;
    virtual AutomatonRef process();
    std::string filename_;
public:
    static std::string name() { return "create-state-sequences"; }
};

class NonWordDependentOperation : public virtual Operation
{
protected:
    static const Core::ParameterBool paramAddNonWords;

public:
    NonWordDependentOperation(const Core::Configuration &c, Resources &r) :
        Operation(c, r), addNonWords_(paramAddNonWords(c)) {}

protected:
    u32 numSpecialSymbols() const;
    bool addNonWords_;
};

/**
 * merge chains of arcs.
 */
class Factorize : public SleeveOperation, public NonWordDependentOperation
{
protected:
    static const Core::ParameterString paramStateSequences;
    static const Core::ParameterString paramNewStateSequences;

public:
    Factorize(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r), NonWordDependentOperation(c, r) {}
protected:
    virtual AutomatonRef process();

public:
    static std::string name() { return "factorize"; }
};

/**
 * Expand HMM arcs to HMM state arcs
 */
class ExpandStates : public SleeveOperation, public NonWordDependentOperation
{
    static const Core::ParameterString paramStateSequences;
    static const Core::ParameterString paramNewStateSequences;
public:
    ExpandStates(const Core::Configuration &c, Resources &r) :
        Operation(c, r), SleeveOperation(c, r), NonWordDependentOperation(c, r) {}
protected:
    virtual AutomatonRef process();
private:
    void expandArc(const OpenFst::Arc &arc, const StateSequence &ss,
                   bool isRegularLabel, TiedStateSequenceMap *sequences,
                   std::vector<StateSequence> *specialSequences,
                   OpenFst::Arc *firstArc);
public:
    static std::string name() { return "expand-states"; }
};

/**
 * convert a hmm list file to a state sequences file.
 * no input automata required. no output automata produced.
 */
class ConvertStateSequences : public Operation
{
protected:
    static const Core::ParameterString paramInput, paramOutput;
    static const Core::ParameterString paramHmmSymbols, paramStateSymbols;
public:
    ConvertStateSequences(const Core::Configuration &c, Resources &r) :
        Operation(c, r) {}
    virtual bool precondition() const;
    virtual u32 nInputAutomata() const { return 0; }
    virtual bool hasOutput() const { return false; }
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "convert-state-sequences"; }
};

/**
 * create a state tree transducer
 */
class BuildStateTree : public Operation
{
    static const Core::ParameterString paramStateSequencesFile;
public:
    BuildStateTree(const Core::Configuration &c, Resources &r) :
        Operation(c, r) {}
protected:
    virtual AutomatonRef process();
public:
    static std::string name() { return "build-state-tree"; }
};


} // namespace Builder
} // namespace Wfst
} // namespace Search

#endif // _SEARCH_CREATE_OPERATIONS_HH
