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
#ifndef _SEARCH_FSA_SEARCH_BUILDER_HH
#define _SEARCH_FSA_SEARCH_BUILDER_HH

#include <Am/AcousticModel.hh>
#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <OpenFst/Types.hh>
#include <Speech/ModelCombination.hh>
#include <fst/fst.h>

namespace Search {
namespace Wfst {
namespace Builder {

/**
 * models (am, lm, lexicon)
 */
class Resources : public Core::Component {
public:
    Resources(const Core::Configuration& c);
    ~Resources();
    Core::Ref<const Bliss::Lexicon>          lexicon() const;
    Core::Ref<const Lm::ScaledLanguageModel> languageModel() const;
    void                                     deleteLanguageModel() const;
    Core::Ref<const Am::AcousticModel>       acousticModel() const;
    Mm::Score                                pronunciationScale() const;

private:
    mutable Speech::ModelCombination* models_;
};

/**
 * OpenFst::VectorFst with additional clone() methods
 */
class Automaton : public OpenFst::VectorFst {
public:
    Automaton() {}
    Automaton(const OpenFst::VectorFst& f)
            : OpenFst::VectorFst(f) {}
    Automaton(const FstLib::Fst<OpenFst::Arc>& f)
            : OpenFst::VectorFst(f) {}
    Automaton(const Automaton& f)
            : OpenFst::VectorFst(f) {}
    Automaton& operator=(const Automaton& f) {
        OpenFst::VectorFst::operator=(f);
        return *this;
    }
    virtual Automaton* clone() const {
        return new Automaton;
    }
    virtual Automaton* cloneWithAttributes() const;
    void               copyAttributes(Automaton* dest) const;
    void               copyAttribute(Automaton* dest, const std::string& name) const;

    std::string getAttribute(const std::string& name) const;
    int         getIntAttribute(const std::string& name) const;
    void        setAttribute(const std::string& name, const std::string& value);
    void        setAttribute(const std::string& name, int value);

    static const int InvalidIntAttribute;

    static const char* attrNumDisambiguators;

protected:
    typedef std::map<std::string, std::string> Attributes;
    Attributes                                 attributes_;
};

/**
 * abstract base class for all operations
 */
class Operation : public Core::Component {
public:
    typedef Automaton* AutomatonRef;

public:
    Operation(const Core::Configuration& c, Resources& r)
            : Core::Component(c), resources_(r), timerChannel_(config, "time") {}
    /**
     * number of automata required as input
     */
    virtual u32 nInputAutomata() const {
        return 0;
    }
    /**
     * should input automata be removed
     */
    virtual bool consumeInput() const {
        return true;
    }
    /**
     * operation is expected to produce an output automaton.
     * i.e. getResult() != 0
     */
    virtual bool hasOutput() const {
        return true;
    }
    /**
     * add input automaton (will be called nInputAutomata() times)
     */
    virtual bool addInput(AutomatonRef) {
        return true;
    }
    /**
     * do the actual processing and create an output automaton
     */
    AutomatonRef getResult();

protected:
    /**
     * precondition for getResult()
     */
    virtual bool precondition() const {
        return true;
    }
    /**
     * do the actual processing
     */
    virtual AutomatonRef process() = 0;

protected:
    Resources&       resources_;
    Core::XmlChannel timerChannel_;
    Core::Timer      timer_;
};

/**
 * base class for operations with 1 input and 1 output automaton
 */
class SleeveOperation : public virtual Operation {
public:
    SleeveOperation(const Core::Configuration& c, Resources& r)
            : Operation(c, r), input_(0) {}
    virtual u32 nInputAutomata() const {
        return 1;
    }
    virtual bool addInput(AutomatonRef f) {
        if (input_)
            return false;
        input_ = f;
        return true;
    }
    void deleteInput() {
        delete input_;
        input_ = 0;
    }

protected:
    virtual bool precondition() const {
        return input_;
    }
    AutomatonRef input_;
};

/**
 * parameter to set the output type
 */
class OutputTypeDependent : private Core::Configurable {
protected:
    enum OutputType {
        outputLemmaPronunciations,
        outputLemmas,
        outputSyntacticTokens
    };
    static const Core::ParameterChoice paramOutputType;
    static const Core::Choice          choiceOutputType;
    OutputType                         outputType() const {
        return static_cast<OutputType>(paramOutputType(config));
    }

public:
    OutputTypeDependent(const Core::Configuration& c)
            : Core::Configurable(c) {}
};

/**
 * parameter to choose the semiring used
 */
class SemiringDependent : private Core::Configurable {
protected:
    static const Core::Choice          choiceSemiring;
    static const Core::ParameterChoice paramSemiring;

public:
    enum SemiringType {
        tropicalSemiring,
        logSemiring
    };
    SemiringType semiring() const {
        return static_cast<SemiringType>(paramSemiring(config));
    }

public:
    SemiringDependent(const Core::Configuration& c)
            : Core::Configurable(c) {}
};

/**
 * parameter to choose input or output labels
 */
class LabelTypeDependent : private Core::Configurable {
protected:
    static const Core::Choice          choiceLabel;
    static const Core::ParameterChoice paramLabel;

public:
    LabelTypeDependent(const Core::Configuration& c)
            : Core::Configurable(c) {}
    enum LabelType { Input,
                     Output };
    LabelType labelType() const {
        return static_cast<LabelType>(paramLabel(config));
    }
};

/**
 * either paramDisambiguators is set or an AutomatonWithDisambiguator
 * is used as input automaton
 */
class DisambiguatorDependentOperation : public virtual Operation {
protected:
    static const Core::ParameterInt paramDisambiguators;
    s32                             nDisambiguators_;

public:
    DisambiguatorDependentOperation(const Core::Configuration& c, Resources& r)
            : Operation(c, r), nDisambiguators_(paramDisambiguators(config)) {}
    virtual u32  nInputAutomata() const;
    virtual bool addInput(AutomatonRef f);
};

/**
 * remove one automaton from the stack
 */
class Pop : public SleeveOperation {
public:
    Pop(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}
    virtual bool hasOutput() const {
        return false;
    }

protected:
    virtual AutomatonRef process() {
        return 0;
    }

public:
    static std::string name() {
        return "pop";
    }
};

}  // namespace Builder
}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_FSA_SEARCH_BUILDER_HH */
