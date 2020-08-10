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
#ifndef _SEARCH_FST_OPERATIONS_HH
#define _SEARCH_FST_OPERATIONS_HH

#include <Search/Wfst/Builder.hh>
#include <Search/Wfst/IoOperations.hh>

namespace Search {
namespace Wfst {
namespace Builder {

/**
 * minimize weighted transducer
 */
class Minimize : public SleeveOperation, public SemiringDependent {
    static const Core::ParameterBool paramEncodeWeights;
    static const Core::ParameterBool paramEncodeLabels;

public:
    Minimize(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), SemiringDependent(c) {}

protected:
    virtual AutomatonRef process();

private:
    template<class A>
    void minimize(FstLib::VectorFst<A>* a, uint32 encodeFlags) const;

public:
    static std::string name() {
        return "minimize";
    }
};

/**
 * determinize weighted transducer
 */
class Determinize : public SleeveOperation, public SemiringDependent {
public:
    Determinize(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), SemiringDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "determinize";
    }
};

/**
 * sort arcs by input label.
 */
class ArcInputSort : public SleeveOperation {
public:
    ArcInputSort(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "sort-input";
    }
};

/**
 * sort arcs by output label.
 */
class ArcOutputSort : public SleeveOperation {
public:
    ArcOutputSort(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "sort-output";
    }
};

/**
 * weighted transducer composition
 */
class Compose : public SleeveOperation {
protected:
    static const Core::ParameterBool paramIgnoreSymbols;
    static const Core::ParameterBool paramSwap;

public:
    Compose(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), right_(0) {}
    virtual u32 nInputAutomata() const {
        return 2;
    }
    virtual bool addInput(AutomatonRef f);

protected:
    virtual bool         precondition() const;
    virtual AutomatonRef process();
    AutomatonRef         right_;

public:
    static std::string name() {
        return "compose";
    }
};

/**
 * base class for LabelEncode, LabelDecode
 */
class LabelCoding : public SleeveOperation {
protected:
    static const Core::ParameterString paramEncoder;

public:
    LabelCoding(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}
};

/**
 * combine input and output labels.
 * required if the final network is non functional, which
 * happens when the allophone states are tied.
 */
class LabelEncode : public LabelCoding {
    static const Core::ParameterBool paramProtectEpsilon;

public:
    LabelEncode(const Core::Configuration& c, Resources& r)
            : Operation(c, r), LabelCoding(c, r) {}

protected:
    virtual AutomatonRef process();
    void                 encode(uint32 flags);

private:
    template<class M>
    void applyMappping(M* mapper);
    template<class M>
    void writeMapping(M* mapper) const;

public:
    static std::string name() {
        return "encode";
    }
};

/**
 * split combined input/output labels to regular
 * input and output labels.
 */
class LabelDecode : public LabelCoding {
public:
    LabelDecode(const Core::Configuration& c, Resources& r)
            : Operation(c, r), LabelCoding(c, r) {}

protected:
    virtual AutomatonRef process();
    void                 decode();

public:
    static std::string name() {
        return "decode";
    }
};

/**
 * Combine input label and weight.
 * Results in an unweighted transducer.
 */
class WeightEncode : public LabelEncode {
public:
    WeightEncode(const Core::Configuration& c, Resources& r)
            : Operation(c, r), LabelEncode(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "weight-encode";
    }
};

/**
 * relabeling
 */
class Relabel : public SleeveOperation {
    static const Core::ParameterStringVector paramInputMapping;
    static const Core::ParameterStringVector paramOutputMapping;

public:
    Relabel(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

private:
    typedef std::pair<OpenFst::Label, OpenFst::Label> LabelPair;
    typedef std::vector<LabelPair>                    LabelMapping;
    void                                              getLabelMapping(const std::vector<std::string>& labels,
                                                                      const OpenFst::SymbolTable*     symbols,
                                                                      LabelMapping*                   mapping) const;

public:
    static std::string name() {
        return "relabel";
    }
};
/**
 * push weights to intial state
 */
class PushWeights : public SleeveOperation, public SemiringDependent {
public:
    PushWeights(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), SemiringDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "push-weights";
    }
};

/**
 * push weights to intial state
 */
class PushLabels : public SleeveOperation {
    static const Core::ParameterBool paramToFinal;

public:
    PushLabels(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "push-labels";
    }
};

/**
 * produce epsilon-normalized automaton
 */
class NormalizeEpsilon : public SleeveOperation, public LabelTypeDependent {
public:
    NormalizeEpsilon(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), LabelTypeDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "epsilon-normalize";
    }
};

/**
 * Project to input or output labels
 */
class Project : public SleeveOperation, public LabelTypeDependent {
public:
    Project(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), LabelTypeDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "project";
    }
};

/**
 * remove epsilon arcs
 */
class RemoveEpsilon : public SleeveOperation, public SemiringDependent

{
public:
    RemoveEpsilon(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), SemiringDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "remove-epsilon";
    }
};

/**
 * synchronize an automaton
 */
class Synchronize : public SleeveOperation, public SemiringDependent {
public:
    Synchronize(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), SemiringDependent(c) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "synchronize";
    }
};

/**
 * invert a transduction
 */
class Invert : public SleeveOperation {
public:
    Invert(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "invert";
    }
};

/**
 * converts the input automaton to an FstLib::StdOLabelLookAheadFst
 * and writes it do disk.
 * optionally writes the relabeling (paramRelabelFile).
 * optionally relabels the second input automaton.
 */
class CreateLookahead : public WriteOperation {
protected:
    static const Core::ParameterString paramRelabelFile;
    static const Core::ParameterBool   paramRelabelInput;
    static const Core::ParameterBool   paramSwap;
    static const Core::ParameterBool   paramKeepRelabelingData;

public:
    CreateLookahead(const Core::Configuration& c, Resources& r)
            : Operation(c, r), WriteOperation(c, r), toRelabel_(0) {}
    virtual bool consumeInput() const {
        return false;
    }
    virtual bool hasOutput() const {
        return false;
    }
    virtual u32  nInputAutomata() const;
    virtual bool addInput(AutomatonRef);

protected:
    virtual AutomatonRef process();
    AutomatonRef         toRelabel_;

private:
    template<class F>
    void relabel(const F& f);

public:
    static std::string name() {
        return "create-lookahead";
    }
};

/**
 * applies composition using a label lookahead matcher.
 */
class ReachableCompose : public Compose {
    enum LookAheadType { LabelLookAhead,
                         ArcLookAhead };
    static const Core::Choice          lookAheadChoice;
    static const Core::ParameterChoice paramLookAheadType;

public:
    ReachableCompose(const Core::Configuration& c, Resources& r)
            : Operation(c, r), Compose(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "reachable-compose";
    }
};

/**
 * scale weights of the automaton
 */
class ScaleWeights : public SleeveOperation {
    static const Core::ParameterFloat paramScale;

public:
    ScaleWeights(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "scale-weights";
    }
};

/**
 * scale weights of arcs with a specific label.
 */
class ScaleLabelWeights : public SleeveOperation {
    static const Core::ParameterFloat  paramScale;
    static const Core::ParameterString paramLabel;

public:
    ScaleLabelWeights(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "scale-label-weights";
    }
};

/**
 * map all weights to Weight::One()
 */
class RemoveWeights : public SleeveOperation {
public:
    RemoveWeights(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "remove-weights";
    }
};

}  // namespace Builder
}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_FST_OPERATIONS_HH
