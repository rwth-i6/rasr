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
#include <Search/Wfst/FstOperations.hh>
#include <OpenFst/Scale.hh>
#include <OpenFst/Encode.hh>
#include <fst/arc-map.h>
#include <fst/compose.h>
#include <fst/connect.h>
#include <fst/determinize.h>
#include <fst/encode.h>
#include <fst/epsnormalize.h>
#include <fst/matcher-fst.h>
#include <fst/minimize.h>
#include <fst/push.h>
#include <fst/project.h>
#include <fst/relabel.h>
#include <fst/synchronize.h>

using namespace Search::Wfst::Builder;

const Core::ParameterBool Minimize::paramEncodeLabels(
    "encode-labels", "combine labels before minimization", false);
const Core::ParameterBool Minimize::paramEncodeWeights(
    "encode-weights", "combine weights and input label before minimization", false);

Operation::AutomatonRef Minimize::process()
{
    uint32 encodeFlags = 0;
    if (paramEncodeLabels(Operation::config)) {
        log("using encoded labels");
        encodeFlags |= FstLib::kEncodeLabels;
    }
    if (paramEncodeWeights(Operation::config)) {
        log("using encoded weights");
        encodeFlags |= FstLib::kEncodeWeights;
    }
    log("minimizing");
    if (semiring() == tropicalSemiring) {
        log("using tropical semiring");
        minimize(input_, encodeFlags);
    } else {
        log("using log semiring");
        OpenFst::LogVectorFst *l = new OpenFst::LogVectorFst();
        FstLib::Cast(*input_, l);
        input_->DeleteStates();
        minimize(l, encodeFlags);
        FstLib::Cast(*l, input_);
        delete l;
    }
    return input_;
}

template<class A>
void Minimize::minimize(FstLib::VectorFst<A> *a, uint32 encodeFlags) const
{
    typedef FstLib::EncodeMapper<A> Mapper;
    Mapper mapper(encodeFlags, FstLib::ENCODE);
    if (encodeFlags) {
        FstLib::Encode(a, &mapper);
    }
    FstLib::Minimize(a);
    if (encodeFlags) {
        Mapper decodeMapper(mapper, FstLib::DECODE);
        FstLib::Decode(a, decodeMapper);
    }
}

Operation::AutomatonRef Determinize::process()
{
    log("determinizing");
    Automaton *result = input_->cloneWithAttributes();
    if (semiring() == tropicalSemiring) {
        log("using tropical semiring");
        FstLib::Determinize(*input_, result);
        deleteInput();
    } else {
        log("using log semiring");
        OpenFst::LogVectorFst *l = new OpenFst::LogVectorFst;
        OpenFst::LogVectorFst *detL = new OpenFst::LogVectorFst;
        FstLib::Cast(*input_, l);
        deleteInput();
        FstLib::Determinize(*l, detL);
        delete l; l = 0;
        FstLib::Cast(*detL, result);
        delete detL;
    }
    return result;
}

Operation::AutomatonRef ArcInputSort::process()
{
    log("sorting arcs by input label");
    FstLib::ArcSort(input_, FstLib::ILabelCompare<OpenFst::Arc>());
    return input_;
}

Operation::AutomatonRef ArcOutputSort::process()
{
    log("sorting arcs by output label");
    FstLib::ArcSort(input_, FstLib::OLabelCompare<OpenFst::Arc>());
    return input_;
}

const Core::ParameterBool Compose::paramIgnoreSymbols(
    "ignore-symbols", "do not check symbol table compatibility", false);

const Core::ParameterBool Compose::paramSwap(
    "swap", "swap order of operands", false);


bool Compose::precondition() const
{
    return SleeveOperation::precondition() && right_;
}

bool Compose::addInput(AutomatonRef f)
{
    if (!input_)
        return SleeveOperation::addInput(f);
    else if (!right_) {
        right_ = f;
        if (paramSwap(config))
            std::swap(input_, right_);
        return true;
    } else
        return false;
}

Operation::AutomatonRef Compose::process()
{
    log("building composition");
    FstLib::ComposeFstOptions<OpenFst::Arc> options;
    options.gc_limit = 0;
    Automaton *result = input_->cloneWithAttributes();
    if (paramIgnoreSymbols(config)) {
        log("ignoring symbols");
        input_->SetOutputSymbols(0);
        right_->SetInputSymbols(0);
    }
    *result = FstLib::ComposeFst<OpenFst::Arc>(*input_, *right_, options);
    deleteInput();
    delete right_; right_ = 0;
    FstLib::Connect(result);
    return result;
}


const Core::ParameterString LabelCoding::paramEncoder(
        "encoder", "filename of the encoder (written by encode, read by decode)", "");

const Core::ParameterBool LabelEncode::paramProtectEpsilon(
        "protect-epsilon", "force epsilon input labels to be mapped to label 0", false);

Operation::AutomatonRef LabelEncode::process()
{
    log("encoding labels");
    /* fixed in openfst 1.2
    if (input_->Properties(FstLib::kInitialCyclic, true) & FstLib::kInitialCyclic) {
        log("automaton has a cycle containing the intial state. adding a new initial state.");
        // Decode fails if the encoded automaton has arcs with an epsilon label
        // an epsilon arc is introduced by Reweight (called by Push (called by Minimize))
        // if kInitialCyclic is true
        OpenFst::StateId s = input_->AddState();
        input_->AddArc(s, OpenFst::Arc(OpenFst::Epsilon, OpenFst::Epsilon, OpenFst::Weight::One(), input_->Start()));
        input_->SetStart(s);
    }
    */
    encode(FstLib::kEncodeLabels);
    return input_;
}

void LabelEncode::encode(uint32 flags)
{

    if (paramProtectEpsilon(Operation::config)) {
        log("protecting epsilon labels");
        OpenFst::EpsilonEncodeMapper<OpenFst::Arc> encoder(flags, FstLib::ENCODE);
        applyMappping(&encoder);
        writeMapping(&encoder);
    } else {
        FstLib::EncodeMapper<OpenFst::Arc> encoder(flags, FstLib::ENCODE);
        applyMappping(&encoder);
        writeMapping(&encoder);
    }
}

template<class M>
void LabelEncode::applyMappping(M *mapper)
{
    mapper->SetInputSymbols(input_->InputSymbols());
    mapper->SetOutputSymbols(input_->OutputSymbols());
    FstLib::ArcMap(input_, mapper);
}

template<class M>
void LabelEncode::writeMapping(M *mapper) const
{
    const std::string encoderFile = paramEncoder(config);
    log("writing encoder '%s'", encoderFile.c_str());
    mapper->Write(encoderFile);
}

Operation::AutomatonRef LabelDecode::process()
{
    const std::string encoderFile = paramEncoder(config);
    log("reading encoder '%s'", encoderFile.c_str());
    FstLib::EncodeMapper<OpenFst::Arc> *encoder = FstLib::EncodeMapper<OpenFst::Arc>::Read(encoderFile);
    uint32 flags = encoder->Flags();
    if (flags & FstLib::kEncodeLabels)
        log("decoding labels");
    if (flags & FstLib::kEncodeWeights)
        log("decoding weights");
    FstLib::Decode(input_, *encoder);
    delete encoder;
    return input_;
}

Operation::AutomatonRef WeightEncode::process()
{
    log("encoding weights");
    encode(FstLib::kEncodeWeights);
    return input_;
}

const Core::ParameterStringVector Relabel::paramInputMapping(
    "input", "input mapping separated by ','", ",");

const Core::ParameterStringVector Relabel::paramOutputMapping(
    "output", "output mapping separated by ','", ",");

Operation::AutomatonRef Relabel::process()
{
    LabelMapping inputMapping;
    LabelMapping outputMapping;
    if (!input_->InputSymbols())
        warning("no input symbols found");
    if (!input_->OutputSymbols())
        warning("no output symbols found");
    getLabelMapping(paramInputMapping(config), input_->InputSymbols(), &inputMapping);
    getLabelMapping(paramOutputMapping(config), input_->OutputSymbols(), &outputMapping);
    log("relabeling using %d input mappings %d output mappings",
        static_cast<int>(inputMapping.size()), static_cast<int>(outputMapping.size()));
    FstLib::Relabel(input_, inputMapping, outputMapping);
    return input_;
}

void Relabel::getLabelMapping(const std::vector<std::string> &labels,
                              const OpenFst::SymbolTable *symbols,
                              LabelMapping *mapping) const
{
    verify(!(labels.size() % 2));
    for (std::vector<std::string>::const_iterator s = labels.begin(); s != labels.end(); s += 2) {
        const std::string from = *s;
        const std::string to = *(s + 1);
        s32 fromId = -1, toId = -1;
        if (symbols) {
            fromId = symbols->Find(from);
            toId = symbols->Find(to);
        }
        if (fromId < 0) {
            Core::strconv(from, fromId);
            warning("interpreting '%s' as '%d'", from.c_str(), fromId);
        }
        if (toId < 0) {
            Core::strconv(to, toId);
            warning("interpreting '%s' as '%d'", to.c_str(), toId);
        }
        log("mapping %d to %d", fromId, toId);
        mapping->push_back(LabelPair(fromId, toId));
    }
}

Operation::AutomatonRef PushWeights::process()
{
    log("pushing weights");
    Automaton *result = input_->cloneWithAttributes();
    if (semiring() == tropicalSemiring) {
        log("using tropical semiring");
        FstLib::Push<FstLib::StdArc, FstLib::REWEIGHT_TO_INITIAL>(*input_, result, FstLib::kPushWeights);
        deleteInput();
    } else {
        log("using log semiring");
        OpenFst::LogVectorFst *l = new OpenFst::LogVectorFst;
        OpenFst::LogVectorFst *r = new OpenFst::LogVectorFst;
        FstLib::Cast(*input_, l);
        deleteInput();
        FstLib::Push<FstLib::LogArc, FstLib::REWEIGHT_TO_INITIAL>(*l, r, FstLib::kPushWeights);
        delete l; l = 0;
        FstLib::Cast(*r, result);
        delete r;
    }
    return result;
}

const Core::ParameterBool PushLabels::paramToFinal(
    "to-final", "push labels ot final states", false);

Operation::AutomatonRef PushLabels::process()
{
    log("pushing labels");
    Automaton *result = input_->cloneWithAttributes();
    if (paramToFinal(config)) {
        log("pushing to final state");
        FstLib::Push<FstLib::StdArc, FstLib::REWEIGHT_TO_FINAL>(*input_, result, FstLib::kPushLabels);
    } else {
        log("pushing to initial state");
        FstLib::Push<FstLib::StdArc, FstLib::REWEIGHT_TO_INITIAL>(*input_, result, FstLib::kPushLabels);
    }
    deleteInput();
    return result;
}

Operation::AutomatonRef NormalizeEpsilon::process()
{
    log("normalizing epsilon arcs");
    FstLib::EpsNormalizeType type = (labelType() == LabelTypeDependent::Input) ?
                                     FstLib::EPS_NORM_INPUT : FstLib::EPS_NORM_OUTPUT;
    log("using %s arcs", (type == FstLib::EPS_NORM_INPUT ? "input" :  "output"));
    Automaton *result = input_->cloneWithAttributes();
    FstLib::EpsNormalize(*input_, result, type);
    deleteInput();
    return result;
}

Operation::AutomatonRef Project::process()
{

    FstLib::ProjectType type = (labelType() == LabelTypeDependent::Input) ?
                                FstLib::PROJECT_INPUT : FstLib::PROJECT_OUTPUT;
    log("projecting to %s", type == FstLib::PROJECT_INPUT ? "input" : "output");
    FstLib::Project(input_, type);
    return input_;
}

Operation::AutomatonRef RemoveEpsilon::process()
{
    log("removing epsilon arcs");
    if (semiring() == tropicalSemiring) {
        log("using tropical semiring");
        FstLib::RmEpsilon(input_);
    } else {
        log("using log semiring");
        OpenFst::LogVectorFst *l = new OpenFst::LogVectorFst;
        FstLib::Cast(*input_, l);
        input_->DeleteStates();
        FstLib::RmEpsilon(l);
        FstLib::Cast(*l, input_);
        delete l; l = 0;
    }
    return input_;
}

Operation::AutomatonRef Synchronize::process()
{
    log("synchronizing");
    Automaton *result = input_->cloneWithAttributes();
    if (semiring() == tropicalSemiring) {
        log("using tropical semiring");
        FstLib::Synchronize(*input_, result);
        deleteInput();
    } else {
        log("using log semiring");
        OpenFst::LogVectorFst *l = new OpenFst::LogVectorFst;
        OpenFst::LogVectorFst *syncL = new OpenFst::LogVectorFst;
        FstLib::Cast(*input_, l);
        deleteInput();
        FstLib::Synchronize(*l, syncL);
        delete l; l = 0;
        FstLib::Cast(*syncL, result);
        delete syncL;
    }
    return result;
}

Operation::AutomatonRef Invert::process()
{
    log("inverting");
    FstLib::Invert(input_);
    return input_;
}

const Core::ParameterBool CreateLookahead::paramRelabelInput(
    "relabel-input", "relabel the second input automaton", false);
const Core::ParameterString CreateLookahead::paramRelabelFile(
    "relabel-filename", "filename to write relabeling pairs", "");
const Core::ParameterBool CreateLookahead::paramSwap(
    "swap", "swap input automata", false);
const Core::ParameterBool CreateLookahead::paramKeepRelabelingData(
    "keep-relabeling", "store relabeling data in the file", false);

u32 CreateLookahead::nInputAutomata() const
{
    return paramRelabelInput(config) ? 2 : 1;
}

bool CreateLookahead::addInput(AutomatonRef f)
{
    if (!input_) {
        return SleeveOperation::addInput(f);
    }
    if (!toRelabel_ && paramRelabelInput(config)) {
        toRelabel_ = f;
        if (paramSwap(config)) {
            log("swap input automata");
            std::swap(input_, toRelabel_);
        }
        return true;
    }
    return false;
}

Operation::AutomatonRef CreateLookahead::process()
{
    u32 flagV = FLAGS_v;
    FLAGS_v = 2;
    std::string flagRelabelOpairs = FLAGS_save_relabel_opairs;
    std::string relabelFile = paramRelabelFile(config);
    if (!relabelFile.empty()) {
        log("writing relabeling pairs to %s", relabelFile.c_str());
        FLAGS_save_relabel_opairs = relabelFile;
    }
    // LookAheadFst is a FstLib::StdOLabelLookAheadFst but
    // stores the relabeling data
    typedef FstLib::StdOLabelLookAheadFst::FST BaseFst;
    typedef FstLib::LabelLookAheadMatcher<
            FstLib::SortedMatcher<BaseFst>,
            FstLib::olabel_lookahead_flags | FstLib::kLookAheadKeepRelabelData,
            FstLib::FastLogAccumulator<BaseFst::Arc> > Matcher;
    typedef FstLib::MatcherFst<BaseFst,
            Matcher,
            FstLib::olabel_lookahead_fst_type,
            FstLib::LabelLookAheadRelabeler<BaseFst::Arc> > LookAheadFst;

    FstLib::StdFst *result = 0;
    if (paramKeepRelabelingData(config)) {
        log("storing relabeling data");
        LookAheadFst *l = new LookAheadFst(*input_);
        relabel(*l);
        result = l;
    } else {
        FstLib::StdOLabelLookAheadFst *l = new FstLib::StdOLabelLookAheadFst(*input_);
        relabel(*l);
        result = l;
    }
    // deleteInput();
    result->Write(paramFilename(config));
    FLAGS_save_relabel_opairs = flagRelabelOpairs;
    FLAGS_v = flagV;
    delete result;
    return 0;
}

template<class F>
void CreateLookahead::relabel(const F &f) {
    if (paramRelabelInput(config)) {
        verify(toRelabel_);
        log("relabeling input");
        FstLib::LabelLookAheadRelabeler<OpenFst::Arc>::Relabel(toRelabel_, f, true);
    }
}

const Core::Choice ReachableCompose::lookAheadChoice(
    "label", LabelLookAhead,
    "arc", ArcLookAhead,
    Core::Choice::endMark());
const Core::ParameterChoice ReachableCompose::paramLookAheadType(
    "lookahead-type", &lookAheadChoice, "type of lookahead", LabelLookAhead);

Operation::AutomatonRef ReachableCompose::process()
{
    LookAheadType laType = static_cast<LookAheadType>(paramLookAheadType(config));
    AutomatonRef result = input_->cloneWithAttributes();
    bool flagCompat = FLAGS_fst_compat_symbols;
    if (paramIgnoreSymbols(config))
        FLAGS_fst_compat_symbols = false;
    if (laType == LabelLookAhead) {
        log("using label look-ahead");
        FstLib::StdOLabelLookAheadFst *left = new FstLib::StdOLabelLookAheadFst(*input_);
        deleteInput();
        FstLib::LabelLookAheadRelabeler<OpenFst::Arc>::Relabel(right_, *left, true);
        *result = FstLib::ComposeFst<OpenFst::Arc>(*left, *right_);
        delete left;
    } else if (laType == ArcLookAhead) {
        log("using arc look-ahead");
        FstLib::StdArcLookAheadFst *left = new FstLib::StdArcLookAheadFst(*input_);
        deleteInput();
        *result = FstLib::ComposeFst<OpenFst::Arc>(*left, *right_);
        delete left;
    }
    FLAGS_fst_compat_symbols = flagCompat;
    delete right_;
    return result;
}

const Core::ParameterFloat ScaleWeights::paramScale(
    "scale", "scaling factor applied to all weights", 1.0);

Operation::AutomatonRef ScaleWeights::process()
{
    f32 scale = paramScale(config);
    log("scaling weights: %f", scale);
    OpenFst::scaleWeights(input_, scale);
    return input_;
}

const Core::ParameterFloat ScaleLabelWeights::paramScale(
    "scale", "scaling factor applied to the selected weights", 1.0);
const Core::ParameterString ScaleLabelWeights::paramLabel(
    "label", "label used to select arcs", "");

Operation::AutomatonRef ScaleLabelWeights::process()
{
    if (!input_->OutputSymbols()) {
        error("symbol table required");
        return input_;
    }
    const std::string symbol = paramLabel(config);
    OpenFst::Label label = input_->OutputSymbols()->Find(symbol);
    log("using label '%s' = %d", symbol.c_str(), label);
    f32 scale = paramScale(config);
    log("using scale %f", scale);
    u32 nModified = 0;
    for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
        OpenFst::StateId state = siter.Value();
        for (OpenFst::MutableArcIterator aiter(input_, state); !aiter.Done(); aiter.Next()) {
            if (aiter.Value().olabel == label) {
                OpenFst::Arc arc = aiter.Value();
                arc.weight = OpenFst::Weight(arc.weight.Value() * scale);
                aiter.SetValue(arc);
                ++nModified;
            }
        }
    }
    log("modified %d arcs", nModified);
    return input_;
}

Operation::AutomatonRef RemoveWeights::process()
{
    log("removing weights");
    FstLib::ArcMap(input_, FstLib::RmWeightMapper<OpenFst::Arc>());
    return input_;
}
