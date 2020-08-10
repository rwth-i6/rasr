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
#include <Core/Assertions.hh>
#include <OpenFst/Scale.hh>
#include <Search/Wfst/ComposeFst.hh>
#include <Search/Wfst/LexiconFst.hh>

using namespace Search::Wfst;

const Core::Choice LexicalFstFactory::choiceLookAheadType_(
        "none", NoLookAhead,
        "label", LabelLookAhead,
        "push-weights", PushWeights,
        "push-labels", PushLabels,
        "push-labels-only", PushLabelsOnly,
        "arc", ArcLookAhead,
        Core::Choice::endMark());
const Core::ParameterChoice LexicalFstFactory::paramLookAheadType_(
        "look-ahead", &choiceLookAheadType_, "type of composition filter",
        PushLabels);
const Core::Choice LexicalFstFactory::choiceAccumulatorType_(
        "fast-log", FastLogAccumulator,
        "log", LogAccumulator,
        "tropical", DefaultAccumulator,
        Core::Choice::endMark());
const Core::ParameterChoice LexicalFstFactory::paramAccumulatorType_(
        "accumulator", &choiceAccumulatorType_, "accumulator using for weight pushing",
        FastLogAccumulator);
const Core::ParameterBool LexicalFstFactory::paramMatcherFst_(
        "matcher-fst", "transducer to load is a olabel lookahead matcher fst", true);
const Core::ParameterFloat LexicalFstFactory::paramScale_(
        "scale", "scale arc weights (only if matcher-fst = false)", 1.0);

LexicalFstFactory::Options LexicalFstFactory::parseOptions() const {
    Options options;
    options.accumulatorType = static_cast<AccumulatorType>(paramAccumulatorType_(config));
    options.lookAhead       = static_cast<LookAheadType>(paramLookAheadType_(config));
    return options;
}

LexicalFstFactory::Options LexicalFstFactory::parseOptions(const Options& defaultValues) const {
    Options options         = defaultValues;
    options.accumulatorType = static_cast<AccumulatorType>(paramAccumulatorType_(config, options.accumulatorType));
    options.lookAhead       = static_cast<LookAheadType>(paramLookAheadType_(config, options.lookAhead));
    return options;
}

void LexicalFstFactory::logOptions(const Options& options) const {
    if (options.lookAhead & LabelLookAheadFlag)
        log("using label look-ahead");
    if (options.lookAhead & PushWeightsFlag) {
        log("using weight pushing");
        log("weight look-ahead accumulator: ")
                << choiceAccumulatorType_[options.accumulatorType];
    }
    if (options.lookAhead & PushLabelsFlag)
        log("using label pushing");
    if (options.lookAhead & ArcLookAheadFlag)
        log("using arc look-ahead");
    if (options.lookAhead == NoLookAhead)
        log("no look-ahead");
}

AbstractLexicalFst* LexicalFstFactory::load(const std::string& filename, AbstractGrammarFst::GrammarType gType,
                                            AbstractGrammarFst* g) const {
    Options options = parseOptions();
    return read(filename, options, paramMatcherFst_(config), paramScale_(config), g);
}

AbstractLexicalFst* LexicalFstFactory::load(const std::string& filename, const Options& o,
                                            AbstractGrammarFst* g) const {
    Options options         = parseOptions(o);
    options.accumulatorType = static_cast<AccumulatorType>(paramAccumulatorType_(config, options.accumulatorType));
    options.lookAhead       = static_cast<LookAheadType>(paramLookAheadType_(config, options.lookAhead));
    return read(filename, options, paramMatcherFst_(config), paramScale_(config), g);
}

AbstractLexicalFst* LexicalFstFactory::convert(OpenFst::VectorFst* base, GrammarType gType,
                                               AbstractGrammarFst* g) const {
    Options options = parseOptions();
    logOptions(options);
    AbstractLexicalFst* l = create(options);
    ensure(l);
    convert(base, paramScale_(config), l, g);
    return l;
}

AbstractLexicalFst* LexicalFstFactory::convert(OpenFst::VectorFst* base, const Options& o,
                                               AbstractGrammarFst* g) const {
    Options options = parseOptions(o);
    logOptions(options);
    AbstractLexicalFst* l = create(options);
    ensure(l);
    convert(base, paramScale_(config), l, g);
    return l;
}

AbstractLexicalFst* LexicalFstFactory::read(const std::string& filename, const Options& options,
                                            bool isMatcherFst, f32 scale, AbstractGrammarFst* g) const {
    logOptions(options);
    AbstractLexicalFst* l = create(options);
    ensure(l);
    if (isMatcherFst ||
        !(options.lookAhead & (LabelLookAheadFlag | ArcLookAheadFlag))) {
        log("assuming required fst type");
        if (!l->load(filename))
            criticalError("cannot load %s", filename.c_str());
    }
    else {
        log("creating required fst type");
        OpenFst::VectorFst* i = OpenFst::VectorFst::Read(filename);
        convert(i, scale, l, g);
        delete i;
    }
    return l;
}

void LexicalFstFactory::convert(OpenFst::VectorFst* base, f32 scale,
                                AbstractLexicalFst* result, AbstractGrammarFst* g) const {
    verify(result);
    if (!Core::isAlmostEqual(scale, static_cast<f32>(1.0), static_cast<f32>(0.001))) {
        log("re-scaling weights of L: %f", scale);
        OpenFst::scaleWeights(base, scale);
    }
    result->create(*base);
    if (g) {
        result->relabel(g);
        log("relabeled G");
    }
}

AbstractLexicalFst* LexicalFstFactory::create(const Options& options) {
    switch (options.lookAhead) {
        case ArcLookAhead:
            return new ArcLookAheadFst();
            break;
        case PushLabels:
            return createFst<PushLabelsLexicalFst>(options.accumulatorType);
            break;
        case PushWeights:
            return createFst<PushWeightsLexicalFst>(options.accumulatorType);
            break;
        case PushLabelsOnly:
            return new PushLabelsOnlyLexicalFst();
        case LabelLookAhead:
            return new LookAheadLexicalFst();
            break;
        case NoLookAhead:
            return new StandardLexicalFst();
            break;
        default:
            defect();
            return 0;
            break;
    }
}

template<template<class> class N>
AbstractLexicalFst* LexicalFstFactory::createFst(AccumulatorType t) {
    AbstractLexicalFst* f = 0;
    switch (t) {
        case DefaultAccumulator:
            f = new N<FstLib::DefaultAccumulator<Arc>>();
            break;
        case LogAccumulator:
            f = new N<FstLib::LogAccumulator<Arc>>();
            break;
        case FastLogAccumulator:
            f = new N<FstLib::FastLogAccumulator<Arc>>();
            break;
        default:
            defect();
            f = 0;
            break;
    }
    return f;
}
