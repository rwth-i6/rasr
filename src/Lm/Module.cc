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
#include "Module.hh"
#include <Core/Application.hh>
#include <Modules.hh>
#include "ClassLm.hh"
#ifdef MODULE_LM_ARPA
#include "ArpaLm.hh"
#endif
#ifdef MODULE_LM_FSA
#include "CheatingSegmentLm.hh"
#include "FsaLm.hh"
#endif
#ifdef MODULE_LM_ZEROGRAM
#include "Zerogram.hh"
#endif
#ifdef MODULE_LM_FFNN
#include "FFNeuralNetworkLanguageModel.hh"
#endif
#ifdef MODULE_LM_TFRNN
#include "TFRecurrentLanguageModel.hh"
#include "SimpleTransformerLm.hh"
#endif
#include "CombineLm.hh"

#ifdef MODULE_LM_TFRNN
#include "DummyCompressedVectorFactory.hh"
#include "FixedQuantizationCompressedVectorFactory.hh"
#include "QuantizedCompressedVectorFactory.hh"
#include "ReducedPrecisionCompressedVectorFactory.hh"
#endif

#include "SimpleHistoryLm.hh"

using namespace Lm;

namespace Lm {
enum LanguageModelType {
    lmTypeArpa,
    lmTypeArpaWithClasses,
    lmTypeFsa,
    lmTypeZerogram,
    lmTypeFFNN,
    lmTypeCombine,
    lmTypeTFRNN,
    lmTypeCheatingSegment,
    lmTypeSimpleHistory,
    lmTypeSimpleTransformer
};
}

const Core::Choice Module_::lmTypeChoice(
        "ARPA", lmTypeArpa,
        "ARPA+classes", lmTypeArpaWithClasses,
        "fsa", lmTypeFsa,
        "zerogram", lmTypeZerogram,
        "ffnn", lmTypeFFNN,
        "combine", lmTypeCombine,
        "tfrnn", lmTypeTFRNN,
        "cheating-segment", lmTypeCheatingSegment,
        "simple-transformer", lmTypeSimpleTransformer,
        "simple-history", lmTypeSimpleHistory,
        Core::Choice::endMark());

const Core::ParameterChoice Module_::lmTypeParam(
        "type", &Module_::lmTypeChoice, "type of language model", lmTypeZerogram);

Core::Ref<LanguageModel> Module_::createLanguageModel(
        const Core::Configuration& c,
        Bliss::LexiconRef          l) {
    Core::Ref<LanguageModel> result;

    switch (lmTypeParam(c)) {
#ifdef MODULE_LM_ARPA
        case lmTypeArpa: result = Core::ref(new ArpaLm(c, l)); break;
        case lmTypeArpaWithClasses: result = Core::ref(new ArpaClassLm(c, l)); break;
#endif
#ifdef MODULE_LM_FSA
        case lmTypeFsa: result = Core::ref(new FsaLm(c, l)); break;
        case lmTypeCheatingSegment: result = Core::ref(new CheatingSegmentLm(c, l)); break;
#endif
#ifdef MODULE_LM_ZEROGRAM
        case lmTypeZerogram: result = Core::ref(new Zerogram(c, l)); break;
#endif
#ifdef MODULE_LM_FFNN
        case lmTypeFFNN: result = Core::ref(new FFNeuralNetworkLanguageModel(c, l)); break;
#endif
        case lmTypeCombine: result = Core::ref(new CombineLanguageModel(c, l)); break;
#ifdef MODULE_LM_TFRNN
        case lmTypeTFRNN: result = Core::ref(new TFRecurrentLanguageModel(c, l)); break;
        case lmTypeSimpleTransformer: result = Core::ref(new SimpleTransformerLm(c, l));          break;
#endif
        case lmTypeSimpleHistory: result = Core::ref(new SimpleHistoryLm(c, l)); break;
        default:
            Core::Application::us()->criticalError("unknwon language model type: %d", lmTypeParam(c));
    }
    result->init();
    if (result->hasFatalErrors())
        result.reset();
    return result;
}

Core::Ref<ScaledLanguageModel> Module_::createScaledLanguageModel(
        const Core::Configuration& c, Core::Ref<LanguageModel> languageModel) {
    return languageModel ? Core::Ref<ScaledLanguageModel>(new LanguageModelScaling(c, languageModel)) : Core::Ref<ScaledLanguageModel>();
}

#ifdef MODULE_LM_TFRNN
enum CompressedVectorFactoryType {
    DummyCompressedVectorFactoryType,
    FixedQuantizationCompressedVectorFactoryType,
    QuantizedCompressedVectorFactoryType,
    ReducedPrecisionCompressedVectorFactoryType
};

const Core::Choice Module_::compressedVectorFactoryTypeChoice(
        "dummy", DummyCompressedVectorFactoryType,
        "fixed-quantization", FixedQuantizationCompressedVectorFactoryType,
        "quantized", QuantizedCompressedVectorFactoryType,
        "reduced-precision", ReducedPrecisionCompressedVectorFactoryType,
        Core::Choice::endMark());

const Core::ParameterChoice Module_::compressedVectorFactoryTypeParam(
        "type", &Module_::compressedVectorFactoryTypeChoice,
        "type of compressed vector factory",
        DummyCompressedVectorFactoryType);

Lm::CompressedVectorFactoryPtr<float> Module_::createCompressedVectorFactory(Core::Configuration const& config) {
    switch (compressedVectorFactoryTypeParam(config)) {
        case DummyCompressedVectorFactoryType: return CompressedVectorFactoryPtr<float>(new Lm::DummyCompressedVectorFactory<float>(config));
        case FixedQuantizationCompressedVectorFactoryType: return CompressedVectorFactoryPtr<float>(new Lm::FixedQuantizationCompressedVectorFactory(config));
        case QuantizedCompressedVectorFactoryType: return CompressedVectorFactoryPtr<float>(new Lm::QuantizedCompressedVectorFactory(config));
        case ReducedPrecisionCompressedVectorFactoryType: return CompressedVectorFactoryPtr<float>(new Lm::ReducedPrecisionCompressedVectorFactory(config));
        default: defect();
    }
}

#endif
