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
#include "AcousticModelTrainer.hh"
#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mm/Module.hh>
#include <Signal/Module.hh>
#include <Speech/Module.hh>
#include <Speech/AverageFeatureScorerActivation.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#include <Speech/AcousticModelTrainer.hh>
#include <Speech/AlignerModelAcceptor.hh>
#include <Speech/AligningFeatureExtractor.hh>
#include <Speech/FeatureScorer.hh>
#include <Speech/LabelingFeatureExtractor.hh>
#include <Speech/LatticeSetProcessor.hh>
#include <Speech/LatticeSetExtractor.hh>
#include <Speech/MixtureSetTrainer.hh>
#include <Speech/ScatterMatricesEstimator.hh>
#include <Speech/TextIndependentMixtureSetTrainer.hh>
#ifdef MODULE_CART
#include <Cart/Cluster.hh>
#include <Cart/DecisionTree.hh>
#include <Cart/Example.hh>
#include <Cart/Parser.hh>
#include <Speech/DecisionTreeTrainer.hh>
#endif
#ifdef MODULE_SPEECH_DT_ADVANCED
#include <Speech/ClusterMixtureSetTrainer.hh>
#endif
#ifdef MODULE_ADAPT_MLLR
#include <Speech/ModelTransformEstimator.hh>
#endif
#ifdef MODULE_ADAPT_CMLLR
#include <Speech/AffineFeatureTransformEstimator.hh>
#endif
#ifdef MODULE_SPEECH_DT
#include <Speech/DiscriminativeMixtureSetTrainer.hh>
#endif
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

APPLICATION(AcousticModelTrainer)

const Core::Choice AcousticModelTrainer::choiceAction(
        "not-given", actionNotGiven,
        "dry", actionDryRun,
        "create-model-acceptors", actionCreateModelAcceptors,
        "show-mixture-set", actionShowMixtureSet,
        "convert-mixture-set", actionConvertMixtureSet,
        "combine-mixture-sets", actionCombineMixtureSets,
        "combine-mixture-set-estimators", actionCombineMixtureSetEstimators,
        "map-mixture-set", actionMapMixtureSet,
        "accumulate-mixture-set-text-dependent", actionAccumulateMixtureSetTextDependent,
        "accumulate-mixture-set-text-dependent-tied", actionAccumulateMixtureSetTextDependentTied,
        "accumulate-mixture-set-text-independent", actionAccumulateMixtureSetTextIndependent,
        "accumulate-tdc-sum", actionAccumulateTdcSumFile,
        "accumulate-cart-examples", actionAccumulateCartExamples,
        "merge-cart-examples", actionMergeCartExamples,
        "estimate-cart", actionEstimateCart,
        "convert-mixture-set-to-mixture-set-estimator", actionConvertMixtureSetToMixtureSetEstimator,
        "score-features", actionScoreFeatures,
        "estimate-scatter-matrices-text-dependent", actionEstimateScatterMatricesTextDependent,
        "estimate-scatter-matrices-text-independent", actionEstimateScatterMatricesTextIndependent,
        "estimate-lda-text-dependent", actionEstimateLdaTextDependent,
        "estimate-lda-text-independent", actionEstimateLdaTextIndependent,
        "combine-lda", actionCombineScatterMatrixAccumulators,
        "combine-scatter-matrix-accumulators", actionCombineScatterMatrixAccumulators,
        "estimate-scatter-matrices-from-accumulator", actionEstimateScatterMatricesFromScatterMatrixAccumulator,
        "estimate-lda", actionEstimateLda,
        "estimate-affine-feature-transform", actionEstimateAffineFeatureTransform,
        "calculate-affine-feature-transform", actionCalculateAffineFeatureTransform,
        "combine-affine-feature-transform-estimators", actionCombineAffineFeatureTransformEstimators,
        "score-affine-feature-transform", actionScoreAffineFeatureTransform,
        "estimate-adaptation", actionEstimateModelTransform,
        "calculate-adaptation", actionCalculateModelTransform,
        "calculate-average-feature-scorer-activation", actionCalculateAverageFeatureScorerActivation,
        Core::Choice::endMark());

const Core::ParameterChoice AcousticModelTrainer::paramAction(
        "action", &choiceAction, "operation to perfom", actionNotGiven);

const Core::ParameterString paramActionName("action", "", "n/a");

const Core::ParameterBool AcousticModelTrainer::paramMixtureSetFilenamesToCombineUseComma(
        "mixture-set-files-to-combine-use-comma", "Use a comma as separator in mixture-set-files-to-combine instead of a space.", false);

const Core::ParameterStringVector AcousticModelTrainer::paramMixtureSetFilenamesToCombine(
        "mixture-set-files-to-combine", "name of mixture set file(s) to combine", " ", 1);

const Core::ParameterStringVector AcousticModelTrainer::paramMixtureSetFilenamesToCombineComma(
        "mixture-set-files-to-combine", "name of mixture set file(s) to combine", ",", 1);

const Core::ParameterString AcousticModelTrainer::paramMappingFilenameToMap(
        "mapping-file", "name of mapping filename");

const Core::ParameterBool AcousticModelTrainer::paramForceExampleRegeneration(
        "force-example-regeneration", "force regeneration of the example file (equivalent to former .sum files) during Cart estimation yes/no", true);

AcousticModelTrainer::AcousticModelTrainer() {
    setTitle("acoustic-model-trainer");
    INIT_MODULE(Flow);
    INIT_MODULE(Am);
    INIT_MODULE(Audio);
    INIT_MODULE(Lm);
    INIT_MODULE(Math);
    INIT_MODULE(Mm);
    INIT_MODULE(Signal);
    INIT_MODULE(Speech);
#ifdef MODULE_NN
    INIT_MODULE(Nn);
#endif
#ifdef MODULE_TENSORFLOW
    INIT_MODULE(Tensorflow);
#endif
}

int AcousticModelTrainer::main(const std::vector<std::string>& arguments) {
    log() << "action: " << paramActionName(config);
    switch ((Action)paramAction(config)) {
        case actionDryRun:
            dryRun();
            break;
        case actionCreateModelAcceptors:
            createModelAcceptors();
            break;
        case actionShowMixtureSet:
            showMixtureSet();
            break;
        case actionConvertMixtureSet:
            convertMixtureSet();
            break;
        case actionCombineMixtureSets:
            combineMixtureSets();
            break;
        case actionCombineMixtureSetEstimators:
            combineMixtureSetEstimators();
            break;
        case actionMapMixtureSet:
            mapMixtureSet();
            break;
        case actionAccumulateMixtureSetTextDependent:
            accumulateMixtureSetTextDependent();
            break;
        case actionAccumulateMixtureSetTextDependentTied:
            accumulateMixtureSetTextDependentTied();
            break;
        case actionAccumulateMixtureSetTextIndependent:
            accumulateMixtureSetTextIndependent();
            break;
        case actionAccumulateCartExamples:
            accumulateCartExamples();
            break;
        case actionMergeCartExamples:
            mergeCartExamples();
            break;
        case actionEstimateCart:
            estimateCart();
            break;
        case actionConvertMixtureSetToMixtureSetEstimator:
            convertMixtureSetToMixtureSetEstimator();
            break;
        case actionScoreFeatures:
            scoreFeatures();
            break;
        case actionEstimateScatterMatricesTextDependent:
            estimateScatterMatricesTextDependent();
            break;
        case actionEstimateScatterMatricesTextIndependent:
            estimateScatterMatricesTextIndependent();
            break;
        case actionEstimateLdaTextDependent:
            estimateLdaTextDependent();
            break;
        case actionEstimateLdaTextIndependent:
            estimateLdaTextIndependent();
            break;
        case actionCombineScatterMatrixAccumulators:
            combineScatterMatrixAccumulators();
            break;
        case actionEstimateScatterMatricesFromScatterMatrixAccumulator:
            estimateScatterMatricesFromScatterMatrixAccumulator();
            break;
        case actionEstimateLdaFromScatterMatrixAccumulator:
            estimateLdaFromScatterMatrixAccumulator();
            break;
        case actionEstimateLda:
            estimateLda();
            break;
        case actionEstimateAffineFeatureTransform:
            estimateAffineFeatureTransform();
            break;
        case actionCalculateAffineFeatureTransform:
            calculateAffineFeatureTransform();
            break;
        case actionScoreAffineFeatureTransform:
            scoreAffineFeatureTransform();
            break;
        case actionCombineAffineFeatureTransformEstimators:
            combineAffineFeatureTransformEstimators();
            break;
        case actionEstimateModelTransform:
            estimateModelTransform();
            break;
        case actionCalculateModelTransform:
            calculateModelTransform();
            break;
        case actionCalculateAverageFeatureScorerActivation:
            calculateAverageFeatureScorerActivation();
            break;
        default:
            criticalError("Action not given.");
    };

    return 0;
}

void AcousticModelTrainer::dryRun() {
    Speech::AlignedFeatureProcessor trainer(select("dummy-trainer"));
    visitCorpus(trainer);
}

void AcousticModelTrainer::createModelAcceptors() {
    Speech::AlignerModelAcceptorGenerator modelAcceptorGenerator(select("model-acceptor-generator"));
    visitCorpus(modelAcceptorGenerator);
}

/**
 * Print the mixture set estimator in XML format. Following parameter combinations are supported:
 *   - old-mixture-set-file
 *       prints the mixture
 *
 *   - old-mixture-set-file
 *     lexicon.file
 *     state-tying.type=monophone
 *       prints monophone state labels for each mixture (context is arbitrary)
 *
 *   - old-mixture-set-file
 *     lexicon.file
 *     state-tying.type=cart
 *     state-tying.file=cart.xml
 *       prints cart labels for each mixture (i.e. mixture number)
 */
void AcousticModelTrainer::showMixtureSet() {
    Speech::MixtureSetTrainer* trainer = Speech::Module::instance().createMixtureSetTrainer(select("mixture-set-trainer"));

    std::map<int, std::string> mixtureLabels;
    std::string                lexiconFile;
    select("lexicon").get("file", lexiconFile);
    if (lexiconFile != "") {
        Bliss::LexiconRef                   lexicon           = Bliss::Lexicon::create(select("lexicon"));
        Core::Ref<Am::ClassicAcousticModel> am                = Core::ref(new Am::ClassicAcousticModel(select("acoustic-model"), lexicon));
        Am::ConstAllophoneStateAlphabetRef  alloStateAlphabet = am->allophoneStateAlphabet();
        Am::ClassicStateTyingRef            stateTying        = am->stateTying();
        am->load(Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTransition);

        std::string tyingType;
        select("state-tying").get("type", tyingType);

        if (tyingType == "monophone") {
            stateTying = am->stateTying();
            for (std::pair<Am::AllophoneStateIterator, Am::AllophoneStateIterator> it = alloStateAlphabet->allophoneStates();
                 it.first != it.second; ++it.first) {
                int idx            = stateTying->classify(it.first.allophoneState());
                mixtureLabels[idx] = alloStateAlphabet->toString(it.first.allophoneState());
            }
        }
        else {
            for (u32 idx = 0; idx < am->nEmissions(); ++idx) {
                mixtureLabels[idx] = Core::itoa(idx);
            }
        }
    }

    trainer->read();
    trainer->write(clog(), mixtureLabels);
    delete trainer;
}

void AcousticModelTrainer::convertMixtureSet() {
    Mm::Module::instance().convertMixtureSet(select("mixture-set"));
}

void AcousticModelTrainer::combineMixtureSets() {
    const Core::Configuration mixtureSetTrainerConfig = select("mixture-set-trainer");
    if (!Speech::MixtureSetTrainer::paramOldMixtureSetFilename(mixtureSetTrainerConfig, "").empty())
        Core::Application::us()->error("Old mixture set file is ignored!");
    Speech::MixtureSetTrainer* trainer = Speech::Module::instance().createMixtureSetTrainer(mixtureSetTrainerConfig);
    trainer->combine(paramMixtureSetFilenamesToCombine(select("mixture-set-trainer")));
    delete trainer;
}

void AcousticModelTrainer::combineMixtureSetEstimators() {
    Speech::MixtureSetTrainer* trainer   = Speech::Module::instance().createMixtureSetTrainer(select("mixture-set-trainer"));
    std::vector<std::string>   toCombine = paramMixtureSetFilenamesToCombine(select("mixture-set-trainer"));
    if (trainer->combinePartitions(toCombine)) {
        trainer->write();
    }
    delete trainer;
}

void AcousticModelTrainer::mapMixtureSet() {
    Speech::MixtureSetTrainer* trainer = Speech::Module::instance().createMixtureSetTrainer(select("mixture-set-trainer"));
    trainer->read();
    if (trainer->map(paramMappingFilenameToMap(select("mixture-set-trainer")))) {
        trainer->write();
    }
    delete trainer;
}

void AcousticModelTrainer::accumulateMixtureSetTextDependent() {
    Speech::TextDependentMixtureSetTrainer trainer(select("mixture-set-trainer"));
    visitCorpus(trainer);
    trainer.write();
}

void AcousticModelTrainer::accumulateMixtureSetTextDependentTied() {
    Speech::TiedTextDependentMixtureSetTrainer trainer(select("mixture-set-trainer"));
    visitCorpus(trainer);
    trainer.write();
}

void AcousticModelTrainer::accumulateMixtureSetTextIndependent() {
    Speech::TextIndependentMixtureSetTrainer trainer(select("mixture-set-trainer"));
    visitCorpus(trainer);
    trainer.write();
}

/*
  Decision tree estimation is done in two steps.
  first:  accumulate and store examples
  second: load examples and train cart
*/
void AcousticModelTrainer::accumulateCartExamples() {
#ifdef MODULE_CART
    Speech::FeatureAccumulator accu(select("cart-trainer"));
    accu.map().writeXml((Core::XmlWriter&)log());
    log("collect examples");
    visitCorpus(accu);
    accu.examples().writeToFile();
#else
    criticalError("Module Cart is not available");
#endif
}

void AcousticModelTrainer::mergeCartExamples() {
#ifdef MODULE_CART
    Cart::ExampleList examples(select("cart-trainer"));
    examples.mergeFromFiles();
    examples.writeToFile();
#else
    criticalError("Module Cart is not available");
#endif
}

void AcousticModelTrainer::estimateCart() {
#ifdef MODULE_CART
    Cart::DecisionTree tree(select("cart-trainer"));
    Cart::ClusterList* clusters;

    Speech::StateTyingDecisionTreeTrainer trainer(select("cart-trainer"));
    log("load training plan");
    if (!trainer.loadFromFile()) {
        criticalError("unable to parse training file \"%s\"",
                      Cart::DecisionTreeTrainer::paramTrainingFilename(config).c_str());
        return;
    }
    trainer.write((Core::XmlWriter&)log());
    log("train decision tree");
    if (!(clusters = trainer.train(&tree))) {
        criticalError("error while training decision tree");
        return;
    }

    tree.writeToFile();

    clusters->writeToFile();
    delete clusters;
#else
    criticalError("Module Cart is not available");
#endif
}

void AcousticModelTrainer::convertMixtureSetToMixtureSetEstimator() {
#ifdef MODULE_MM_DT
    Speech::ConvertMixtureSetTrainer trainer(select("mixture-set-trainer"));
    trainer.read();
    trainer.write();
#else
    criticalError("Module MM_DT is not available");
#endif
}

void AcousticModelTrainer::scoreFeatures() {
    Speech::FeatureScorer featureScorer(select("feature-scorer"));
    visitCorpus(featureScorer);
    featureScorer.write();
}

void AcousticModelTrainer::estimateScatterMatricesTextDependent() {
    Speech::TextDependentScatterMatricesEstimator estimator(select("scatter-matrices-estimator"));
    visitCorpus(estimator);
    estimator.getEstimator()->write();
}

void AcousticModelTrainer::estimateScatterMatricesTextIndependent() {
    Speech::TextIndependentScatterMatricesEstimator estimator(select("scatter-matrices-estimator"));
    visitCorpus(estimator);
    estimator.getEstimator()->write();
}

void AcousticModelTrainer::estimateLdaTextDependent() {
    Speech::TextDependentScatterMatricesEstimator estimator(select("scatter-matrices-estimator"));
    visitCorpus(estimator);
    estimateLda(*estimator.getEstimator());
}

void AcousticModelTrainer::estimateLdaTextIndependent() {
    Speech::TextIndependentScatterMatricesEstimator estimator(select("scatter-matrices-estimator"));
    visitCorpus(estimator);
    estimateLda(*estimator.getEstimator());
}

void AcousticModelTrainer::combineScatterMatrixAccumulators() {
    const Core::Configuration scatterMatricesEstimatorConfig          = select("scatter-matrix-estimator");
    const Core::Configuration scatterMatricesEstimatorToCombineConfig = select("scatter-matrix-estimator-to-combine");

    Signal::ScatterMatricesEstimator estimator(scatterMatricesEstimatorConfig);
    if (!Signal::ScatterMatricesEstimator::paramOldAccumulatorFilename(scatterMatricesEstimatorToCombineConfig, "").empty()) {
        // old scatter matrices accumulator combination syntax
        estimator.load();
        Signal::ScatterMatricesEstimator other(scatterMatricesEstimatorToCombineConfig);
        other.loadAccumulatorFile(Signal::ScatterMatricesEstimator::paramOldAccumulatorFilename(scatterMatricesEstimatorToCombineConfig));
        estimator.accumulate(other);
    }
    else {
        if (!Signal::ScatterMatricesEstimator::paramOldAccumulatorFilename(scatterMatricesEstimatorConfig, "").empty())
            Core::Application::us()->error("Old scatter accumulator file is ignored!");
        // multiple accumulator combination syntax
        estimator.addAccumulatorFiles(Signal::ScatterMatricesEstimator::paramAccumulatorFilesToCombine(scatterMatricesEstimatorConfig));
    }
    estimator.write();
}

void AcousticModelTrainer::estimateScatterMatricesFromScatterMatrixAccumulator() {
    Signal::ScatterMatricesEstimator estimator(select("scatter-matrix-estimator"));
    estimator.load();
    estimator.write();
}

void AcousticModelTrainer::estimateLdaFromScatterMatrixAccumulator() {
    Signal::ScatterMatricesEstimator estimator(select("scatter-matrix-estimator"));
    estimator.load();
    estimateLda(estimator);
}

void AcousticModelTrainer::estimateLda(Signal::ScatterMatricesEstimator& estimator) {
    Signal::ScatterMatrix betweenClassScatterMatrix;
    Signal::ScatterMatrix withinClassScatterMatrix;
    Signal::ScatterMatrix totalScatterMatrix;
    estimator.finalize(betweenClassScatterMatrix, withinClassScatterMatrix, totalScatterMatrix);
    Signal::LinearDiscriminantAnalysis lda(select("lda-estimator"));
    lda.work(betweenClassScatterMatrix, withinClassScatterMatrix);
    lda.write();
}

void AcousticModelTrainer::estimateLda() {
    Signal::LinearDiscriminantAnalysis lda(select("lda-estimator"));
    lda.work();
    lda.write();
}

void AcousticModelTrainer::estimateAffineFeatureTransform() {
#ifdef MODULE_ADAPT_CMLLR
    Speech::AffineFeatureTransformEstimator estimator(select("affine-feature-transform-estimator"));
    visitCorpus(estimator);
    estimator.postProcess();
#else
    criticalError("Module ADAPT_CMLLR is not available");
#endif
}

void AcousticModelTrainer::calculateAffineFeatureTransform() {
#ifdef MODULE_ADAPT_CMLLR
    Speech::AffineFeatureTransformEstimator estimator(select("affine-feature-transform-estimator"), Speech::AffineFeatureTransformEstimator::calculate);
    estimator.postProcess();
#else
    criticalError("Module ADAPT_CMLLR is not available");
#endif
}

void AcousticModelTrainer::combineAffineFeatureTransformEstimators() {
#ifdef MODULE_ADAPT_CMLLR
    Speech::AffineFeatureTransformEstimator estimator(select("affine-feature-transform-estimator"), Speech::AffineFeatureTransformEstimator::combines);
    estimator.combine();
    estimator.postProcess();
#else
    criticalError("Module ADAPT_CMLLR is not available");
#endif
}

void AcousticModelTrainer::scoreAffineFeatureTransform() {
#ifdef MODULE_ADAPT_CMLLR
    Speech::AffineFeatureTransformEstimator estimator(select("affine-feature-transform-estimator"), Speech::AffineFeatureTransformEstimator::calculate);
    estimator.scoreTransforms();
#else
    criticalError("Module ADAPT_CMLLR is not available");
#endif
}

void AcousticModelTrainer::estimateModelTransform() {
#ifdef MODULE_ADAPT_MLLR
    Speech::ModelTransformEstimator estimator(select("adaptation-estimator"));
    visitCorpus(estimator);
    estimator.postProcess();
#else
    criticalError("Module ADAPT_MLLR is not available");
#endif
}

void AcousticModelTrainer::calculateModelTransform() {
#ifdef MODULE_ADAPT_MLLR
    Speech::ModelTransformEstimator estimator(select("adaptation-estimator"),
                                              Speech::ModelTransformEstimator::calculate);
    estimator.postProcess();
#else
    criticalError("Module ADAPT_MLLR is not available");
#endif
}

void AcousticModelTrainer::calculateAverageFeatureScorerActivation() {
    Speech::AverageFeatureScorerActivation afse(select("average-feature-scorer-activation"));
    visitCorpus(afse);
    afse.write();
}

void AcousticModelTrainer::visitCorpus(Speech::CorpusProcessor& corpusProcessor) {
    Speech::CorpusVisitor corpusVisitor(select("corpus"));
    corpusProcessor.signOn(corpusVisitor);

    Bliss::CorpusDescription corpusDescription(select("corpus"));
    corpusDescription.accept(&corpusVisitor);
}

void AcousticModelTrainer::visitCorpus(Speech::CorpusProcessor* corpusProcessor) {
    visitCorpus(*corpusProcessor);
    delete corpusProcessor;
}

void AcousticModelTrainer::visitCorpus(Speech::AlignedFeatureProcessor& alignedFeatureProcessor) {
    visitCorpus(Speech::Module::instance().createAligningFeatureExtractor(
            select("aligning-feature-extractor"), alignedFeatureProcessor));
}

void AcousticModelTrainer::visitCorpus(Speech::LabeledFeatureProcessor& labeledFeatureProcessor) {
    Speech::LabelingFeatureExtractor labelingFeatureExtractor(select("labeling"), labeledFeatureProcessor);
    visitCorpus(labelingFeatureExtractor);
}

