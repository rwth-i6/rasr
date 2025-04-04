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
#include "Estimator.hh"
#include "LinearLayer.hh"
#include "MeanNormalizedSgdEstimator.hh"

using namespace Nn;

template<class T>
const Core::Choice Estimator<T>::choiceEstimatorType(
        "dummy", dummy,
        "dry-run", dryRun,
        Core::Choice::endMark());

template<class T>
const Core::ParameterChoice Estimator<T>::paramEstimatorType(
        "estimator", &choiceEstimatorType,
        "estimator for weights estimation in training", dummy);

template<typename T>
const Core::ParameterBool Estimator<T>::paramBatchMode(
        "batch-mode",
        "use batch estimator, i.e. do not update after each mini-batch, but accumulate statistics", false);

template<typename T>
const Core::ParameterInt Estimator<T>::paramAccumulateMultipleBatches(
        "accumulate-multiple-batches",
        "If greater than 1, will accumulate these number of batches. "
        "0 = not used. batch-mode=true is like accumulate-multiple-batches=<corpus-batch-number>. "
        "Note that this option only make sense if you have mini-batches with different sizes "
        "such as with BufferedSegmentFeatureProcessor -- otherwise, you could just change the "
        "mini-batch size.",
        0);

template<typename T>
const Core::ParameterFloat Estimator<T>::paramLearningRate("learning-rate", "(initial) learning-rate", 1.0);

template<typename T>
const Core::ParameterFloat Estimator<T>::paramBiasLearningRate(
        "bias-learning-rate", "bias is optimized with bias-learning-rate * learning-rate", 1.0);

template<typename T>
const Core::ParameterBool Estimator<T>::paramLogStepSize(
        "log-step-size", "log the step size, if true", false);

template<typename T>
Estimator<T>::Estimator(const Core::Configuration& config)
        : Precursor(config),
          statisticsChannel_(config, "statistics"),
          fullBatchMode_(paramBatchMode(config)),
          accumulateMultipleBatches_(paramAccumulateMultipleBatches(config)),
          initialLearningRate_(paramLearningRate(config)),
          biasLearningRate_(paramBiasLearningRate(config)),
          logStepSize_(paramLogStepSize(config)) {
    this->log("initial learning rate: ") << initialLearningRate_;
    if (fullBatchMode_)
        this->log("using full batch estimator");
    if (!fullBatchMode_ && accumulateMultipleBatches_ == 0)
        accumulateMultipleBatches_ = 1;
    if (accumulateMultipleBatches_ > 1)
        this->log("accumulate over %i batches", accumulateMultipleBatches_);
    if (accumulateMultipleBatches_ < 0)
        this->criticalError("%s cannot be negative", paramAccumulateMultipleBatches.name().c_str());
    if (fullBatchMode_ && accumulateMultipleBatches_ > 0)
        this->criticalError("full batch (%s=true) and %s does not make sense",
                            paramBatchMode.name().c_str(), paramAccumulateMultipleBatches.name().c_str());
    if (biasLearningRate_ != 1.0)
        this->log("bias learning rate: ") << biasLearningRate_;
    if (logStepSize_) {
        this->log("logging step size norm");
    }
}

template<class T>
Estimator<T>* Estimator<T>::createEstimator(const Core::Configuration& config) {
    Estimator<T>* estimator;

    EstimatorType type = (EstimatorType)paramEstimatorType(config);
    switch (type) {
        case dryRun:
            estimator = new DryRunEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: dry-run (with gradient calculation)");
            break;
        default:
            verify_eq(type, EstimatorType::dummy);
            estimator = new Estimator<T>(config);
            Core::Application::us()->log("Create Estimator: dummy");
            break;
    };

    return estimator;
}

// explicit template instantiation
namespace Nn {

template class Estimator<f32>;
template class Estimator<f64>;

}  // namespace Nn
