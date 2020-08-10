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
#include "Prior.hh"

#include <sstream>

#include <Math/Module.hh>

using namespace Nn;

template<typename T>
const Core::ParameterString Prior<T>::paramPriorFile(
        "prior-file", "", "");

template<typename T>
const Core::ParameterFloat Prior<T>::paramPrioriScale(
        "priori-scale", "scaling of the logarithmized state priori probability", 1.0);

template<typename T>
const Core::ParameterBool Prior<T>::paramCompatibilityMode(
        "compatibility-mode", "calculate prior as in old version", false);

template<typename T>
const Core::ParameterInt Prior<T>::paramBackOffCount(
        "back-off-count", "minimal count assumed in prior estimation", 1);

template<typename T>
const Core::ParameterFloat Prior<T>::paramLearningRate(
        "prior-learning-rate", "learning rate", 0.0);

template<typename T>
Prior<T>::Prior(const Core::Configuration& c)
        : Core::Component(c),
          priorFilename_(paramPriorFile(c)),
          compatibilityMode_(paramCompatibilityMode(c)),
          scale_(paramPrioriScale(c)),
          backOffCount_(paramBackOffCount(c)),
          learningRate_(paramLearningRate(c)),
          statisticsChannel_(c, "statistics") {
    this->log("using priori scale ") << scale_;
    if (compatibilityMode_)
        this->log("using compatibility mode for prior calculation");
}

template<typename T>
void Prior<T>::initUniform(u32 classCount) {
    logPrior_.resize(classCount);
    logPrior_.initComputation(false);
    logPrior_.setToZero();
    T logNorm = -std::log(classCount);          // = log(1/classCount)
    logPrior_.addConstantElementwise(logNorm);  // in +log space
    logPrior_.finishComputation(true);
}

template<typename T>
static void updatePriors(
        const typename Types<T>::NnVector& logPriors,
        T                                  logPriorScale,
        typename Types<T>::NnVector&       priors,
        T&                                 logNorm) {
    priors.copy(logPriors);
    priors.scale(logPriorScale);
    // apply softmax
    priors.exp();
    T norm = priors.l1norm();
    norm   = std::max(norm, Core::Type<T>::epsilon);
    norm   = std::min(norm, T(1.0) / Core::Type<T>::epsilon);
    priors.scale(1.0 / norm);
    logNorm = std::log(norm);
}

template<typename T>
void Prior<T>::trainSoftmax(const NnVector& errorSignal, T errFactor) {
    u32 nClasses = size();
    require_eq(errorSignal.nRows(), nClasses);

    logPrior_.initComputation(false);  // no need to sync, GPU-mem is already correct

    T        logNorm;
    NnVector priors(nClasses);
    priors.initComputation(false);
    updatePriors(logPrior_, scale_, priors, logNorm);

    NnVector p(nClasses);
    p.initComputation(false);
    p.copy(errorSignal);

    NnVector grad(nClasses);
    grad.initComputation(false);
    grad.setToZero();
    grad.add(p, (T)-1);

    // XXX: This would probably be more stable in log space.
    // P''' = P'' .* 1/p(a)
    p.elementwiseDivision(priors);

    // XXX: This syncs to CPU. Bottleneck?
    // = P''' * p(a)
    T f = p.dot(priors);

    // grad = -P'' + p(a) * (P''' * p(a))
    grad.add(priors, f);

    // XXX: This would probably be more stable in log space.
    // errFactor is the error factor of some outer error function. (L(P))
    // This is the gradient descent step.
    logPrior_.add(grad, -errFactor * learningRate_ / scale_);

    updatePriors(logPrior_, scale_, priors, logNorm);
    logPrior_.addConstantElementwise(-logNorm / scale_);

    logPrior_.finishComputation(true);
}

template<typename T>
void Prior<T>::setFromClassCounts(const Statistics<T>& statistics, const Math::Vector<T>& classWeights) {
    logPrior_.resize(classWeights.size());
    this->log("calculating prior from class counts");
    this->log("using back-off-count ") << backOffCount_;
    T totalWeight = 0.0;
    for (u32 c = 0; c < logPrior_.size(); c++) {
        if (statistics.classCount(c) == 0)
            this->warning("zero observations for class: ") << c;
        logPrior_.at(c) = std::max(statistics.classCount(c), backOffCount_);
        logPrior_.at(c) *= classWeights.at(c);
        if (statisticsChannel_.isOpen()) {
            std::stringstream ss;
            ss << "class-" << c;
            std::string xmlName = ss.str();
            statisticsChannel_ << Core::XmlOpen(xmlName.c_str())
                               << Core::XmlFull("number-of-observations", statistics.classCount(c))
                               << Core::XmlFull("weighted-number-of-observations", logPrior_.at(c))
                               << Core::XmlClose(xmlName.c_str());
        }
        totalWeight += logPrior_.at(c);
    }
    for (u32 c = 0; c < logPrior_.size(); c++)
        logPrior_.at(c) = logPrior_.at(c) == 0 ? Core::Type<T>::min : std::log(logPrior_.at(c) / totalWeight);
    if (statisticsChannel_.isOpen())
        statisticsChannel_ << logPrior_;
    // sync to GPU memory. train() expects it to be up-to-date
    logPrior_.initComputation(true);
    logPrior_.finishComputation(false);
}

template<typename T>
void Prior<T>::setFromMixtureSet(Core::Ref<const Mm::MixtureSet> mixtureSet, const ClassLabelWrapper& labelWrapper) {
    log("calculating prior from mixture set");
    // get counts from mixture set
    std::vector<f32> priorFromMixtureSet;
    priorFromMixtureSet.resize(mixtureSet->nMixtures());
    require_eq(priorFromMixtureSet.size(), labelWrapper.nClasses());
    for (size_t m = 0; m < mixtureSet->nMixtures(); ++m) {
        const Mm::Mixture* mixture    = mixtureSet->mixture(m);
        size_t             nDensities = mixture->nDensities();
        for (size_t dns = 0; dns < nDensities; ++dns) {
            priorFromMixtureSet.at(m) += mixture->weight(dns);  /// returns exp(logWeights_[densityInMixture]), @see Mm/Mixture.cc
        }
    }

    // map counts corresponding to order in output layer
    logPrior_.resize(labelWrapper.nClassesToAccumulate());
    for (u32 m = 0; m < mixtureSet->nMixtures(); m++) {
        if (labelWrapper.isClassToAccumulate(m))
            logPrior_.at(labelWrapper.getOutputIndexFromClassIndex(m)) = priorFromMixtureSet.at(m);
    }

    f32 observationWeight = 0.0;
    // normalize and apply log
    if (compatibilityMode_)
        observationWeight = std::accumulate(priorFromMixtureSet.begin(), priorFromMixtureSet.end(), 0.0);
    else
        observationWeight = std::accumulate(logPrior_.begin(), logPrior_.end(), 0.0);

    for (u32 c = 0; c < logPrior_.size(); ++c)
        logPrior_.at(c) = std::log(logPrior_.at(c) / observationWeight);
    if (statisticsChannel_.isOpen())
        statisticsChannel_ << logPrior_;
    // sync to GPU memory. train() expects it to be up-to-date
    logPrior_.initComputation(true);
    logPrior_.finishComputation(false);
}

template<typename T>
void Prior<T>::getVector(NnVector& prior) const {
    bool computingMode = prior.isComputing();
    if (computingMode)
        logPrior_.initComputation(false);
    prior.copy(logPrior_);
    if (computingMode)
        logPrior_.finishComputation(false);
}

template<typename T>
bool Prior<T>::read() {
    require(priorFilename_ != "");
    return read(priorFilename_);
}

template<typename T>
bool Prior<T>::read(const std::string& filename) {
    this->log("reading prior from file ") << filename;
    Math::Vector<T> priors;
    if (!Math::Module::instance().formats().read(filename, priors))
        return false;
    logPrior_.resize(priors.size());
    logPrior_.copy(priors);
    // sync to GPU memory. train() expects it to be up-to-date
    logPrior_.initComputation(true);
    logPrior_.finishComputation(false);
    return true;
}

template<typename T>
bool Prior<T>::write() const {
    require(priorFilename_ != "");
    return write(priorFilename_);
}

template<typename T>
bool Prior<T>::write(const std::string& filename) const {
    this->log("writing prior to file ") << filename;
    Math::Vector<T> priors(logPrior_.size());
    logPrior_.convert(priors);
    return Math::Module::instance().formats().write(filename, priors, 20);
}

namespace Nn {
template class Prior<f32>;
template class Prior<f64>;
}  // namespace Nn
