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
#include "SmoothingFunction.hh"

using namespace Lattice;

SmoothingFunction::SmoothingFunction()
        : sumF_(0) {}

void SmoothingFunction::dumpStatistics(
        Core::XmlWriter& os) const {
    os << Core::XmlFull("objective-function", sumF_);
}

/*
 * logarithmic smoothing function
 * Used for frame-based mmi, includes margin parameter.
 */
const Core::ParameterFloat LogSmoothingFunction::paramM(
        "m",
        "margin in distance",
        0);

LogSmoothingFunction::LogSmoothingFunction(
        const Core::Configuration& config)
        : xM_(exp(paramM(config))) {}

void LogSmoothingFunction::dumpStatistics(Core::XmlWriter& os) const {
    os << Core::XmlOpen("log-smoothing-function");
    os << Core::XmlOpen("statistics");
    Precursor::dumpStatistics(os);
    os << Core::XmlClose("statistics");
    os << Core::XmlClose("log-smoothing-function");
}

/*
 * sigmoid smoothing function
 * Used for frame-based mce, includes margin parameter.
 */
const Core::ParameterFloat SigmoidSmoothingFunction::paramBeta(
        "beta",
        "smoothing parameter",
        1,
        0);

const Core::ParameterFloat SigmoidSmoothingFunction::paramM(
        "m",
        "margin in distance",
        0);

SigmoidSmoothingFunction::SigmoidSmoothingFunction(
        const Core::Configuration& config)
        : beta_(paramBeta(config)),
          xM_(exp(paramM(config))) {}

void SigmoidSmoothingFunction::dumpStatistics(Core::XmlWriter& os) const {
    os << Core::XmlOpen("sigmoid-smoothing-function");
    os << Core::XmlOpen("statistics");
    Precursor::dumpStatistics(os);
    os << Core::XmlClose("statistics");
    os << Core::XmlClose("sigmoid-smoothing-function");
}

/*
 * smoothing function for unsupervized
 * Defined on the distance d=log(p/(1-p))
 * where p is the probability of the correct state.
 * The smoothing function is constant in the
 * intervals [-inf,-b], [-a,a], and [a,inf].
 * In the intervals [-b,-a] and [a,b], a cosine function
 * models the smooth transition from 0 to 1.
 */
const Core::ParameterFloat UnsupervizedSmoothingFunction::paramA(
        "a",
        "lower threshold in distance",
        0);

const Core::ParameterFloat UnsupervizedSmoothingFunction::paramB(
        "b",
        "upper threshold in distance",
        10);

UnsupervizedSmoothingFunction::UnsupervizedSmoothingFunction(const Core::Configuration& config)
        : dA_(paramA(config)),
          dB_(paramB(config)),
          dS_((M_PI / (dB_ - dA_))),
          xBn_(exp(-dB_) / (1 + exp(-dB_))),
          xAn_(exp(-dA_) / (1 + exp(-dA_))),
          xAp_(exp(dA_) / (1 + exp(dA_))),
          xBp_(exp(dB_) / (1 + exp(dB_))),
          nInfB_(0),
          nBA_(0),
          nAA_(0),
          nAB_(0),
          nBInf_(0) {
    require(dA_ < dB_);
    require(xBn_ > 0);
    require(xBn_ < xAn_);
    require(xAn_ < xAp_);
    require(xAp_ < xBp_);
    require(xBp_ < 1);
}

// f(x)
f64 UnsupervizedSmoothingFunction::f(f64 x) const {
    if ((xAn_ < x) and (x < xAp_)) {
        return 0;
    }
    else if ((x < xBn_) or (xBp_ < x)) {
        return 1;
    }
    else {
        const f64 d = std::log(x / (1 - x));
        return (1 - cos((Core::abs(d) - dA_) * dS_)) / 2;
    }
}

// f'(x)*x
f64 UnsupervizedSmoothingFunction::dfx(f64 x) const {
    if ((xAn_ < x) and (x < xAp_)) {
        return 0;
    }
    else if ((x < xBn_) or (xBp_ < x)) {
        return 0;
    }
    else {
        const f64 d = std::log(x / (1 - x));
        return (dS_ / 2) * sin((Core::abs(d) - dA_) * dS_) / (1 - x);
    }
}

void UnsupervizedSmoothingFunction::updateStatistics(f64 x) {
    Precursor::updateStatistics(x);
    if ((xAn_ < x) and (x < xAp_)) {
        ++nAA_;
    }
    else if (x < xBn_) {
        ++nInfB_;
    }
    else if (xBp_ < x) {
        ++nBInf_;
    }
    else if ((xBn_ < x) and (x < xAn_)) {
        ++nBA_;
    }
    else if ((xAp_ < x) and (x < xBp_)) {
        ++nAB_;
    }
    else {
        defect();
    }
}

void UnsupervizedSmoothingFunction::dumpStatistics(Core::XmlWriter& os) const {
    os << Core::XmlOpen("unsupervized-smoothing-function");
    os << Core::XmlOpen("statistics");
    os << Core::XmlFull("n-inf-b", nInfB_);
    os << Core::XmlFull("n-b-a", nBA_);
    os << Core::XmlFull("n-a-a", nAA_);
    os << Core::XmlFull("n-a-b", nAB_);
    os << Core::XmlFull("n-b-inf", nBInf_);
    Precursor::dumpStatistics(os);
    os << Core::XmlClose("statistics");
    os << Core::XmlClose("unsupervized-smoothing-function");
}

/*
 *  factory for smoothing function
 */
Core::Choice SmoothingFunction::choiceType(
        "identity", identity,
        "log", log,
        "sigmoid", sigmoid,
        "unsupervized", unsupervized,
        Core::Choice::endMark());

Core::ParameterChoice SmoothingFunction::paramType(
        "type",
        &choiceType,
        "type of smoothing function f in discriminative training",
        identity);

SmoothingFunction* SmoothingFunction::createSmoothingFunction(
        const Core::Configuration& config) {
    switch (paramType(config)) {
        case SmoothingFunction::identity:
            return new SmoothingFunction();
            break;
        case SmoothingFunction::log:
            return new LogSmoothingFunction(config);
            break;
        case SmoothingFunction::sigmoid:
            return new SigmoidSmoothingFunction(config);
            break;
        case SmoothingFunction::unsupervized:
            return new UnsupervizedSmoothingFunction(config);
            break;
        default:
            return 0;
    }
}
