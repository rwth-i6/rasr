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
#include "LinearWarping.hh"
#include <Math/AnalyticFunctionFactory.hh>
#include <Math/PiecewiseLinearFunction.hh>

using namespace Signal;

Core::ParameterFloat LinearWarpingNode::paramWarpingFactor("warping-factor", "warping factor", 1);

Core::ParameterFloat LinearWarpingNode::paramWarpingLimit("warping-limit", "warping limit", 0.875, 0.0000000001, .99999999999);

const f64 LinearWarpingNode::hash::strechFactor = 100.0;

size_t LinearWarpingNode::hash::operator()(f64 x) const {
    f64 result = rint(x * strechFactor);
    verify_(result > 0);

    return (size_t)result;
}

LinearWarpingNode::LinearWarpingNode(const Core::Configuration& c)
        : Component(c),
          Node(c),
          Predecessor(c),
          warpingLimit_(0) {
    setWarpingFactor(paramWarpingFactor(c));
    setWarpingLimit(paramWarpingLimit(c));
}

void LinearWarpingNode::setWarpingLimit(f64 warpingLimit) {
    if (warpingLimit_ != warpingLimit) {
        warpingLimit_ = warpingLimit;
        setNeedInit();
    }
}

const Warping& LinearWarpingNode::warping() {
    WarpingCache::iterator w = warpingCache_.find(warpingFactor_());
    if (w == warpingCache_.end())
        w = warpingCache_.insert(std::make_pair(warpingFactor_(), createWarping())).first;
    return *(w->second);
}

Warping* LinearWarpingNode::createWarping() {
    if (warpingFactor_() <= 0)
        error("Cannot warp with factor %f.", warpingFactor_());
    if (warpingLimit_ <= 0 || warpingLimit_ >= 1)
        error("Cannot warp with limit %f.", warpingLimit_);
    respondToDelayedErrors();

    Warping* result = new Warping;

    Math::AnalyticFunctionFactory factory(select("warping-function"));
    factory.setSampleRate(sampleRate_);
    factory.setDomainType(Math::AnalyticFunctionFactory::discreteDomain);
    factory.setMaximalArgument(inputSize_ - 1);
    Math::UnaryAnalyticFunctionRef warpingFunction = factory.createTwoPieceLinearFunction(warpingFactor_(), warpingLimit_);
    ensure(warpingFunction);

    if (interpolateOverWarpedAxis_)
        result->setWarpingFunction(warpingFunction, inputSize_, mergeType_, interpolationType_);
    else
        result->setInverseWarpingFunction(warpingFunction->invert(), inputSize_, interpolationType_);
    return result;
}

void LinearWarpingNode::clear() {
    for (WarpingCache::iterator w = warpingCache_.begin(); w != warpingCache_.end(); ++w)
        delete w->second;

    warpingCache_.clear();
}

void LinearWarpingNode::reset() {
    warpingFactor_.setStartTime(Core::Type<Time>::min);
    warpingFactor_.setEndTime(warpingFactor_.startTime());
}

Flow::PortId LinearWarpingNode::getInput(const std::string& name) {
    if (name == "warping-factor") {
        addInput(1);
        return 1;
    }
    return Predecessor::getInput(name);
}

bool LinearWarpingNode::setParameter(const std::string& name, const std::string& value) {
    if (paramWarpingFactor.match(name))
        setWarpingFactor(paramWarpingFactor(value));
    else if (paramWarpingLimit.match(name))
        setWarpingLimit(paramWarpingLimit(value));
    else
        return Predecessor::setParameter(name, value);
    return true;
}

bool LinearWarpingNode::configure() {
    reset();

    Flow::Attributes warpingFactorAttributes;
    if (nInputs() >= 2) {
        Core::Ref<const Flow::Attributes> a = getInputAttributes(1);
        if (configureDatatype(a, Flow::Float64::type()))
            warpingFactorAttributes.merge(*a);
        else
            return false;
    }

    return Predecessor::configure(warpingFactorAttributes);
}

void LinearWarpingNode::apply(const Flow::Vector<f32>& in, std::vector<f32>& out) {
    if (nInputs() >= 2)
        updateWarpingFactor(in);
    warping().apply(in, out);
}

void LinearWarpingNode::updateWarpingFactor(const Flow::Timestamp& featureTimeStamp) {
    verify(nInputs() >= 2);

    Flow::DataPtr<Flow::Float64> in;
    while (!warpingFactor_.contains(featureTimeStamp)) {
        if (getData(1, in)) {
            warpingFactor_ = *in;
        }
        else {
            criticalError("Warping factor stream stopped before start-time (%f).",
                          featureTimeStamp.startTime());
        }
    }
}
