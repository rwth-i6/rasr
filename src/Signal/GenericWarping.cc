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
#include "GenericWarping.hh"
#include <math.h>

using namespace Signal;

const Core::ParameterString GenericWarpingNode::paramWarpingFunction(
        "warping-function", "warping function declaration");

GenericWarpingNode::GenericWarpingNode(const Core::Configuration& c)
        : Component(c),
          Node(c),
          WarpingNode(c),
          Flow::StringExpressionNode(c, 1) {
    Flow::StringExpressionNode::setTemplate(paramWarpingFunction(c));
}

GenericWarpingNode::~GenericWarpingNode() {}

void GenericWarpingNode::updateWarping() {
    Math::AnalyticFunctionFactory factory(select(paramWarpingFunction.name()));
    factory.setSampleRate(sampleRate_);
    factory.setDomainType(Math::AnalyticFunctionFactory::discreteDomain);
    factory.setMaximalArgument(inputSize_ - 1);
    Math::UnaryAnalyticFunctionRef warpingFunction = factory.createUnaryFunction(Flow::StringExpressionNode::value());
    if (!warpingFunction)
        criticalError("Could not create warping function.");

    if (interpolateOverWarpedAxis_) {
        warping_.setWarpingFunction(warpingFunction, inputSize_, mergeType_, interpolationType_);
    }
    else {
        Math::UnaryAnalyticFunctionRef inverseWarpingFunction = warpingFunction->invert();
        if (!inverseWarpingFunction)
            criticalError("Warping function is not invertable.");
        warping_.setInverseWarpingFunction(inverseWarpingFunction, inputSize_, interpolationType_);
    }
}

bool GenericWarpingNode::configure() {
    Flow::Attributes attributes;
    return StringExpressionNode::configure(attributes) && WarpingNode::configure(attributes);
}

bool GenericWarpingNode::setParameter(const std::string& name, const std::string& value) {
    if (paramWarpingFunction.match(name))
        Flow::StringExpressionNode::setTemplate(paramWarpingFunction(value));
    else
        return WarpingNode::setParameter(name, value);
    return true;
}

void GenericWarpingNode::apply(const Flow::Vector<f32>& in, std::vector<f32>& out) {
    if (StringExpressionNode::update(in))
        updateWarping();
    warping_.apply(in, out);
}
