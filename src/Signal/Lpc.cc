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
#include "Lpc.hh"
#include "LinearFilter.hh"

using namespace Signal;

Core::ParameterInt LinearPredictionCodingNode::paramOrder_B("order-B", "order of numerator",0, 0);
Core::ParameterInt LinearPredictionCodingNode::paramOrder_A("order-A", "order of denominator",0, 0);

LinearPredictionCodingNode::LinearPredictionCodingNode(const Core::Configuration &c) :
    Core::Component(c),
    SleeveNode(c)
{
    setOrder_B(paramOrder_B(c));
    setOrder_A(paramOrder_A(c));

    addInput(1);
    addOutput(1);
}

bool LinearPredictionCodingNode::setParameter(const std::string &name, const std::string &value)
{
    if (paramOrder_B.match(name)) setOrder_B(paramOrder_B(value));
    else if (paramOrder_A.match(name)) setOrder_A(paramOrder_A(value));
    return false;
}

bool LinearPredictionCodingNode::configure()
{
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    Core::Ref<const Flow::Attributes> yAttributes = getInputAttributes(0);
    if (!configureDatatype(yAttributes, Flow::Vector<f32>::type()))
        return false;
    attributes->merge(*yAttributes);

    Core::Ref<const Flow::Attributes> uAttributes = getInputAttributes(1);
    if (!configureDatatype(uAttributes, Flow::Vector<f32>::type()))
        return false;
    attributes->merge(*uAttributes);

    return putOutputAttributes(0, attributes) && putOutputAttributes(1, attributes);
}

Flow::PortId LinearPredictionCodingNode::getInput(const std::string &name)
{
    if (name == "u")
        return 1;
    return 0;
}

Flow::PortId LinearPredictionCodingNode::getOutput(const std::string &name)
{
    if (name == "parameter")
        return 1;
    return 0;
}

bool LinearPredictionCodingNode::work(Flow::PortId p)
{
    bool ret = false;
    Flow::DataPtr<Flow::Vector<f32> > u;
    Flow::DataPtr<Flow::Vector<f32> > y;

    if (!getData(0, y))
        {
            putData(0, y.get());
            if (nOutputLinks(1) > 0)
                putData(1, y.get());

            return true;
        }

    getData(1, u);

    f32 estimation_error;
    std::vector<f32> B_tilde;
    std::vector<f32> A_tilde;

    if (!ArxEstimator::work(u.get(), y.get(), 0, &estimation_error, &B_tilde, &A_tilde))
        y->dump(criticalError("Frame="));

    if (sendVector(*y, estimation_error, B_tilde, A_tilde))
        ret = true;
    if (sendLinearFilterParameter(*y, estimation_error, B_tilde, A_tilde, !u))
        ret = true;

    if (!ret)
        y->dump(criticalError("Frame="));

    return ret;
}

bool LinearPredictionCodingNode::sendVector(const Flow::Timestamp& time_stamp,
                                            const f32& estimation_error,
                                            const std::vector<f32>& B_tilde,
                                            const std::vector<f32>& A_tilde)
{
    if (nOutputLinks(0) == 0)
        return false;

    Flow::Vector<f32>* out = new Flow::Vector<f32>(1 + B_tilde.size() + A_tilde.size());

    u32 i = 0;
    (*out)[i ++] = sqrt(estimation_error);
    for(u8 j = 0; j < B_tilde.size(); j++)
        (*out)[i ++] = B_tilde[j];
    for(u8 j = 0; j < A_tilde.size(); j++)
        (*out)[i ++] = A_tilde[j];

    out->setTimestamp(time_stamp);

    return putData(0, out);
}

bool LinearPredictionCodingNode::sendLinearFilterParameter(const Flow::Timestamp& time_stamp,
                                                           const f32& estimation_error,
                                                           const std::vector<f32>& B_tilde,
                                                           const std::vector<f32>& A_tilde,
                                                           bool initialize)
{
    if (nOutputLinks(1) == 0)
        return false;

    LinearFilterParameter* out = new LinearFilterParameter;

    out->getB() = B_tilde;
    out->getA() = A_tilde;

    if (initialize)
        {
            out->getY0().resize(out->getA().size(), 0.0);
            out->getY0().back() = sqrt(estimation_error) / -out->getA().back();
        }

    out->setTimestamp(time_stamp);

    return putData(1, out);
}
