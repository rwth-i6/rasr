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
#include "LinearFilter.hh"

using namespace Signal;

//===================================================================================================

Core::XmlWriter& LinearFilterParameter::dump(Core::XmlWriter& o) const {
    o << Core::XmlOpen("linear-filter");

    o << Core::XmlOpen("B");
    o << (_float)0 << " ";
    for (std::vector<_float>::const_iterator x = B_tilde_.begin(); x != B_tilde_.end(); ++x)
        o << *x << " ";
    o << Core::XmlClose("B");

    o << Core::XmlOpen("A");
    o << (_float)1 << " ";
    for (std::vector<_float>::const_iterator x = A_tilde_.begin(); x != A_tilde_.end(); ++x)
        o << *x << " ";
    o << Core::XmlClose("A");

    o << Core::XmlOpen("y0");
    for (std::vector<_float>::const_iterator x = y0_.begin(); x != y0_.end(); ++x)
        o << *x << " ";
    o << Core::XmlClose("y0");

    o << Core::XmlClose("linear-filter");
    return o;
}

//===================================================================================================

Core::ParameterString LinearFilterNode::paramB("B", "B polinom where b0 = 0", "");
Core::ParameterString LinearFilterNode::paramA("A", "A polinom where a0 = 1", "");
Core::ParameterString LinearFilterNode::paramY0("y0", "y0 polinom y(-1) ... y(-n)", "");
Core::ParameterInt    LinearFilterNode::paramZeroInputLength("zero-input-length",
                                                          "length of an artificial zero input", 0, 0);
LinearFilterNode::LinearFilterNode(const Core::Configuration& c)
        : Core::Component(c), SleeveNode(c) {
    std::vector<f32> v;
    if (parsePolinom(paramB(c), v))
        setB(v);
    if (parsePolinom(paramA(c), v))
        setA(v);
    if (parsePolinom(paramY0(c), v))
        setY0(v);
    zero_input_length = paramZeroInputLength(c);

    addInput(1);
}

bool LinearFilterNode::setParameter(const std::string& name, const std::string& value) {
    std::vector<f32> v;

    if (paramB.match(name) && parsePolinom(value, v))
        setB(v);
    else if (paramA.match(name) && parsePolinom(value, v))
        setA(v);
    else if (paramY0.match(name) && parsePolinom(value, v))
        setY0(v);
    else if (paramZeroInputLength.match(name))
        zero_input_length = paramZeroInputLength(value);
    else
        return false;

    return true;
}

bool LinearFilterNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());

    Core::Ref<const Flow::Attributes> signalAttributes = getInputAttributes(0);
    if (!configureDatatype(signalAttributes, Flow::Vector<f32>::type()))
        return false;
    attributes->merge(*signalAttributes);

    Core::Ref<const Flow::Attributes> parameterAttributes = getInputAttributes(1);
    if (!configureDatatype(parameterAttributes, LinearFilterParameter::type()))
        return false;
    attributes->merge(*parameterAttributes);

    return putOutputAttributes(0, attributes);
}

bool LinearFilterNode::work(Flow::PortId p) {
    Flow::DataPtr<LinearFilterParameter> param;
    if (getData(1, param)) {
        if (param->getB().size())
            setB(param->getB());
        if (param->getA().size())
            setA(param->getA());
        if (param->getY0().size())
            setY0(param->getY0());
    }
    else if (zero_input_length)
        return putData(0, param.get());

    Flow::DataPtr<Flow::Vector<f32>> in;
    if (zero_input_length) {
        in = Flow::dataPtr(new Flow::Vector<f32>());
        in->resize(zero_input_length, 0.0);
        in->setTimestamp(*param);
    }
    else if (!getData(0, in)) {
        reset();
        return putData(0, in.get());
    }

    in.makePrivate();
    if (!LinearFilter<f32>::work(*in))
        in->dump(criticalError("Frame="));

    return putData(0, in.get());
}

bool LinearFilterNode::parsePolinom(const std::string& value, std::vector<f32>& v) {
    std::string            str(value);
    std::string::size_type pos = 0;

    v.erase(v.begin(), v.end());
    while (!str.empty()) {
        pos = str.find_first_of(" ,;");

        if (pos != 0)
            v.push_back(atof(str.substr(0, pos).c_str()));

        str = (pos == std::string::npos) ? "" : str.substr(pos + 1);
    }

    return v.size();
}
