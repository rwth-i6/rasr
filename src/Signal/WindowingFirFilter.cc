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
#include "WindowingFirFilter.hh"
#include "Utility.hh"

using namespace Core;
using namespace Signal;

// WindowingFirFilter
/////////////////////

WindowingFirFilter::WindowingFirFilter()
        : lengthInS_(5.0),
          overshoot_(0.01),
          transitionRegionWidthInHz_(400),
          M_(0),
          removeDelay_(false) {}

void WindowingFirFilter::setSampleRate(f64 sampleRate) {
    if (WindowBuffer::sampleRate() != sampleRate) {
        WindowBuffer::setSampleRate(sampleRate);
        setNeedInit();
    }
}

void WindowingFirFilter::setCutOff(const std::vector<std::pair<f64, f64>>& cutOff) {
    cutOff_ = cutOff;

    setNeedInit();
}

void WindowingFirFilter::setOvershoot(f64 overshoot) {
    overshoot_ = overshoot;

    setNeedInit();
}

void WindowingFirFilter::setTransitionRegionWidthInHz(f64 transitionRegionWidthInHz) {
    transitionRegionWidthInHz_ = transitionRegionWidthInHz;

    setNeedInit();
}

void WindowingFirFilter::setLengthInS(Time length) {
    if (lengthInS_ != length) {
        lengthInS_ = length;

        setNeedInit();
    }
}

void WindowingFirFilter::getFilterResponse(std::vector<Sample>& h) {
    verify(!cutOff_.empty());

    h.resize(M_ + 1);

    for (u32 n = 0; n <= M_ / 2; n++) {
        h[n] = h[M_ - n] = 0;

        for (u8 k = 1; k <= cutOff_.size(); k++) {
            f64 omega = cutOff_[k - 1].second * 2 * M_PI / (f64)sampleRate();

            h[n] = h[M_ - n] += (cutOff_[k - 1].first - (k != cutOff_.size() ? cutOff_[k].first : 0.0)) *
                                (omega * sinc(omega * ((Sample)n - ((Sample)M_ / (Sample)2))) / (Sample)M_PI);
        }
    }

    window_.setBeta(getBeta(overshoot_));
    window_.setLength(M_ + 1);

    if (!window_.work(h.begin(), h.end()))
        defect();
}

f64 WindowingFirFilter::getBeta(f64 overshoot) {
    f64 beta;

    f64 A = -20.0 * log10(overshoot);

    if (A > 50.0)
        beta = 0.1102 * (A - 8.7);
    else if (A >= 21.0)
        beta = 0.5842 * exp(0.4 * std::log(A - 21.0)) + 0.07886 * (A - 21.0);
    else
        beta = 0;

    return beta;
}

u32 WindowingFirFilter::getM(f64 overshoot, f64 deltaOmega) {
    f64 A = -20.0 * log10(overshoot);

    u32 M = (u32)((A - 8) / 2.285 / deltaOmega);

    M = u32(M / 2) * 2;  //make M even

    verify(M > 0);

    return M;
}

void WindowingFirFilter::init() {
    verify(sampleRate() > 0);

    M_ = getM(overshoot_, 2 * M_PI * transitionRegionWidthInHz_ / sampleRate());

    setShift((u32)rint(lengthInS_ * sampleRate()));

    setLength(shift() + M_);

    std::vector<Sample> response;

    getFilterResponse(response);

    convolution_.setResponse(response, length());

    Predecessor::init();
}

void WindowingFirFilter::transform(Flow::Vector<Sample>& out) {
    if (nOutputs() == 1)
        convolution_.setOutputBegin(removeDelay_ ? delay() : 0);
    else
        convolution_.setOutputBegin(M_);

    if (!flushed())
        convolution_.setOutputEnd(out.size());
    else
        convolution_.setOutputEnd(out.size() + (removeDelay_ ? delay() : M_));

    convolution_.apply(out);

    out.setStartTime(out.startTime() +
                     ((Time)convolution_.outputBegin() - (Time)delay()) / sampleRate());
    out.setEndTime(out.startTime() + (Time)out.size() / sampleRate());
}

// WindowingFirFilterParameter
//////////////////////////////

Core::XmlWriter& WindowingFirFilterParameter::dump(Core::XmlWriter& o) const {
    o << Core::XmlOpen("windowing-fir-filter");
    for (u8 i = 0; i < cut_off_.size(); i++) {
        o << Core::XmlEmpty("cut-off") + Core::XmlAttribute("gain", cut_off_[i].first) + Core::XmlAttribute("omega", cut_off_[i].second);
    }
    o << Core::XmlClose("windowing-fir-filter");
    return o;
}

bool WindowingFirFilterParameter::read(BinaryInputStream& i) {
    u32 size;
    i >> size;
    cut_off_.resize(size);
    for (std::vector<std::pair<f64, f64>>::iterator it = cut_off_.begin(); it != cut_off_.end(); ++it) {
        i >> it->first;
        i >> it->second;
    }
    return i.good();
}

bool WindowingFirFilterParameter::write(BinaryOutputStream& o) const {
    o << (u32)cut_off_.size();
    for (std::vector<std::pair<f64, f64>>::const_iterator it = cut_off_.begin(); it != cut_off_.end(); ++it) {
        o << it->first;
        o << it->second;
    }
    return o.good();
}

// WindowingFirFilterNode
/////////////////////////

Core::ParameterFloat WindowingFirFilterNode::paramLength("length", "length of output", 5.0);

Core::ParameterBool WindowingFirFilterNode::paramRemoveDelay("remove-delay", "removes padding from the beginning and the end of the output", false);

Core::ParameterString WindowingFirFilterNode::paramCutOff("cut-off", "gain1:omega1;gain2:omega2;... ", "");

Core::ParameterFloat WindowingFirFilterNode::paramOvershoot("overshoot", "max amplitude oscillation in pass band and in stop band interval", .01, 0);

Core::ParameterFloat WindowingFirFilterNode::paramTransitionRegionWidth("transition-region-width", "width of transition region in Hz", 400, 0);

WindowingFirFilterNode::WindowingFirFilterNode(const Core::Configuration& c)
        : Component(c),
          Predecessor(c) {
    setLengthInS(paramLength(c));

    setRemoveDelay(paramRemoveDelay(c));

    std::vector<std::pair<f64, f64>> cut_off;
    if (parseCutOff(paramCutOff(c), cut_off))
        setCutOff(cut_off);

    setOvershoot(paramOvershoot(c));

    setTransitionRegionWidthInHz(paramTransitionRegionWidth(c));
}

bool WindowingFirFilterNode::setParameter(const std::string& name, const std::string& value) {
    std::vector<std::pair<f64, f64>> cut_off;

    if (paramLength.match(name))
        setLengthInS(paramLength(value));
    else if (paramRemoveDelay.match(name))
        setRemoveDelay(paramRemoveDelay(value));
    else if (paramCutOff.match(name) && parseCutOff(paramCutOff(value), cut_off))
        setCutOff(cut_off);
    else if (paramOvershoot.match(name))
        setOvershoot(paramOvershoot(value));
    else if (paramTransitionRegionWidth.match(name))
        setTransitionRegionWidthInHz(paramTransitionRegionWidth(value));
    else
        return false;

    return true;
}

bool WindowingFirFilterNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes);

    Core::Ref<const Flow::Attributes> signalAttributes = getInputAttributes(0);
    if (!configureDatatype(signalAttributes, Flow::Vector<f32>::type()))
        return false;
    attributes->merge(*signalAttributes);

    if (nInputs() > 1) {
        Core::Ref<const Flow::Attributes> parameterAttributes = getInputAttributes(1);
        if (!configureDatatype(parameterAttributes, WindowingFirFilterParameter::type()))
            return false;
        attributes->merge(*parameterAttributes);
    }

    setSampleRate(atof(signalAttributes->get("sample-rate").c_str()));

    reset();

    return putOutputAttributes(0, attributes);
}

Flow::PortId WindowingFirFilterNode::getInput(const std::string& name) {
    if (name == "parameter") {
        addInput(1);
        return 1;
    }

    return 0;
}

bool WindowingFirFilterNode::work(Flow::PortId p) {
    if (nInputs() > 1) {
        Flow::DataPtr<WindowingFirFilterParameter> param;
        if (getData(1, param)) {
            if (param->getCutOff().size())
                setCutOff(param->getCutOff());
        }
    }

    return Predecessor::work(p);
}

bool WindowingFirFilterNode::parseCutOff(const std::string& value, std::vector<std::pair<f64, f64>>& cutOff) {
    std::string            str(value);
    std::string            element;
    std::string::size_type pos;

    char*               error;
    std::pair<f64, f64> gainFrequencyPair;
    cutOff.clear();
    while (!str.empty()) {
        pos = str.find_first_of(";");  // semicolon-separated elements

        if (pos != 0) {
            element = str.substr(0, pos);  // element is gain:frequency

            gainFrequencyPair.first = strtod(element.c_str(), &error);
            if (*error != ':')
                return false;
            element                  = error + 1;
            gainFrequencyPair.second = strtod(element.c_str(), &error);

            cutOff.push_back(gainFrequencyPair);
        }

        str.erase(0, (pos == std::string::npos) ? pos : pos + 1);
    }

    return cutOff.size();
}
