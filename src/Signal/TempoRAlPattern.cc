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
#include "TempoRAlPattern.hh"

using namespace Signal;

//============================================================================

TemporalPattern::TemporalPattern()
        : needInit_(true),
          nFeatures_(0),
          nFrames_(0),
          dctSize_(0),
          windowFunction_(0) {}

TemporalPattern::~TemporalPattern() {
    if (windowFunction_)
        delete windowFunction_;
}

void TemporalPattern::setWindowFunction(Signal::WindowFunction* windowFunction) {
    if (windowFunction_)
        delete windowFunction_;
    windowFunction_ = windowFunction;
}

bool TemporalPattern::init() {
    // init/check all the member values
    if ((nFrames() == 0) || (nFeatures() == 0) || (dctSize() == 0) || (!(dctSize() < nFrames())))
        return false;

    // init the window/cosine function
    windowFunction()->setLength((u32)nFrames());
    cosineTransform().init(Signal::CosineTransform::NplusOneData, nFrames(), (size_t)dctSize(), true);

    // init the final dct vector
    dctVec.resize(dctSize());
    bandVec.resize(nFrames());

    // all inits done
    needInit_ = false;

    return true;
}

bool TemporalPattern::apply(std::vector<Value>& in, std::vector<Value>& out) {
    // init is requested, initialization failed?
    if ((needInit_) && (!init()))
        return false;

    for (size_t band = 0; band < nFeatures(); band++) {
        getBand(band, in, bandVec);
        applyWindow(bandVec);
        applyDCT(bandVec, dctVec);
        setBand(band, dctVec, out);
    }

    return true;
}

void TemporalPattern::getBand(size_t band, std::vector<Value>& in, std::vector<Value>& out) {
    // Copy each frame of a band.
    for (Flow::Vector<Value>::iterator it = out.begin(), itv = (in.begin() + band);
         it != out.end();
         it++, itv += nFeatures())
        *it = *itv;
}

void TemporalPattern::setBand(size_t band, std::vector<Value>& in, std::vector<Value>& out) {
    // Copy value to the end of the output vector.
    for (Flow::Vector<Value>::iterator it = in.begin(), itv = (out.begin() + band);
         it != in.end();
         it++, itv += nFeatures())
        *itv = *it;
}

void TemporalPattern::applyWindow(std::vector<Value>& in) {
    windowFunction()->work(in.begin(), in.end());
}

void TemporalPattern::applyDCT(std::vector<Value>& in, std::vector<Value>& out) {
    cosineTransform().apply(in, out);
}

//============================================================================
const Core::ParameterInt TemporalPatternNode::paramContextLength(
        "context-length", "number of frames", 51);
const Core::ParameterInt TemporalPatternNode::paramOutputSize(
        "output-size", "number of final frames", 16);

TemporalPatternNode::TemporalPatternNode(const Core::Configuration& c)
        : Core::Component(c), Precursor(c), needInit_(true) {
    setOutputSize(paramOutputSize(c));
    setContextLength(paramContextLength(c));
}

TemporalPatternNode::~TemporalPatternNode() {}

bool TemporalPatternNode::setParameter(const std::string& name, const std::string& value) {
    if (paramContextLength.match(name))
        setContextLength(paramContextLength(value));
    else if (paramOutputSize.match(name))
        setOutputSize(paramOutputSize(value));
    else if (Signal::WindowFunction::paramType.match(name))
        setWindowFunction(Signal::WindowFunction::create((Signal::WindowFunction::Type)Signal::WindowFunction::paramType(value)));
    else
        return false;

    return true;
}

bool TemporalPatternNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);

    if (!configureDatatype(attributes, Flow::Vector<Value>::type()))
        return false;

    attributes->set("datatype", Flow::Vector<Value>::type()->name());
    return putOutputAttributes(0, attributes);
}

bool TemporalPatternNode::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<Value>> ptrFeatures;

    if (getData(0, ptrFeatures)) {
        if (needInit_)
            init(ptrFeatures.get()->size());

        // generate features
        Flow::Vector<Value>* out = new Flow::Vector<Value>(nFeatures() * outputSize_);
        apply(*(ptrFeatures.get()), *out);

        out->setTimestamp(*ptrFeatures);
        return putData(0, out);
    }
    return putData(0, ptrFeatures.get());
};

void TemporalPatternNode::init(size_t length) {
    // length = nFrames * nElements
    TemporalPattern::init((size_t)(length / contextLength_), contextLength_, outputSize_);

    // check the size of the incoming/outgoing elements, report errors
    if (outputSize_ == 0) {
        error("Incorrect output size (%zd). Output size > %zd.", outputSize_, length + 1);
    }
    respondToDelayedErrors();

    // all inits done
    needInit_ = false;
}
