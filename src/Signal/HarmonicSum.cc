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
#include "HarmonicSum.hh"

using namespace Core;
using namespace Signal;


// HarmonicSumNode
//////////////////


ParameterFloat HarmonicSumNode::paramSize
("size", "size in continous unit depending on previous nodes (0: use input size)", 0, 0);

ParameterInt HarmonicSumNode::paramH
("H", "max number of harmonics (0: use all harmonics)", 0, 0);


HarmonicSumNode::HarmonicSumNode(const Configuration &c) :
    Core::Component(c), SleeveNode(c),
    continuousSize_(0),
    size_(0),
    H_(0),
    sampleRate_(0),
    needInit_(true)
{
    setContinuousSize(paramSize(c));
    setH(paramH(c));
}



bool HarmonicSumNode::configure() {
    Core::Ref<const Flow::Attributes> a = getInputAttributes(0);
    if (!configureDatatype(a, Flow::Vector<f32>::type()))
        return false;

    setSampleRate(atof(a->get("sample-rate").c_str()));

    return putOutputAttributes(0, a);
}

bool HarmonicSumNode::setParameter(const std::string &name, const std::string &value) {

    if (paramSize.match(name))
        setContinuousSize(paramSize(value));
    else if (paramH.match(name))
        setH(paramH(value));
    else
        return false;

    return true;
}


void HarmonicSumNode::init() {

    if (H_ == 0)
        criticalError("Please set the maximum shrinkage (H)");

    if (sampleRate_ <= 0)
        criticalError("Sample rate (%f) is smaller or equal to 0.", sampleRate_);

    size_ = (u32)rint(continuousSize_ * sampleRate_) + 1;

    needInit_ = false;
}


void HarmonicSumNode::initOutput(const std::vector<f32> &x, std::vector<f32> &s) {

    if (continuousSize_ > 0.0 && x.size() < size_) {
        criticalError("Input data length (%zd) is smaller then harmonic sum length (%d).",
                      x.size(), size_);
    }

    s.resize(continuousSize_ > 0.0 ? size_ : x.size());
}


bool HarmonicSumNode::work(Flow::PortId p) {

    Flow::DataPtr<Flow::Vector<f32> > in;
    if (!getData(0, in))
        return SleeveNode::putData(0, in.get());

    Flow::Vector<f32> *out = new Flow::Vector<f32>;

    if (needInit_)
        init();

    initOutput(*in, *out);

    apply(*in, *out);

    out->setTimestamp(*in);

    return putData(0, out);
}
