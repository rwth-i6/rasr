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
#include "AllPolesPowerSpectrum.hh"
#include "ArEstimator.hh"

using namespace Signal;

const Core::ParameterInt AllPolesPowerSpectrumNode::paramDiscreteTotalLength(
    "total-length", "total length of power spectrum in discrete units", 0, 0);

const Core::ParameterFloat AllPolesPowerSpectrumNode::paramContinuousTotalLength(
    "continuous-total-length", "total length of power spectrum in continuous units", 0, 0);

AllPolesPowerSpectrumNode::AllPolesPowerSpectrumNode(const Core::Configuration &c) :
    Core::Component(c), Precursor(c),
    discreteTotalLength_(0), continuousTotalLength_(0), totalLength_(0)
{
    discreteTotalLength_ = paramDiscreteTotalLength(c);
    continuousTotalLength_ = paramContinuousTotalLength(c);
}

AllPolesPowerSpectrumNode::~AllPolesPowerSpectrumNode()
{}

bool AllPolesPowerSpectrumNode::configure()
{
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);
    if (!configureDatatype(attributes, AutoregressiveCoefficients::type()))
        return false;

    f64 sampleRate = atof(attributes->get("sample-rate").c_str());
    if (sampleRate <= 0)
        error("Sample rate (%f) is smaller or equal to 0.", sampleRate);
    init(sampleRate);
    attributes->set("sample-rate", totalLength_ / sampleRate);

    attributes->set("datatype", Flow::Vector<f32>::type()->name());

    respondToDelayedErrors();
    return putOutputAttributes(0, attributes);

}

void AllPolesPowerSpectrumNode::init(f64 sampleRate)
{
    sampleRate_ = sampleRate;

    totalLength_ = (u32)ceil(continuousTotalLength_ * sampleRate_);
    if (discreteTotalLength_ != 0) {
        if (continuousTotalLength_ != 0 && totalLength_ != discreteTotalLength_) {
            warning("continuous-total-length (%f) will be overwitten by parameter total-length (%zd).",
                    continuousTotalLength_, discreteTotalLength_);
        }
        totalLength_ = discreteTotalLength_;
    }
    if (totalLength_ == 0) error("Total length should be at least one.");
}

bool AllPolesPowerSpectrumNode::setParameter(const std::string &name, const std::string &value)
{
    if (paramDiscreteTotalLength.match(name))
        discreteTotalLength_ = paramDiscreteTotalLength(value);
    else if (paramContinuousTotalLength.match(name))
        continuousTotalLength_ = paramContinuousTotalLength(value);
    else
        return false;
    return true;
}

bool AllPolesPowerSpectrumNode::work(Flow::PortId p)
{
    Flow::DataPtr<AutoregressiveCoefficients> arCoefficients;
    if (getData(0, arCoefficients)) {
        Flow::Vector<f32> *out = new Flow::Vector<f32>;
        out->setTimestamp(*arCoefficients);
        allPolesPowerSpectrum(arCoefficients->gain(), arCoefficients->a(), totalLength_, *out);
        std::transform(out->begin(), out->end(), out->begin(),
                       std::bind2nd(std::divides<f32>(), sampleRate_ * sampleRate_));
        return putData(0, out);
    }
    return putData(0, arCoefficients.get());
};
