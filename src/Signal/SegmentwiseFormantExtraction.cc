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
#include "SegmentwiseFormantExtraction.hh"
#include "ArEstimator.hh"

using namespace Signal;

//=================================================================================================

bool SegmentwiseFormantExtraction::init() {
    if (!FormantExtraction::init() || !segmentwise_estimator_ || !segment_estimator_)
        return false;

    segmentwise_estimator_->setOrder(order_);

    segment_estimator_->setStep(step_);
    segment_estimator_->setMaxNumberOfSegments(max_nr_segment_);
    if (!segment_estimator_->setSearchInterval((_float)max_frequency_ / ((_float)getSampleRate() / 2.0)))
        return false;

    return !(need_init_ = false);
}

void SegmentwiseFormantExtraction::setSegmentwiseEstimator(DomainType type) {
    if (segmentwise_estimator_)
        delete segmentwise_estimator_;

    switch (type) {
        case frequency:
            segmentwise_estimator_ = new SegmentwiseArEstimator;
            break;
        default:
            defect();
    }
    need_init_ = true;
}

void SegmentwiseFormantExtraction::setSegmentEstimator(SegmentEstimatorType type) {
    if (segment_estimator_)
        delete segment_estimator_;

    switch (type) {
        case dynamic_programing:
            segment_estimator_ = new DynamicProgramingSegmentEstimator;
            break;
        default:
            defect();
    }

    need_init_ = true;
}

bool SegmentwiseFormantExtraction::work(const Flow::Vector<_float>& in,
                                        const std::vector<s32>&     segments,
                                        std::vector<Formant>&       out) {
    if (segments.size() == 1) {
        in.dump(warning("No segments found. Zero formants generated. Frame="));

        out.clear();
        out.resize(getMaxNumberOfSegments());
        return true;
    }

    out.resize(segments.size() - 1);

    std::vector<_float> A_tilde;
    _float              estimation_error;
    _float              energy;

    for (u8 i = 1; i < segments.size(); i++) {
        segmentwise_estimator_->setSegment(segments[i - 1], segments[i]);
        if (!segmentwise_estimator_->work(&estimation_error, &A_tilde, &energy) ||
            A_tilde.size() != 2 ||
            !calculateProperties(in, i - 1, estimation_error, A_tilde[0], A_tilde[1], energy, out[i - 1]))
            return false;
    }
    return true;
}

bool SegmentwiseFormantExtraction::work(const Flow::Vector<_float>&         in,
                                        const std::vector<s32>&             segments,
                                        std::vector<LinearFilterParameter>& out) {
    if (segments.size() == 1) {
        in.dump(warning("No segments found. Zero parameters generated. Frame="));

        out.clear();
        out.resize(getMaxNumberOfSegments());
        return true;
    }

    out.resize(segments.size() - 1);

    _float estimation_error;
    for (u8 i = 1; i < segments.size(); i++) {
        out[i - 1].getB().clear();
        out[i - 1].getY0().clear();
        out[i - 1].getY0().resize(order_, 0.0);

        segmentwise_estimator_->setSegment(segments[i - 1], segments[i]);
        if (!segmentwise_estimator_->work(&estimation_error, &(out[i - 1].getA()), 0))
            return false;

        out[i - 1].getY0().back() = sqrt(estimation_error) / -out[i - 1].getA().back();
    }
    return true;
}

//=================================================================================================

Core::Choice SegmentwiseFormantExtractionNode::domain_choice(
        "frequency", frequency,
        Core::Choice::endMark());
Core::ParameterChoice SegmentwiseFormantExtractionNode::paramDomain("domain", &domain_choice, "calculation domain", frequency);

Core::Choice SegmentwiseFormantExtractionNode::segment_estimator_choice(
        "dinamic-programming", dynamic_programing,
        Core::Choice::endMark());
Core::ParameterChoice SegmentwiseFormantExtractionNode::paramSegmentEstimator("segment-estimator", &segment_estimator_choice, "segment estimator type", dynamic_programing);

Core::ParameterInt SegmentwiseFormantExtractionNode::paramOrder("order", "LPC order for one segment", 2, 0);
Core::ParameterInt SegmentwiseFormantExtractionNode::paramMaxNumberOfSegments("max-number-segment", "maximum number of segments");
Core::ParameterInt SegmentwiseFormantExtractionNode::paramStep("step", "devide frequency resolution by step", 1, 0);
Core::ParameterInt SegmentwiseFormantExtractionNode::paramMaxFrequency("max-frequency", "frequency range to search", 5000, 0);

SegmentwiseFormantExtractionNode::SegmentwiseFormantExtractionNode(const Core::Configuration& c)
        : Core::Component(c), SleeveNode(c), SegmentwiseFormantExtraction(c) {
    setOrder(paramOrder(c));
    setMaxNumberOfSegments(paramMaxNumberOfSegments(c));
    setStep(paramStep(c));
    setMaxFrequency(paramMaxFrequency(c));
    setSegmentwiseEstimator(DomainType(paramDomain(c)));
    setSegmentEstimator(SegmentEstimatorType(paramSegmentEstimator(c)));

    addOutput(0);
}

bool SegmentwiseFormantExtractionNode::setParameter(const std::string& name, const std::string& value) {
    if (paramOrder.match(name))
        setOrder(paramOrder(value));
    else if (paramMaxNumberOfSegments.match(name)) {
        setMaxNumberOfSegments(paramMaxNumberOfSegments(value));
    }
    else if (paramStep.match(name)) {
        setStep(paramStep(value));
    }
    else if (paramMaxFrequency.match(name)) {
        setMaxFrequency(paramMaxFrequency(value));
    }
    else if (paramDomain.match(name))
        setSegmentwiseEstimator(DomainType(paramDomain(value)));
    else if (paramSegmentEstimator.match(name))
        setSegmentEstimator(SegmentEstimatorType(paramSegmentEstimator(value)));
    else
        return false;

    return true;
}

bool SegmentwiseFormantExtractionNode::configure() {
    Core::Ref<Flow::Attributes> a(new Flow::Attributes());
    getInputAttributes(0, *a);
    if (!configureDatatype(a, Flow::Vector<f32>::type()))
        return false;

    setSampleRate(atoi(a->get("sample-rate").c_str()));

    // sample rate cannot be interpreted on formant vectors
    a->set("sample-rate", Core::Type<f64>::min);

    bool status = true;
    for (Flow::PortId i = 0; i < nOutputs(); i++) {
        if (!putOutputAttributes(i, a))
            status = false;
    }
    return status;
}

Flow::PortId SegmentwiseFormantExtractionNode::getOutput(const std::string& name) {
    std::string tmp("linear-filter-parameter-");
    if (name.find(tmp) == 0) {
        s32 port = atoi(name.substr(tmp.size()).c_str());
        if (port < 1 || port > getMaxNumberOfSegments())
            return Flow::IllegalPortId;
        else
            return port;
    }
    return 0;
}

bool SegmentwiseFormantExtractionNode::work(Flow::PortId p) {
    bool                             ret = false;
    Flow::DataPtr<Flow::Vector<f32>> in;

    if (!getData(0, in)) {
        putData(0, in.get());

        for (u8 i = 0; i < getMaxNumberOfSegments(); i++)
            putData(1 + i, in.get());

        return true;
    }

    std::vector<s32> segments;
    if (!SegmentwiseFormantExtraction::work(*in, segments))
        in->dump(SegmentwiseFormantExtraction::criticalError("Frame="));

    if (sendFormant(*in, segments))
        ret = true;
    if (sendLinearFilterParameter(*in, segments))
        ret = true;

    if (!ret)
        in->dump(SegmentwiseFormantExtraction::criticalError("Frame="));

    return ret;
}

bool SegmentwiseFormantExtractionNode::sendFormant(const Flow::Vector<_float>& in,
                                                   const std::vector<s32>&     segments) {
    if (nOutputLinks(0) == 0)
        return false;

    Flow::Vector<f32>*   out = new Flow::Vector<f32>(getMaxNumberOfSegments() * 4);
    std::vector<Formant> formant_structure;

    if (!SegmentwiseFormantExtraction::work(in, segments, formant_structure))
        return false;

    if (formant_structure.size() != getMaxNumberOfSegments())
        return false;

    _float energy_scale = 1000.0;

    for (u8 i = 0; i < getMaxNumberOfSegments(); i++) {
        (*out)[4 * i + 0] = formant_structure[i].frequency_;
        (*out)[4 * i + 1] = formant_structure[i].bandwidth_;
        (*out)[4 * i + 2] = log10(formant_structure[i].amplitude_) * energy_scale;
        (*out)[4 * i + 3] = log10(formant_structure[i].energy_) * energy_scale;
    }

    out->setTimestamp(in);

    return putData(0, out);
}

bool SegmentwiseFormantExtractionNode::sendLinearFilterParameter(const Flow::Vector<_float>& in,
                                                                 const std::vector<s32>&     segments) {
    bool ret = false;

    std::vector<LinearFilterParameter> param;
    for (u8 i = 0; i < getMaxNumberOfSegments(); i++) {
        if (nOutputLinks(1 + i) != 0) {
            if (param.size() == 0 && !SegmentwiseFormantExtraction::work(in, segments, param))
                return false;

            if (i < param.size()) {
                param[i].setTimestamp(in);
                if (putData(1 + i, param[i].clone()))
                    ret = true;
            }
            else
                return false;
        }
    }
    return ret;
}

void SegmentwiseFormantExtractionNode::setMaxNumberOfSegments(u8 max_nr_segment) {
    addOutputs(1 + max_nr_segment);
    SegmentwiseFormantExtraction::setMaxNumberOfSegments(max_nr_segment);
}
