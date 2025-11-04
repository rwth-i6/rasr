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
#ifndef _SIGNAL_SEGMENTWISEFORMANTEXTRACTION_HH
#define _SIGNAL_SEGMENTWISEFORMANTEXTRACTION_HH

#include <Core/Parameter.hh>
#include <Flow/Vector.hh>

#include "Formant.hh"
#include "LinearFilter.hh"
#include "Node.hh"
#include "SegmentEstimator.hh"

namespace Signal {

class SegmentwiseFormantExtraction : public FormantExtraction {
public:
    typedef f32 _float;
    enum DomainType {
        frequency
    };
    enum SegmentEstimatorType {
        dynamic_programing
    };

private:
    u8  order_;
    u8  max_nr_segment_;
    s32 step_;
    u32 max_frequency_;

    SegmentwiseEstimator*              segmentwise_estimator_;
    DynamicProgramingSegmentEstimator* segment_estimator_;

    bool need_init_;

    bool init();

public:
    SegmentwiseFormantExtraction(const Core::Configuration& c)
            : Core::Component(c),
              FormantExtraction(c),
              order_(0),
              max_nr_segment_(0),
              step_(0),
              segmentwise_estimator_(0),
              segment_estimator_(0) {}
    virtual ~SegmentwiseFormantExtraction() {
        if (segmentwise_estimator_)
            delete segmentwise_estimator_;
        if (segment_estimator_)
            delete segment_estimator_;
    }

    void setMaxNumberOfSegments(u8 max_nr_segment) {
        if (max_nr_segment_ != max_nr_segment) {
            max_nr_segment_ = max_nr_segment;
            setMaxNrFormant(max_nr_segment);
            need_init_ = true;
        }
    }
    u8 getMaxNumberOfSegments() {
        return max_nr_segment_;
    }

    void setMaxFrequency(u32 max_frequency) {
        if (max_frequency_ != max_frequency) {
            max_frequency_ = max_frequency;
            ;
            need_init_ = true;
        }
    }

    void setOrder(u8 order) {
        if (order_ != order) {
            order_     = order;
            need_init_ = true;
        }
    }
    void setStep(s32 step) {
        if (step_ != step) {
            step_      = step;
            need_init_ = true;
        }
    }
    void setSegmentwiseEstimator(DomainType type);
    void setSegmentEstimator(SegmentEstimatorType type);

    bool work(const std::vector<_float>& in, std::vector<s32>& segments) {
        if (need_init_ && !init())
            return false;

        if (!segmentwise_estimator_->setSignal(in))
            return false;
        segment_estimator_->setSegmentwiseEstimator(segmentwise_estimator_);

        return segment_estimator_->work(segments);
    }

    bool work(const Flow::Vector<_float>& in,
              const std::vector<s32>&     segments,
              std::vector<Formant>&       out);
    bool work(const Flow::Vector<_float>&         in,
              const std::vector<s32>&             segments,
              std::vector<LinearFilterParameter>& out);
};

class SegmentwiseFormantExtractionNode : public SleeveNode, public SegmentwiseFormantExtraction {
public:
    enum FormantExtractionType {
        segmentwise
    };

private:
    static Core::Choice          domain_choice;
    static Core::ParameterChoice paramDomain;
    static Core::Choice          segment_estimator_choice;
    static Core::ParameterChoice paramSegmentEstimator;
    static Core::ParameterInt    paramOrder;
    static Core::ParameterInt    paramMaxNumberOfSegments;
    static Core::ParameterInt    paramStep;
    static Core::ParameterInt    paramMaxFrequency;

    bool sendFormant(const Flow::Vector<_float>& in, const std::vector<s32>& segments);
    bool sendLinearFilterParameter(const Flow::Vector<_float>& in, const std::vector<s32>& segments);
    bool sendLinearFilterSignal(const Flow::Vector<_float>& in);

public:
    static std::string filterName() {
        return "signal-formant-segmentwise";
    }
    SegmentwiseFormantExtractionNode(const Core::Configuration& c);
    virtual ~SegmentwiseFormantExtractionNode() {}

    virtual bool         setParameter(const std::string& name, const std::string& value);
    virtual bool         configure();
    virtual Flow::PortId getOutput(const std::string& name);

    virtual bool work(Flow::PortId p);

    void setMaxNumberOfSegments(u8 max_nr_segment);
};

}  // namespace Signal

#endif  // _SIGNAL_SEGMENTWISEFORMANTEXTRACTION_HH
