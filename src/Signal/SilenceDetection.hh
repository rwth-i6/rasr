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
#ifndef _SIGNAL_SILENCE_DETECTION_HH
#define _SIGNAL_SILENCE_DETECTION_HH

#include <Core/Parameter.hh>
#include <Flow/Vector.hh>
#include "Node.hh"
#include "SlidingWindow.hh"

/**\file
 * \todo SilenceDetection not tested yet.
 */

namespace Signal {

// SilenceDetection
///////////////////

/**
 * \warning SilenceDetection not tested yet
 */
class SilenceDetection {
public:
    typedef f32 Float;
    enum SilenceType {
        silence,
        speech,
        unsure
    };

protected:
    // Frame
    ////////

    class Frame : public Flow::DataPtr<Flow::Vector<Float>> {
    private:
        Float       energy_;
        SilenceType silence_;

        Float getEnergy(const Float energyScalingFactor) const;

    public:
        Frame()
                : energy_(0),
                  silence_(unsure) {}
        Frame(const Flow::DataPtr<Flow::Vector<Float>>& data, const Float energyScalingFactor)
                : Flow::DataPtr<Flow::Vector<Float>>(data),
                  energy_(0),
                  silence_(unsure) {
            energy_ = getEnergy(energyScalingFactor);
        }

        Float energy() const {
            return energy_;
        }
        SilenceType& silence() {
            return silence_;
        }
    };

private:
    std::vector<Float>   energyHistogram_;
    u32                  histogramSlidingWindowSize_;
    u32                  histogramSlidingWindowRight_;
    SlidingWindow<Frame> histogramSlidingWindow_;
    bool                 updateHistogram(const Frame& in, Frame& out);

    Float sparseEventRatio_;
    Float thresholdInterpolationFactor_;
    Float minSnr_;
    Float threshold_;
    void  updateThreshold();
    u32   fractile(f32 percent /*[0..1]*/) const;

    u32 bin(Float v) const {
        return (u32)(v < 0 ? 0 : (v > energyHistogram_.size() ? energyHistogram_.size() - 1 : v));
    }

    Float                accumulateBlockEnergy_;
    u32                  blockSize_;
    SlidingWindow<Frame> blockSlidingWindow_;
    bool                 updateBlock(const Frame& in, Frame& out);

    u32                  minSpeechLength_;
    u32                  minSilenceLength_;
    u32                  nUnsure_;
    SilenceType          lastDecision_;
    u32                  endDelay_;
    SlidingWindow<Frame> decisionSlidingWindow_;
    bool                 updateDecision(Frame& in, Frame& out);
    SilenceType          isSilence(const Float energy);

    u32                  beginDelay_;
    SlidingWindow<Frame> delaySlidingWindow_;
    bool                 updateDelay(Frame& in, Frame& out);

    bool need_init_;
    bool init();

    Float energyScalingFactor() {
        return (Float)energyHistogram_.size() / 90.0;
    }

public:
    SilenceDetection()
            : energyHistogram_(270),
              need_init_(true) {}
    ~SilenceDetection() {}

    void setHistogramBufferSize(const u32 size) {
        if (histogramSlidingWindowSize_ != size) {
            histogramSlidingWindowSize_ = size;
            need_init_                  = true;
        }
    }
    void setHistogramBufferDelay(const u32 delay) {
        if (histogramSlidingWindowRight_ != delay) {
            histogramSlidingWindowRight_ = delay;
            need_init_                   = true;
        }
    }

    void setBlockSize(const u32 size) {
        if (blockSize_ != size) {
            blockSize_ = size;
            need_init_ = true;
        }
    }

    void setSparseEventRatio(const Float sparseEventRatio) {
        if (sparseEventRatio_ != sparseEventRatio) {
            sparseEventRatio_ = sparseEventRatio;
            need_init_        = true;
        }
    }
    void setThresholdInterpolationFactor(const Float thresholdInterpolationFactor) {
        if (thresholdInterpolationFactor_ != thresholdInterpolationFactor) {
            thresholdInterpolationFactor_ = thresholdInterpolationFactor;
            need_init_                    = true;
        }
    }
    void setMinSnr(const Float minSnr) {
        if (minSnr_ != (minSnr * energyScalingFactor())) {
            minSnr_    = minSnr * energyScalingFactor();
            need_init_ = true;
        }
    }

    void setMinSpeechLength(const u32 minSpeechLength) {
        if (minSpeechLength_ != minSpeechLength) {
            minSpeechLength_ = minSpeechLength;
            need_init_       = true;
        }
    }
    void setMinSilenceLength(const u32 minSilenceLength) {
        if (minSilenceLength_ != minSilenceLength) {
            minSilenceLength_ = minSilenceLength;
            need_init_        = true;
        }
    }
    void setEndDelay(const u32 endDelay) {
        if (endDelay_ != endDelay) {
            endDelay_  = endDelay;
            need_init_ = true;
        }
    }

    void setBeginDelay(const u32 delay) {
        if (beginDelay_ != delay) {
            beginDelay_ = delay;
            need_init_  = true;
        }
    }

    bool update(const Flow::DataPtr<Flow::Vector<Float>>& in, Frame& out);
    bool flush(Frame& out);
    void reset() {
        need_init_ = true;
    }
};

// SilenceDetectionNode
///////////////////////

class SilenceDetectionNode : public SleeveNode, public SilenceDetection {
private:
    static Core::ParameterInt   paramHistogramBufferSize;
    static Core::ParameterInt   paramHistogramBufferDelay;
    static Core::ParameterInt   paramBlockSize;
    static Core::ParameterFloat paramSparseEventRatio;
    static Core::ParameterFloat paramThresholdInterpolationFactor;
    static Core::ParameterFloat paramMinSnr;
    static Core::ParameterInt   paramMinSpeechLength;
    static Core::ParameterInt   paramMinSilenceLength;
    static Core::ParameterInt   paramEndDelay;
    static Core::ParameterInt   paramBeginDelay;

    bool send(Frame& out) {
        if (!out)
            return false;
        if (out.silence() == unsure)
            criticalError("Unsure frame");

        Flow::Vector<Float>* s = new Flow::Vector<Float>;
        s->push_back((f32)(out.silence() == silence));
        putData(1, s);

        if (out.silence() == speech)
            return putData(0, out.get());
        return false;
    }

public:
    static std::string filterName() {
        return "signal-silence-detection";
    }

    SilenceDetectionNode(const Core::Configuration& c)
            : Core::Component(c),
              SleeveNode(c) {
        setHistogramBufferSize(paramHistogramBufferSize(c));
        setHistogramBufferDelay(paramHistogramBufferDelay(c));
        setBlockSize(paramBlockSize(c));
        setSparseEventRatio(paramSparseEventRatio(c));
        setThresholdInterpolationFactor(paramThresholdInterpolationFactor(c));
        setMinSnr(paramMinSnr(c));
        setMinSpeechLength(paramMinSpeechLength(c));
        setMinSilenceLength(paramMinSilenceLength(c));
        setEndDelay(paramEndDelay(c));
        setBeginDelay(paramBeginDelay(c));

        addOutput(1);
    }
    virtual ~SilenceDetectionNode() {}

    virtual bool setParameter(const std::string& name, const std::string& value) {
        if (paramHistogramBufferSize.match(name))
            setHistogramBufferSize(paramHistogramBufferSize(value));
        else if (paramHistogramBufferDelay.match(name))
            setHistogramBufferDelay(paramHistogramBufferDelay(value));
        else if (paramBlockSize.match(name))
            setBlockSize(paramBlockSize(value));
        else if (paramSparseEventRatio.match(name))
            setSparseEventRatio(paramSparseEventRatio(value));
        else if (paramThresholdInterpolationFactor.match(name))
            setThresholdInterpolationFactor(paramThresholdInterpolationFactor(value));
        else if (paramMinSnr.match(name))
            setMinSnr(paramMinSnr(value));
        else if (paramMinSpeechLength.match(name))
            setMinSpeechLength(paramMinSpeechLength(value));
        else if (paramMinSilenceLength.match(name))
            setMinSilenceLength(paramMinSilenceLength(value));
        else if (paramEndDelay.match(name))
            setEndDelay(paramEndDelay(value));
        else if (paramBeginDelay.match(name))
            setBeginDelay(paramBeginDelay(value));
        else
            return false;
        return true;
    }

    virtual bool configure() {
        Core::Ref<const Flow::Attributes> a0 = getInputAttributes(0);
        if (!configureDatatype(a0, Flow::Vector<f32>::type()))
            return false;

        return putOutputAttributes(0, a0) && putOutputAttributes(1, a0);
    }

    virtual Flow::PortId getOutput(const std::string& name) {
        if (name == "decision")
            return 1;
        return 0;
    }

    virtual bool work(Flow::PortId p) {
        Flow::DataPtr<Flow::Vector<f32>> in;
        Frame                            out;

        while (getData(0, in)) {
            if (!update(in, out))
                criticalError("Update failed");
            if (send(out))
                return true;
        }

        // in is invalid
        while (flush(out))
            send(out);

        reset();
        return putData(0, in.get()) && putData(1, in.get());
    }

    virtual void reset() {
        SilenceDetection::reset();
    }
};
}  // namespace Signal

#endif  // _SIGNAL_SILENCE_DETECTION_HH
