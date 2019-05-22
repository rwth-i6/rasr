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
#include "SilenceDetection.hh"

using namespace Core;
using namespace Flow;
using namespace Signal;

// SilenceDetection::Frame
///////////////////////////////

SilenceDetection::Float SilenceDetection::Frame::getEnergy(const Float energyScalingFactor) const {
    if (!operator bool())
        return 0;

    SilenceDetection::Float energy = 0;
    for (u32 i = 0; i < get()->size(); i++)
        energy += (*get())[i] * (*get())[i];

    energy /= get()->size();

    if (energy == 0.0)
        return -10000.0;

    return 10 * log10(energy) * energyScalingFactor;
}

// SilenceDetection
///////////////////

bool SilenceDetection::updateHistogram(const Frame& in, Frame& out) {
    if (in) {
        histogramSlidingWindow_.add(in);
        energyHistogram_[bin(in.energy())]++;
    }
    else
        histogramSlidingWindow_.flushOut();

    Frame removed;
    if (histogramSlidingWindow_.removed(removed))
        energyHistogram_[bin(removed.energy())]--;

    return histogramSlidingWindow_.out(out);
}

void SilenceDetection::updateThreshold() {
    // if histogram is not representative, move
    f32 sparseEventRatio = sparseEventRatio_ *
                           (f32)histogramSlidingWindow_.size() / (f32)histogramSlidingWindow_.maxSize();

    Float noiseFractile  = fractile(sparseEventRatio);
    Float speechFractile = fractile(1.0 - sparseEventRatio);

    if (speechFractile - noiseFractile > minSnr_) {
        // histogram has two clear maximum
        threshold_ = (1.0 - thresholdInterpolationFactor_) * noiseFractile +
                     thresholdInterpolationFactor_ * speechFractile;
        // for continuoity
        threshold_ = std::max(threshold_, noiseFractile + minSnr_);
    }
    else {
        // histogram has one maximum, probably just noise
        threshold_ = noiseFractile + minSnr_;
    }
}

u32 SilenceDetection::fractile(f32 percent /*[0..1]*/) const {
    percent *= histogramSlidingWindow_.size();
    float p = 0;
    u32   e;
    for (e = 0; e < energyHistogram_.size(); e++) {
        p += energyHistogram_[e];
        if (p > percent)
            break;
    }
    return e;
}

bool SilenceDetection::updateBlock(const Frame& in, Frame& out) {
    if (in) {
        blockSlidingWindow_.add(in);
        accumulateBlockEnergy_ += in.energy();
    }
    else
        blockSlidingWindow_.flushOut();

    Frame removed;
    if (blockSlidingWindow_.removed(removed))
        accumulateBlockEnergy_ -= removed.energy();

    return blockSlidingWindow_.out(out);
}

SilenceDetection::SilenceType SilenceDetection::isSilence(const SilenceDetection::Float energy) {
    if (lastDecision_ == silence) {
        if (energy < threshold_)
            return silence;
        else if (nUnsure_ + 1 < minSpeechLength_) {
            nUnsure_++;
            return unsure;
        }
        else
            return speech;
    }
    else {
        if (energy >= threshold_)
            return speech;
        else if (nUnsure_ + 1 < minSilenceLength_) {
            nUnsure_++;
            return unsure;
        }
        else
            return silence;
    }
}

bool SilenceDetection::updateDecision(Frame& in, Frame& out) {
    SilenceType currentDecision = (!in ? lastDecision_ : isSilence(accumulateBlockEnergy_ / blockSlidingWindow_.size()));
    if (currentDecision != unsure) {
        u32 i = 1;
        // set begining of a silence interval to speech
        if (lastDecision_ == speech && currentDecision == silence)
            for (; i <= endDelay_; i++)
                decisionSlidingWindow_[nUnsure_ - i].silence() = speech;

        //set unsures to the last decision
        for (; i <= nUnsure_; i++)
            decisionSlidingWindow_[nUnsure_ - i].silence() = currentDecision;

        lastDecision_ = currentDecision;
        nUnsure_      = 0;
    }

    in.silence() = currentDecision;
    if (in)
        decisionSlidingWindow_.add(in);
    else
        decisionSlidingWindow_.flushOut();
    return decisionSlidingWindow_.out(out);
}

bool SilenceDetection::updateDelay(Frame& in, Frame& out) {
    // set end of a silence interval to speech
    if (in.silence() == speech && delaySlidingWindow_[0].silence() == silence) {
        for (u32 i = 0; i < beginDelay_; i++)
            delaySlidingWindow_[i].silence() = speech;
    }

    if (in)
        delaySlidingWindow_.add(in);
    else
        delaySlidingWindow_.flushOut();
    return delaySlidingWindow_.out(out);
}

bool SilenceDetection::init() {
    for (u32 i = 0; i < energyHistogram_.size(); i++)
        energyHistogram_[i] = 0;
    if (!histogramSlidingWindow_.init(histogramSlidingWindowSize_, histogramSlidingWindowRight_))
        return false;

    threshold_ = 0;

    accumulateBlockEnergy_ = 0;
    if (blockSize_ < 1 || !blockSlidingWindow_.init(blockSize_, blockSize_ / 2))
        return false;

    lastDecision_ = silence;
    nUnsure_      = 0;
    if (std::max(minSpeechLength_, minSilenceLength_) < 1 ||
        !decisionSlidingWindow_.init(std::max(minSpeechLength_, minSilenceLength_),
                                     std::max(minSpeechLength_, minSilenceLength_) - 1))
        return false;

    if (!delaySlidingWindow_.init(std::max(beginDelay_, (u32)1), std::max(beginDelay_, (u32)1) - 1))
        return false;

    if (beginDelay_ + endDelay_ > minSilenceLength_)
        return false;

    return !(need_init_ = false);
}

bool SilenceDetection::update(const Flow::DataPtr<Flow::Vector<Float>>& in, Frame& out) {
    if (need_init_ && !init())
        return false;

    Frame add(in, energyScalingFactor()), outHistogram, outBlock, outDecision;
    out = Frame();

    updateHistogram(add, outHistogram);
    updateThreshold();
    updateBlock(outHistogram, outBlock);
    updateDecision(outBlock, outDecision);
    updateDelay(outDecision, out);
    return true;
}

bool SilenceDetection::flush(Frame& out) {
    if (delaySlidingWindow_.futureSize() == 0)
        return false;

    Frame add, outHistogram, outBlock, outDecision;
    out = Frame();

    updateHistogram(add, outHistogram);
    //threshold not updated
    updateBlock(outHistogram, outBlock);
    updateDecision(outBlock, outDecision);
    updateDelay(outDecision, out);
    return true;
}

// SilenceDetectionNode
///////////////////////

ParameterInt   SilenceDetectionNode::paramHistogramBufferSize("histogram-buffer-size", "size of the histogram ringbuffer in frames", 600, 101);
ParameterInt   SilenceDetectionNode::paramHistogramBufferDelay("histogram-buffer-delay", "delay of the histogram ringbuffer in frames", 100, 100);
ParameterInt   SilenceDetectionNode::paramBlockSize("block-size", "number of averaged frames for energy calculation", 5, 1);
ParameterFloat SilenceDetectionNode::paramSparseEventRatio("sparse-event-ration", "fractile value", 0.1, 0.0, 1.0);
ParameterFloat SilenceDetectionNode::paramThresholdInterpolationFactor("threshold-interpolation-factor", "threshold interpolation factor", 0.3, 0.0, 1.0);
ParameterFloat SilenceDetectionNode::paramMinSnr("min-snr", "threshold interpolation limit in dB", 13);
ParameterInt   SilenceDetectionNode::paramMinSpeechLength("min-speech-length", "min number of speech frames to decide for speech", 6, 1);
ParameterInt   SilenceDetectionNode::paramMinSilenceLength("min-silence-length", "min number of silence frames to decide for silence", 16, 1);
ParameterInt   SilenceDetectionNode::paramEndDelay("end-delay", "number of silence frames after speech set to speech", 12, 0);
ParameterInt   SilenceDetectionNode::paramBeginDelay("begin-delay", "number of silence frames before speech set to speech", 4, 0);
