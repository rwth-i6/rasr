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
#include "SampleNormalization.hh"
#include <Core/Utility.hh>

using namespace Signal;
using namespace Core;
using namespace Flow;

// SampleNormalization
//////////////////////

SampleNormalization::SampleNormalization()
        : mean_(0),
          sumWeight_(0),
          sum_(0),
          changed_(true),
          minOutputSize_(0),
          outputStartTime_(0),
          sampleRate_(0),
          lengthInS_(0),
          rightInS_(0),
          needInit_(true) {}

bool SampleNormalization::setSampleRate(Time sampleRate) {
    if (sampleRate_ != sampleRate) {
        sampleRate_ = sampleRate;
        return (needInit_ = true);
    }
    return false;
}

bool SampleNormalization::setLengthInS(Time lengthInS) {
    if (lengthInS_ != lengthInS) {
        lengthInS_ = lengthInS;
        return (needInit_ = true);
    }
    return false;
}

bool SampleNormalization::setRightInS(Time rightInS) {
    if (rightInS_ != rightInS) {
        rightInS_ = rightInS;
        return (needInit_ = true);
    }
    return false;
}

void SampleNormalization::init() {
    require(sampleRate_ > 0);

    u32 length = lengthInS_ != Type<f64>::max ? (u32)Core::floor(lengthInS_ * sampleRate_ + .05) : (u32)Type<s32>::max;

    u32 right = rightInS_ != Type<f64>::max ? (u32)Core::floor(rightInS_ * sampleRate_ + .05) : (u32)Type<s32>::max;

    if (!slidingWindow_.init(length, right))
        defect();

    reset();

    needInit_ = false;
}

void SampleNormalization::reset() {
    mean_      = 0;
    sumWeight_ = 0;
    sum_       = 0;

    changed_ = true;
    out_.clear();
    outputStartTime_ = 0;
    slidingWindow_.clear();
}

bool SampleNormalization::update(const Sample* in) {
    if (in) {
        slidingWindow_.add(*in);

        Sample removed;
        updateStatistics(in, (slidingWindow_.removed(removed) ? &removed : 0));
    }
    else
        slidingWindow_.flushOut();

    Sample out;
    if (slidingWindow_.out(out)) {
        normalize(out);
        out_.push_back(out);
        return true;
    }
    return false;
}

void SampleNormalization::updateStatistics(const Sample* add, const Sample* remove) {
    if (add) {
        sum_ += *add;

        sumWeight_++;
        changed_ = true;
    }
    if (remove) {
        sum_ -= (*remove);

        sumWeight_--;
        changed_ = true;
    }
}

void SampleNormalization::normalizeStatistics() {
    if (!changed_)
        return;
    mean_ = f32(sum_ / sumWeight_);
}

void SampleNormalization::normalize(Sample& out) {
    normalizeStatistics();
    normalizeMean(out);
    changed_ = false;
}

bool SampleNormalization::put(const Vector<Sample>& in) {
    if (needInit_)
        init();

    size_t nPendingSamples = out_.size() + slidingWindow_.futureSize();
    Time   bufferEndTime   = outputStartTime_ + (Time)nPendingSamples / sampleRate_;
    if (!in.equalsToStartTime(bufferEndTime)) {
        if (nPendingSamples > 0)
            return false;
        outputStartTime_ = in.startTime();
    }
    for (u32 t = 0; t < in.size(); t++)
        update(&in[t]);
    return true;
}

bool SampleNormalization::get(Vector<Sample>& out) {
    if (needInit_)
        init();

    if (out_.size() >= minOutputSize_) {
        copyOutput(out);
        return true;
    }

    return false;
}

bool SampleNormalization::flush(Vector<Sample>& out) {
    if (needInit_)
        init();

    while (update(0)) {
        if (get(out))
            return true;
    }
    if (!out_.empty()) {
        copyOutput(out);
        return true;
    }
    reset();
    return false;
}

void SampleNormalization::copyOutput(Vector<Sample>& out) {
    verify(!out_.empty());

    out.clear();
    out_.swap(out);

    out.setStartTime(outputStartTime_);
    out.setEndTime(outputStartTime_ + (Time)out.size() / sampleRate_);
    outputStartTime_ = out.endTime();
}

// LengthDependentSampleNormalization
/////////////////////////////////////

LengthDependentSampleNormalization::LengthDependentSampleNormalization()
        : nShortInputSamples_(0),
          maxShortLength_(0),
          maxShortLengthInS_(0),
          needInit_(true) {
    short_.setLengthInS(Type<f64>::max);
    short_.setRightInS(Type<f64>::max);
}

void LengthDependentSampleNormalization::setMinOuptutSize(u32 size) {
    short_.setMinOuptutSize(size);
    long_.setMinOuptutSize(size);
}

void LengthDependentSampleNormalization::setLengthInS(Time lengthInS) {
    if (long_.setLengthInS(lengthInS))
        needInit_ = true;
}

void LengthDependentSampleNormalization::setRightInS(Time rightInS) {
    if (long_.setRightInS(rightInS))
        needInit_ = true;
}

void LengthDependentSampleNormalization::setMaxShortLengthInS(Time maxShortLengthInS) {
    if (maxShortLengthInS_ != maxShortLengthInS) {
        maxShortLengthInS_ = maxShortLengthInS;
        needInit_          = true;
    }
}

void LengthDependentSampleNormalization::setSampleRate(Time sampleRate) {
    short_.setSampleRate(sampleRate);
    if (long_.setSampleRate(sampleRate))
        needInit_ = true;
}

void LengthDependentSampleNormalization::init() {
    verify(sampleRate() > 0);
    maxShortLength_ = (u32)Core::floor(maxShortLengthInS_ * sampleRate() + .05);

    reset();

    needInit_ = false;
}

bool LengthDependentSampleNormalization::put(const Vector<Sample>& in) {
    if (needInit_)
        init();

    if (long_.put(in)) {
        if (nShortInputSamples_ < maxShortLength_) {
            if (!short_.put(in))
                defect();
            nShortInputSamples_ += in.size();
        }
        return true;
    }
    return false;
}

bool LengthDependentSampleNormalization::get(Vector<Sample>& out) {
    if (needInit_)
        init();

    if (nShortInputSamples_ >= maxShortLength_)
        return long_.get(out);

    return false;
}

bool LengthDependentSampleNormalization::flush(Vector<Sample>& out) {
    if (needInit_)
        init();

    bool result;

    if (nShortInputSamples_ >= maxShortLength_)
        result = long_.flush(out);
    else
        result = short_.flush(out);

    if (!result)
        reset();

    return result;
}

void LengthDependentSampleNormalization::reset() {
    long_.reset();
    short_.reset();

    nShortInputSamples_ = 0;
}

// SampleNormalizationNode
//////////////////////////

ParameterFloat SampleNormalizationNode::paramLengthInS(
        "length", "length of the sliding window in seconds");

ParameterFloat SampleNormalizationNode::paramRightInS(
        "right", "output point in seconds");

ParameterInt SampleNormalizationNode::paramMinOuptutSize(
        "block-size", "size of output blocks in samples", 4096, 0);

ParameterFloat SampleNormalizationNode::paramMaxShortLengthInS(
        "short-sentence-length", "max length of short sentence in seconds, normalized sentencewise", 0, 0);

SampleNormalizationNode::SampleNormalizationNode(const Core::Configuration& c)
        : Component(c),
          Predecessor(c) {
    setLengthInS(paramLengthInS(c));
    setRightInS(paramRightInS(c));
    setMinOuptutSize(paramMinOuptutSize(c));
    setMaxShortLengthInS(paramMaxShortLengthInS(c));
}

bool SampleNormalizationNode::setParameter(const std::string& name, const std::string& value) {
    if (paramLengthInS.match(name))
        setLengthInS(paramLengthInS(value));
    else if (paramRightInS.match(name))
        setRightInS(paramRightInS(value));
    else if (paramMinOuptutSize.match(name))
        setMinOuptutSize(paramMinOuptutSize(value));
    else if (paramMaxShortLengthInS.match(name))
        setMaxShortLengthInS(paramMaxShortLengthInS(value));
    else
        return false;

    return true;
}

bool SampleNormalizationNode::configure() {
    Core::Ref<const Attributes> a = getInputAttributes(0);
    if (!configureDatatype(a, Vector<f32>::type()))
        return false;

    setSampleRate(atof(a->get("sample-rate").c_str()));
    reset();

    return putOutputAttributes(0, a);
}
