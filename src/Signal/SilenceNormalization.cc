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
#include "SilenceNormalization.hh"
#include <Core/Extensions.hh>
#include <Core/Utility.hh>
#include <stdio.h>
#include <stdlib.h>

using namespace Core;
using namespace Flow;
using namespace Signal;

class GaussianDensity {
public:
    GaussianDensity(f32 bias = 1.0)
            : mean_(0),
              variance_(0),
              sigma_(0),
              offset_(0),
              energySum_(0),
              energySquareSum_(0),
              energyCount_(0),
              bias_(bias) {
    }

    // Returns the score regarding the gaussian distribution
    f64 score(f64 value) const {
        f64 d = (value - mean_) / sigma_;
        d     = (d * d) * 0.5;
        return (d + offset_) / bias_;
    }

    void add(f64 energy) {
        energySum_ += energy;
        energyCount_ += 1;
        energySquareSum_ += energy * energy;
    }

    void estimate(bool estimateVariance = false) {
        if (energyCount_) {
            mean_ = energySum_ / energyCount_;
            if (!estimateVariance) {
                variance_ = (energySquareSum_ - 2 * mean_ * energySum_ + energyCount_ * mean_ * mean_) / energyCount_;
                if (variance_ < 1)
                    variance_ = 1;
                sigma_ = sqrt(variance_);
            }
            offset_ = sigma_ * sqrt(2 * M_PI);
        }
        energySum_       = 0;
        energySquareSum_ = 0;
        energyCount_     = 0;
    }

    f64 mean() const {
        return mean_;
    }

    f64 sigma() const {
        return sigma_;
    }

private:
    f64 mean_;
    f64 variance_;
    f64 sigma_;
    f64 offset_;

    f64 energySum_;
    f64 energySquareSum_;
    u32 energyCount_;

    f32 bias_;
};

// SilenceNormalization
//////////////

SilenceNormalization::SilenceNormalization()
        : sampleRate_(0),
          minSurroundingSilence_(.02),
          silenceFraction_(0),
          initializationFraction_(0),
          blockSize_(0),
          blockSizeSamples_(0),
          iterations_(0),
          discardUnsure_(false),
          silenceThreshold_(0.1),
          absoluteSilenceThreshold_(0.0),
          addNoise_(0),
          fillUpSilence_(false),
          preserveTiming_(false),
          needInit_(true) {
}

void SilenceNormalization::setSilenceThreshold(f32 threshold) {
    silenceThreshold_ = threshold;
}

void SilenceNormalization::setAbsoluteSilenceThreshold(f32 threshold) {
    absoluteSilenceThreshold_ = threshold;
}

void SilenceNormalization::setAddNoise(f32 noise) {
    log() << "setting add-noise " << noise;
    addNoise_ = noise;
}

void SilenceNormalization::setIterations(u32 iterations) {
    iterations_ = iterations;
}

void SilenceNormalization::setDiscardUnsure(bool discard) {
    discardUnsure_ = discard;
}

void SilenceNormalization::setFillUpSilence(bool fill) {
    fillUpSilence_ = fill;
}

void SilenceNormalization::setPreserveTiming(bool preserve) {
    log() << "preserving timing: " << preserve;
    preserveTiming_ = preserve;
}

void SilenceNormalization::setInitializationFraction(f64 fraction) {
    initializationFraction_ = fraction;
}

void SilenceNormalization::setBlockSize(f64 size) {
    blockSize_ = size;
}

void SilenceNormalization::setMinSurroundingSilence(Time duration) {
    if (minSurroundingSilence_ != duration) {
        minSurroundingSilence_ = duration;
        needInit_              = true;
    }
}

void SilenceNormalization::setSampleRate(Time sampleRate) {
    if (sampleRate_ != sampleRate) {
        sampleRate_ = sampleRate;
        needInit_   = true;
    }
}

void SilenceNormalization::setSilenceFraction(f64 fraction) {
    silenceFraction_ = fraction;
}

void SilenceNormalization::init() {
    verify(sampleRate_ > 0);

    blockSizeSamples_ = std::max((u32)rint(blockSize_ * sampleRate_), 1u);

    reset();

    needInit_ = false;
}

void SilenceNormalization::reset() {
    buffer_.clear();
    flushQueue_.clear();
}

bool SilenceNormalization::put(const Vector<Sample>& in) {
    if (needInit_)
        init();

    Time startTime = in.startTime();

    for (std::vector<Sample>::const_iterator it = in.begin(); it != in.end(); ++it)
        buffer_.push_back(std::make_pair((Time)(it - in.begin()) / sampleRate_ + startTime, *it));

    return true;
}

bool SilenceNormalization::get(Vector<Sample>& out) {
    if (needInit_)
        init();

    return false;
}

Core::Component::Message SilenceNormalization::log() const {
    return dynamic_cast<const Component*>(this)->log();
}

bool SilenceNormalization::flushFromQueue(Vector<SilenceNormalization::Sample>& out) {
    verify(out.empty());
    if (flushQueue_.empty())
        return false;
    out = flushQueue_.front();
    verify(out.size());
    flushQueue_.pop_front();
    if (addNoise_) {
        for (std::vector<Sample>::iterator it = out.begin(); it != out.end(); ++it)
            *it += ((((f64)rand()) / ((f64)RAND_MAX)) - 0.5) * 2 * addNoise_;
    }
    return true;
}

void SilenceNormalization::startFlushingFromQueue(const std::vector<bool>& isSpeech) {
    if (buffer_.empty())
        return;
    Time           tolerance    = 0.5 / sampleRate_;
    Time           samplelength = 1 / sampleRate_;
    Vector<Sample> o;
    u32            ofs = 0;
    for (u32 i = 0; i < isSpeech.size(); ++i, ++ofs) {
        if (isSpeech[i])
            break;
    }
    std::cout << "offset: " << ofs << std::endl;
    for (u32 sample = 0; sample < buffer_.size(); ++sample) {
        if (!isSpeech.empty()) {
            u32 block = sample / blockSizeSamples_;
            verify(block < isSpeech.size());
            if (!isSpeech[block])
                continue;
        }
        if (!o.empty() && buffer_[sample].first > o.endTime() + tolerance && preserveTiming_) {
            flushQueue_.push_back(o);
            o = Vector<Sample>();
        }
        if (o.empty())
            o.setStartTime(buffer_[sample].first);
        o.push_back(buffer_[sample].second);
        o.setEndTime(buffer_[sample].first + samplelength);
    }

    if (!preserveTiming_) {
        verify(flushQueue_.empty());
        o.setStartTime(buffer_.front().first);
    }

    o.setEndTime(o.startTime() + o.size() * samplelength);

    if (!o.empty())
        flushQueue_.push_back(o);

    if (!flushQueue_.empty())
        std::cout << "time range of first flushed item: " << flushQueue_.front().startTime() << " " << flushQueue_.front().endTime() << "(real start: " << buffer_.front().first << ")" << std::endl;

    buffer_.clear();
}

bool SilenceNormalization::flush(Vector<Sample>& out) {
    if (needInit_)
        init();

    out.clear();

    if (flushFromQueue(out))
        return true;

    if (buffer_.empty())
        return false;

    if (silenceFraction_ == 1.0 || buffer_.size() < blockSizeSamples_ * 6) {
        if (silenceFraction_ < 1.0)
            log() << "buffer too short:" << buffer_.size() << " min. " << blockSizeSamples_ * 6;
        startFlushingFromQueue(std::vector<bool>());
        return flushFromQueue(out);
    }

    std::vector<f64> blocks;

    for (u32 b = 0; b < 1 + buffer_.size() / blockSizeSamples_; ++b) {
        u32 start = b * blockSizeSamples_;
        if (start >= buffer_.size())
            break;
        u32 end = (b + 1) * blockSizeSamples_;
        if (end > buffer_.size())
            end = buffer_.size();
        f64 energySum = 0;
        for (u32 s = start; s < end; ++s)
            energySum += Core::abs(buffer_[s].second);
        blocks.push_back(energySum / (end - start));
    }
    verify(blocks.size());

    std::vector<std::pair<f64, u32>> sortedBlocks;
    for (u32 b = 0; b < blocks.size(); ++b)
        sortedBlocks.push_back(std::make_pair(blocks[b], b));
    std::sort(sortedBlocks.begin(),
              sortedBlocks.end(),
              Core::composeBinaryFunction(std::less<f64>(), Core::select1st<std::pair<f64, u32>>(), Core::select1st<std::pair<f64, u32>>()));

    bool classificationFailed = false;
    u32  initOffset           = std::max(sortedBlocks.size() * initializationFraction_, 3.0);

    GaussianDensity speech;
    GaussianDensity silence;

    // True for speech, false for silence
    std::vector<bool> isSpeech(blocks.size(), true);

    for (u32 it = 0; it < iterations_; ++it) {
        if (it > 0) {
            // Assign
            for (u32 block = 0; block < blocks.size(); ++block) {
                isSpeech[block] = (blocks[block] - silence.mean() >= (speech.mean() - silence.mean()) * silenceThreshold_) && blocks[block] > absoluteSilenceThreshold_;
            }
        }

        // Always re-assign the boundary sets, to make sure that we never flip speech and silence
        for (u32 i = 0; i < initOffset; ++i) {
            silence.add(sortedBlocks[i].first);
            speech.add(sortedBlocks[sortedBlocks.size() - 1 - i].first);
        }

        speech.estimate();
        if (it == 0)
            silence.estimate();
    }

    {
        u32 speechCount = 0, silenceCount = 0;
        for (u32 block = 0; block < blocks.size(); ++block)
            if (isSpeech[block])
                speechCount += 1;
            else
                silenceCount += 1;

        if (speech.mean() <= silence.mean() || speechCount == 0 || silenceCount == 0) {
            classificationFailed = true;
            log() << "segment failed due to misclassification. Total speech: " << speechCount << " total silence: " << silenceCount;
            isSpeech.assign(isSpeech.size(), !discardUnsure_);  // Assign everything to speech, so that the speech-recognizer can decide
            if (discardUnsure_)
                log() << "discarded all";
            else
                log() << "accepted all";
        }
    }

    log() << "silence mean " << silence.mean() << " deviation " << silence.sigma();
    log() << "speech mean " << speech.mean() << " deviation " << speech.sigma();

    // Apply consistency constraints on the blocks
    u32 minSurroundingSilenceBlocks = std::max(minSurroundingSilence_ / blockSize_, 1.0);

    u32 speechCount = 0, silenceInSpeech = 0, silenceCount = 0;
    for (u32 block = 0; block < blocks.size(); ++block)
        if (isSpeech[block])
            speechCount += 1;
        else
            silenceCount += 1;

    f32 oldRatio = silenceCount / (f32)(silenceCount + speechCount);
    f32 ratio    = silenceInSpeech / (f32)speechCount;

    u32 surrounding = 0;

    // Extend speech until the targeted ratio is achieved
    while (surrounding < minSurroundingSilenceBlocks || ratio < silenceFraction_) {
        ++surrounding;
        std::vector<bool> isSpeechOld = isSpeech;
        u32               oldCount    = speechCount;
        for (u32 block = 0; block < blocks.size(); ++block) {
            if (!isSpeechOld[block] && !isSpeech[block]) {
                if ((block > 0 && isSpeechOld[block - 1]) ||
                    (block + 1 < blocks.size() && isSpeechOld[block + 1])) {
                    isSpeech[block] = true;
                    silenceInSpeech += 1;
                    speechCount += 1;
                    ratio = silenceInSpeech / (f32)speechCount;
                    if (surrounding >= minSurroundingSilenceBlocks && ratio >= silenceFraction_)
                        break;
                }
            }
        }

        if (oldCount == speechCount) {
            log() << "not enough silence available (speech count " << speechCount << ")";
            break;  // Nothing can be done any more
        }
    }

    u32 oldBufferSize = buffer_.size();

    startFlushingFromQueue(isSpeech);

    if (fillUpSilence_ && !classificationFailed && flushQueue_.size()) {
        Flow::Vector<Sample> silenceBlock;
        while (ratio < silenceFraction_) {
            u32 block       = sortedBlocks[silenceBlock.size() % initOffset].second;
            u32 firstSample = block * blockSizeSamples_;
            u32 endSample   = std::min((block + 1) * blockSizeSamples_, (u32)buffer_.size());
            for (u32 sample = firstSample; sample < endSample; ++sample) {
                silenceBlock.push_back(buffer_[sample].second);
            }
            silenceInSpeech += 1;
            speechCount += 1;
            ratio = silenceInSpeech / (f32)speechCount;
        }
        if (silenceBlock.size()) {
            silenceBlock.setStartTime(flushQueue_.back().endTime());
            silenceBlock.setEndTime(silenceBlock.startTime() + silenceBlock.size() / sampleRate_);
            flushQueue_.push_back(silenceBlock);
            log() << "added " << silenceBlock.size() << " additional silence samples to reach silence fraction";
        }
    }
    u32 outputSize = 0;
    for (std::list<Vector<Sample>>::iterator it = flushQueue_.begin(); it != flushQueue_.end(); ++it) {
        outputSize += it->size();
        log() << "speech " << it->startTime() << " " << it->endTime() << " " << it->size();
    }
    s32 difference = (oldBufferSize - outputSize);
    log() << "accepted silence/speech samples: " << outputSize << " difference: " << difference << " difference fraction: " << difference / (f32)oldBufferSize << " new silence ratio: " << ratio << " old: " << oldRatio;
    return flushFromQueue(out);
}

// SilenceNormalizationNode
//////////////////

ParameterFloat SilenceNormalizationNode::paramSilenceFraction(
        "silence-ratio", "target fraction of silence. If 1.0, this flow node does nothing. Recommendation: 0.3", 1.0, 0, 1.0);

ParameterBool SilenceNormalizationNode::paramFillUpSilence(
        "fill-up-silence", "whether artificial silence frames should be added to match the targeted silence fraction", false);

ParameterBool SilenceNormalizationNode::paramPreserveTiming(
        "preserve-timing", "whether incoming timeframe information should be preserved when doing silence normalization (may confuse the feature extraction)", true);

ParameterFloat SilenceNormalizationNode::paramMinSurroundingSilence(
        "min-surrounding-silence", "minimum length of added silence surrounding speech (in seconds)", .05, 0);

ParameterFloat SilenceNormalizationNode::paramSilenceThreshold(
        "silence-threshold", "relative threshold defining the variance of the silence model (lower means less silence is detected, higher means more silence is detected)", 0.1, 0.0, 1.0);

ParameterFloat SilenceNormalizationNode::paramAbsoluteSilenceThreshold(
        "absolute-silence-threshold", "absolute magnitude threshold below which everything is considered silence", 0.0, 0.0, Core::Type<f32>::max);

ParameterFloat SilenceNormalizationNode::paramAddNoise(
        "add-noise", "magnitude of random noise added to the signal", 0.0, 0.0, Core::Type<f32>::max);

ParameterFloat SilenceNormalizationNode::paramBlockSize(
        "block-size", "size of blocks (in seconds) which are averaged together", .01, 0);

ParameterFloat SilenceNormalizationNode::paramInitializationFraction(
        "initialization-fraction", "minimum fraction of the signal which is expected to be available for both silence and speech", 0.01, 0.001);

ParameterBool SilenceNormalizationNode::paramDiscardUnsureSegments(
        "discard-unsure-segments", "whether segments where classification of silence fails should be discarded", true);

ParameterInt SilenceNormalizationNode::paramEMIterations(
        "em-iterations", "number of expectation maximization iterations", 20);

SilenceNormalizationNode::SilenceNormalizationNode(const Core::Configuration& c)
        : Component(c),
          Predecessor(c) {
    setSilenceFraction(paramSilenceFraction(c));
    setMinSurroundingSilence(paramMinSurroundingSilence(c));
    setInitializationFraction(paramInitializationFraction(c));
    setDiscardUnsure(paramDiscardUnsureSegments(c));
    setBlockSize(paramBlockSize(c));
    setIterations(paramEMIterations(c));
    setSilenceThreshold(paramSilenceThreshold(c));
    setAbsoluteSilenceThreshold(paramAbsoluteSilenceThreshold(c));
    setPreserveTiming(paramPreserveTiming(c));
    setAddNoise(paramAddNoise(c));
}

bool SilenceNormalizationNode::setParameter(const std::string& name, const std::string& value) {
    if (paramSilenceFraction.match(name))
        setSilenceFraction(paramSilenceFraction(value));
    else if (paramFillUpSilence.match(name))
        setFillUpSilence(paramFillUpSilence(value));
    else if (paramPreserveTiming.match(name))
        setPreserveTiming(paramPreserveTiming(value));
    else if (paramMinSurroundingSilence.match(name))
        setMinSurroundingSilence(paramMinSurroundingSilence(value));
    else if (paramInitializationFraction.match(name))
        setInitializationFraction(paramInitializationFraction(value));
    else if (paramBlockSize.match(name))
        setBlockSize(paramBlockSize(value));
    else if (paramDiscardUnsureSegments.match(name))
        setDiscardUnsure(paramDiscardUnsureSegments(value));
    else if (paramEMIterations.match(name))
        setIterations(paramEMIterations(value));
    else if (paramSilenceThreshold.match(name))
        setSilenceThreshold(paramSilenceThreshold(value));
    else if (paramAbsoluteSilenceThreshold.match(name))
        setAbsoluteSilenceThreshold(paramAbsoluteSilenceThreshold(value));
    else if (paramAddNoise.match(name))
        setAddNoise(paramAddNoise(value));
    else
        return false;

    return true;
}

bool SilenceNormalizationNode::configure() {
    Core::Ref<const Flow::Attributes> a = getInputAttributes(0);
    if (!configureDatatype(a, Flow::Vector<f32>::type()))
        return false;

    setSampleRate(atof(a->get("sample-rate").c_str()));
    reset();

    return putOutputAttributes(0, a);
}
