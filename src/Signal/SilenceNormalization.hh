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
#ifndef _SIGNAL_SILENCE_NORMALIZATION_HH
#define _SIGNAL_SILENCE_NORMALIZATION_HH

#include <Core/Parameter.hh>
#include <Flow/Vector.hh>
#include "SlidingAlgorithmNode.hh"

namespace Signal {
/**
 * This node normalizes the fraction of silence in the outgoing signal.
 * The returned speech segments contain exactly the requested
 * amount of silence.
 *
 * An EM-like algorithm is used to segment the silence from the non-silence,
 * which expects that there is both silence and speech in the segment. Therefore
 * data put into this node should already have been somehow segmented externally.
 *
 * The constraints are: At least 1% of the signal must be silence, and at least 1%
 * must be speech (see the parameter initialization-fraction).
 * */

class SilenceNormalization {
public:
    typedef f32        Sample;
    typedef Flow::Time Time;

    typedef Flow::Vector<Sample> InputData;
    typedef Flow::Vector<Sample> OutputData;

private:
    // Config:
    Time sampleRate_;
    Time minSurroundingSilence_;
    f64  silenceFraction_;
    f64  initializationFraction_;
    f64  blockSize_;
    u32  blockSizeSamples_;
    u32  iterations_;
    bool discardUnsure_;
    f32  silenceThreshold_, absoluteSilenceThreshold_, addNoise_;
    bool fillUpSilence_;
    bool preserveTiming_;

    std::list<Flow::Vector<Sample>> flushQueue_;

    // Temporary values:
    bool                                needInit_;
    std::deque<std::pair<Time, Sample>> buffer_;

private:
    void init();

    Core::Component::Message log() const;

public:
    SilenceNormalization();

    virtual ~SilenceNormalization() {}

    void setFillUpSilence(bool fill);

    void setPreserveTiming(bool preserve);

    void setInitializationFraction(f64 fraction);

    void setBlockSize(f64 size);

    void setMinSurroundingSilence(Time duration);

    void setSampleRate(Time sampleRate);

    void setSilenceFraction(f64 fraction);

    void setDiscardUnsure(bool discard);

    void setIterations(u32 iterations);

    void setSilenceThreshold(f32 threshold);

    void setAddNoise(f32 noise);

    void setAbsoluteSilenceThreshold(f32 threshold);

    /** @return is false if the is a time gap between start time of in
     *  and end time of the buffer.
     */
    bool put(const Flow::Vector<Sample>& in);

    /** delivers a block of filtered samples
     *  @return is false if decision could not be made yet
     */
    bool get(Flow::Vector<Sample>& out);

    /** delivers everything remaining
     *  @return is false if buffer is empty
     */
    bool flush(Flow::Vector<Sample>& out);

    bool flushFromQueue(Flow::Vector<Sample>& out);

    /** If selectedBlocks is empty, then all blocks are used */
    void startFlushingFromQueue(const std::vector<bool>& selectedBlocks);

    void reset();
};

// SilenceNormalizationNode
///////////////////////

class SilenceNormalizationNode : public SlidingAlgorithmNode<SilenceNormalization> {
public:
    typedef SlidingAlgorithmNode<SilenceNormalization> Predecessor;

private:
    static Core::ParameterFloat paramSilenceFraction;
    static Core::ParameterFloat paramBlockSize;
    static Core::ParameterBool  paramFillUpSilence;
    static Core::ParameterBool  paramPreserveTiming;
    static Core::ParameterFloat paramMinSurroundingSilence;
    static Core::ParameterFloat paramInitializationFraction;
    static Core::ParameterBool  paramDiscardUnsureSegments;
    static Core::ParameterInt   paramEMIterations;
    static Core::ParameterFloat paramSilenceThreshold;
    static Core::ParameterFloat paramAbsoluteSilenceThreshold;
    static Core::ParameterFloat paramAddNoise;

public:
    static std::string filterName() {
        return "signal-silence-normalization";
    }

    SilenceNormalizationNode(const Core::Configuration& c);
    virtual ~SilenceNormalizationNode() {}

    virtual bool setParameter(const std::string& name, const std::string& value);

    virtual bool configure();
};
}  // namespace Signal

#endif  // _SIGNAL_DC_DETECTION_HH
