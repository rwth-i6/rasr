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
#ifndef _SIGNAL_SAMPLE_NORMALIZATION_HH
#define _SIGNAL_SAMPLE_NORMALIZATION_HH

#include <Core/Parameter.hh>
#include <Flow/Vector.hh>
#include "SlidingWindow.hh"
#include "SlidingAlgorithmNode.hh"

namespace Signal {

    // SampleNormalization
    //////////////////////

    class SampleNormalization {
    public:
        typedef f32 Sample;
        typedef Flow::Time Time;
    private:
        SlidingWindow<Sample> slidingWindow_;
        f32 mean_;
        f64 sumWeight_;
        f64 sum_;
        bool changed_;

        std::vector<Sample> out_;
        size_t minOutputSize_;
        Time outputStartTime_;

        Time sampleRate_;
        Time lengthInS_;
        Time rightInS_;

        bool needInit_;
    private:
        void init();

        bool update(const Sample *in);
        void updateStatistics(const Sample *add, const Sample *remove);
        void normalizeStatistics();

        void normalize(Sample &out);
        void normalizeMean(Sample &out) const { out -= mean_; }

        void copyOutput(Flow::Vector<Sample> &out);
    public:
        SampleNormalization();

        void setMinOuptutSize(size_t size) { minOutputSize_ = size; }

        bool setSampleRate(Time sampleRate);
        Time sampleRate() const { return sampleRate_; }

        bool setLengthInS(Time lengthInS);
        bool setRightInS(Time rightInS);

        /** @return is false if there is a time gap between two subsequent inputs
         */
        bool put(const Flow::Vector<Sample> &in);
        /** @return is false if there is less processed samples than minSize_.
         */
        bool get(Flow::Vector<Sample> &out);
        /** @return true if there has been data which have not been output yet.
         *  To retrieve all the processed input samples call flush until it returns false.
         */
        bool flush(Flow::Vector<Sample> &out);
        void reset();
    };

    class LengthDependentSampleNormalization {
    public:
        typedef SampleNormalization::Sample Sample;
        typedef SampleNormalization::Time Time;

        typedef Flow::Vector<Sample> InputData;
        typedef Flow::Vector<Sample> OutputData;
    private:
        SampleNormalization short_;
        SampleNormalization long_;

        u32 nShortInputSamples_;

        u32 maxShortLength_;
        Time maxShortLengthInS_;

        bool needInit_;
    private:
        void init();
    public:
        LengthDependentSampleNormalization();

        void setMinOuptutSize(u32 size);
        void setLengthInS(Time lengthInS);
        void setRightInS(Time rightInS);
        void setMaxShortLengthInS(Time maxShortLengthInS);

        void setSampleRate(Time sampleRate);
        Time sampleRate() const { return long_.sampleRate(); }

        bool put(const Flow::Vector<Sample> &in);
        bool get(Flow::Vector<Sample> &out);
        bool flush(Flow::Vector<Sample> &out);
        void reset();
    };


    /** SampleNormalizationNode
     */
    class SampleNormalizationNode : public SlidingAlgorithmNode<LengthDependentSampleNormalization> {
        typedef SlidingAlgorithmNode<LengthDependentSampleNormalization> Predecessor;
    private:
        static Core::ParameterFloat paramLengthInS;
        static Core::ParameterFloat paramRightInS;
        static Core::ParameterInt paramMinOuptutSize;
        static Core::ParameterFloat paramMaxShortLengthInS;
    public:
        static std::string filterName() { return "signal-sample-normalization"; }

        SampleNormalizationNode(const Core::Configuration &c);
        virtual ~SampleNormalizationNode() {}

        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool configure();
    };

} // namespace Signal


#endif // _SIGNAL_SAMPLE_NORMALIZATION_HH
