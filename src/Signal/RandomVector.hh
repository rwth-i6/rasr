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
#ifndef _SIGNAL_RANDOM_VECTOR_HH
#define _SIGNAL_RANDOM_VECTOR_HH

#include <Flow/Vector.hh>
#include <Math/Random.hh>
#include "Node.hh"

namespace Signal {

    /** Generates random vectors
     *  Parameters:
     *    -type: type of random vector (@see Math::RandomVectorGenerator).
     *    -size: number of components to generate
     *    -start-time: start-time of the given segment thus the start-time of the first output vector
     *    -sample-rate: sample rate in one output vector thus end-time of a output vector is
     *     start-time + size / sampe-rate.
     *    -frame-shift: incement of start-times.
     */
    class RandomVectorNode : public Flow::SourceNode {
        typedef Flow::SourceNode Precursor;
    public:
        typedef f32 Data;
    public:
        static const Core::ParameterInt paramSize;
        static const Core::ParameterFloat paramStartTime;
        static const Core::ParameterFloat paramSampleRate;
        static const Core::ParameterFloat paramFrameShift;
    private:
        Math::RandomVectorGenerator *randomVectorGenerator_;
        size_t size_;
        f64 sampleRate_;
        Flow::Time startTime_;
        Flow::Time frameShift_;
        size_t nOutputs_;
    private:
        void setType(Math::RandomVectorGenerator::Type);
        Flow::Vector<Data>* createOutput() const;
        void reset() { nOutputs_ = 0; }
    public:
        static std::string filterName() {
            return std::string("signal-random-vector-") + Core::Type<Data>::name;
        }
        RandomVectorNode(const Core::Configuration &c);
        virtual ~RandomVectorNode();

        virtual bool configure();
        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool work(Flow::PortId p);
    };
}

#endif // _SIGNAL_RANDOM_VECTOR_HH
