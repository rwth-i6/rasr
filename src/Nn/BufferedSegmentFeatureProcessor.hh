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
#ifndef _NN_BUFFEREDSEGMENTFEATUREPROCESSOR_HH
#define _NN_BUFFEREDSEGMENTFEATUREPROCESSOR_HH

#include <Core/Component.hh>
#include "BufferedFeatureExtractor.hh"

namespace Nn {

/* BufferedSegmentFeatureProcessor uses BufferedFeatureExtractor to extract
 * features, prepare them for NN training and buffer them.
 * It expects to operate on whole segments (BufferType::utterance).
 *
 */
template<typename FloatT>
class BufferedSegmentFeatureProcessor : public BufferedFeatureExtractor<FloatT>
{
public:
    typedef BufferedFeatureExtractor<FloatT> Precursor;

public:
    BufferedSegmentFeatureProcessor(const Core::Configuration &config);
    virtual ~BufferedSegmentFeatureProcessor();

    virtual void processBuffer();

    // Override in BufferedFeatureExtractor, which creates, owns and feeds the trainer.
    virtual NeuralNetworkTrainer<FloatT>* createTrainer(const Core::Configuration& config);

};

}

#endif // BUFFEREDSEGMENTFEATUREPROCESSOR_HH
