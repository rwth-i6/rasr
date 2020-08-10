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
#ifndef _NN_NEURAL_NETWORK_FORWARD_NODE_HH
#define _NN_NEURAL_NETWORK_FORWARD_NODE_HH

#include <deque>

#include <Flow/Attributes.hh>
#include <Flow/Datatype.hh>
#include <Flow/Node.hh>
#include <Math/Matrix.hh>
#include <Mm/Types.hh>

#include "ClassLabelWrapper.hh"
#include "NeuralNetwork.hh"
#include "NeuralNetworkTrainer.hh"
#include "Prior.hh"
#include "Types.hh"

namespace Nn {

/**	neural network forward node.
 *
 *	Neural network forwarding as a flow node.
 *	Useful, when output of network is reused, for example for tandem GMM systems
 *
 */
class NeuralNetworkForwardNode : public Flow::SleeveNode {
    typedef Flow::SleeveNode Precursor;

public:
    typedef Mm::FeatureType FeatureType;

protected:
    typedef Types<f32>::NnVector NnVector;
    typedef Types<f32>::NnMatrix NnMatrix;

    typedef Flow::DataPtr<Flow::Vector<FeatureType>>                       FeatureVector;
    typedef Flow::DataPtr<Flow::TypedAggregate<Flow::Vector<FeatureType>>> AggregateFeatureVector;

public:
    static Core::ParameterString paramId;
    static Core::ParameterInt    paramBufferSize;
    static Core::ParameterBool   paramCheckValues;
    static Core::ParameterBool   paramDynamicBuffer;

private:
    const u32  bufferSize_;          // number of features that are processed at once
    const bool checkValues_;         // check output of network for finiteness
    const bool dynamicBuffer_;       // do not use fixed buffer size, but extend it until eos
    bool       needInit_;            // needs initialization of network
    bool       measureTime_;         // measure run time
    bool       aggregatedFeatures_;  // features are aggregated (multiple input streams)

    std::vector<u32>                    nFeatures_;
    std::vector<NnMatrix>               inputBuffer_;  // features are saved in buffer and then processed in batch mode
    std::vector<FeatureVector>          featureBuffer_;
    std::vector<AggregateFeatureVector> aggregateBuffer_;

    NnVector column_;              // one column of the output
    u32      outputDimension_;     // output dimension of the network
    u32      totalOutputFrames_;   // number of output timeframes
    u32      currentOutputFrame_;  // keeps track of the output-column

    NeuralNetwork<FeatureType> network_;  // the neural network
    Prior<f32>                 prior_;    // state prior
public:
    static std::string filterName() {
        return std::string("neural-network-forward");
    };

public:
    NeuralNetworkForwardNode(Core::Configuration const& c);
    virtual ~NeuralNetworkForwardNode();

    void initialize(std::vector<u32> const& nFeatures);

    virtual bool configure();
    virtual bool setParameter(const std::string& name, const std::string& value);
    // override Flow::Node::work
    virtual bool work(Flow::PortId p);

private:
    // checks whether we have aggregated or simple Flow features
    bool configureDataType(Core::Ref<const Flow::Attributes> a, const Flow::Datatype* d);
    // do the work
    void processBuffer();
    bool bufferEmpty() const {
        return std::max(featureBuffer_.size(), aggregateBuffer_.size()) == 0ul;
    }
    bool bufferFull() const {
        return (not dynamicBuffer_) and std::max(featureBuffer_.size(), aggregateBuffer_.size()) >= bufferSize_;
    }
    // send feature to ouput port
    bool putNextFeature();
};

}  // namespace Nn

#endif  // _NN_NEURAL_NETWORK_NODE_HH
