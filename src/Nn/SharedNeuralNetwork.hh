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
#ifndef SHAREDNEURALNETWORK_HH_
#define SHAREDNEURALNETWORK_HH_

#include "ClassLabelWrapper.hh"
#include "LinearAndActivationLayer.hh"
#include "NeuralNetwork.hh"
#include "Prior.hh"

namespace Nn {

/*
 * Provides a static neural network and the corresponding prior and class label wrapper.
 * The prior is removed from the bias of the output layer after network construction.
 *
 * We can not use the Singleton implementation, because the neural network does not have a default constructor.
 * Instead, the network has to be created by calling the create method which has a configuration argument.
 *
 */
class SharedNeuralNetwork  {
private:
    static NeuralNetwork<f32> *network_;
    static Prior<f32> *prior_;
    static ClassLabelWrapper *labelWrapper_;
public:
    SharedNeuralNetwork() {}

    ~SharedNeuralNetwork(){}

    static NeuralNetwork<f32>& network();

    static const ClassLabelWrapper& labelWrapper();

    static const Prior<f32>& prior();

    static bool hasInstance();

    static void create(const Core::Configuration &c);
};

}

#endif /* SHAREDNEURALNETWORK_HH_ */
