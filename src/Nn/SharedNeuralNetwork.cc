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
#include "SharedNeuralNetwork.hh"
#include <Core/Debug.hh>

using namespace Nn;

NeuralNetwork<f32>* SharedNeuralNetwork::network_      = 0;
Prior<f32>*         SharedNeuralNetwork::prior_        = 0;
ClassLabelWrapper*  SharedNeuralNetwork::labelWrapper_ = 0;

NeuralNetwork<f32>& SharedNeuralNetwork::network() {
    return *network_;
}

const ClassLabelWrapper& SharedNeuralNetwork::labelWrapper() {
    return *labelWrapper_;
}

const Prior<f32>& SharedNeuralNetwork::prior() {
    return *prior_;
}

bool SharedNeuralNetwork::hasInstance() {
    return network_ != 0 && prior_ != 0 && labelWrapper_ != 0;
}

void SharedNeuralNetwork::create(const Core::Configuration& c) {
    if (!hasInstance()) {
        network_ = new NeuralNetwork<f32>(c);
        network_->initializeNetwork(1);
        labelWrapper_ = new ClassLabelWrapper(Core::Configuration(c, "class-labels"));
        prior_        = new Prior<f32>(c);

        auto* topLayer        = &network().getTopLayer();
        auto* softmaxTopLayer = dynamic_cast<Nn::LinearAndSoftmaxLayer<f32>*>(topLayer);
        auto* biasTopLayer    = dynamic_cast<Nn::BiasLayer<f32>*>(topLayer);
        if (!softmaxTopLayer && !biasTopLayer) {
            auto* maxoutLayer = dynamic_cast<Nn::MaxoutVarLayer<f32>*>(topLayer);
            if (maxoutLayer)
                softmaxTopLayer = dynamic_cast<Nn::LinearAndSoftmaxLayer<f32>*>(&network().getLayer(maxoutLayer->getPredecessor(0)));
        }

        if (softmaxTopLayer) {
            // forward until softmax only
            // assume that log-prior is already removed from bias parameters of last layer
            // assume that parameters are already scaled according to mixture-scale
            if (softmaxTopLayer->evaluatesSoftmax())
                softmaxTopLayer->setEvaluateSoftmax(false);  // switch off the softmax
            Core::printLog("SharedNeuralNetwork: switched off softmax eval on softmax-layer");
        }
        else if (biasTopLayer) {
            Core::printLog("SharedNeuralNetwork: bias-layer is top layer, we assume it's in log space already");
        }
        else {
            Core::printWarning("SharedNeuralNetwork: top layer type is unknown, we assume it's in log space");
        }

        if (prior_->fileName() != "" && prior_->scale() != 0.0) {
            prior_->read();
            if (softmaxTopLayer) {
                softmaxTopLayer->removeLogPriorFromBias(prior());
                Core::printLog("SharedNeuralNetwork: substract log prior from softmax-layer bias");
            }
            else if (biasTopLayer) {
                biasTopLayer->removeLogPriorFromBias(prior());
                Core::printLog("SharedNeuralNetwork: substract log prior from bias-layer bias");
            }
            else {
                Core::printWarning("SharedNeuralNetwork: cannot subtract prior");
            }
        }
        else {
            Core::printLog("SharedNeuralNetwork: not substracting log prior (either file-name not set or scale is zero)");
        }
    }
}
