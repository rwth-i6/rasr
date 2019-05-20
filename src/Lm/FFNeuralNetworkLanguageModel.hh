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
#ifndef _LM_FF_NEURAL_NETWORK_LANGUAGE_MODEL_HH
#define _LM_FF_NEURAL_NETWORK_LANGUAGE_MODEL_HH

#include <Nn/NeuralNetwork.hh>

#include "AbstractNNLanguageModel.hh"

namespace Lm {

class FFNeuralNetworkLanguageModel : public AbstractNNLanguageModel {
public:
    typedef AbstractNNLanguageModel Precursor;
    typedef f32                     FeatureType;

    static Core::ParameterBool paramExpandOneHot;
    static Core::ParameterBool paramEagerForwarding;
    static Core::ParameterInt  paramContextSize;
    static Core::ParameterInt  paramHistorySize;
    static Core::ParameterInt  paramBufferSize;

    FFNeuralNetworkLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~FFNeuralNetworkLanguageModel();

    virtual History startHistory() const;
    virtual History extendedHistory(History const&, Token w) const;
    virtual Score   score(History const&, Token w) const;

protected:
    virtual void load();

private:
    bool   expand_one_hot_;
    bool   eager_forwarding_;
    size_t context_size_;  // number of words passed to the neural network
    size_t history_size_;  // length of history for recombination purposes (used to estimate runtime-performance for recurrent LMs)
    size_t buffer_size_;

    mutable Nn::NeuralNetwork<FeatureType> nn_;
};

}  // namespace Lm

#endif  // _LM_FF_NEURAL_NETWORK_LANGUAGE_MODEL_HH
