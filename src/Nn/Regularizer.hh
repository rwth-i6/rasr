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
#ifndef _NN_REGULARIZER_HH
#define _NN_REGULARIZER_HH

#include "NeuralNetwork.hh"
#include "Statistics.hh"
#include "Types.hh"

namespace Nn {

/*
 * base class for Regularizer
 */
template<typename T>
class Regularizer : virtual public Core::Component {
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    typedef Core::Component Precursor;
    enum RegularizerType {
        none,
        l1Regularizer,
        l2Regularizer,
        centeredRegularizer,
    };
    static const Core::Choice          choiceRegularizerType;
    static const Core::ParameterChoice paramRegularizerType;

public:
    Regularizer(const Core::Configuration& config);
    virtual ~Regularizer() {}
    virtual T objectiveFunction(NeuralNetwork<T>& network, T factor) {
        return 0;
    }
    virtual void addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor) {}

    static Regularizer<T>* createRegularizer(const Core::Configuration& config);
};

/*
 * l1-Regularizer (regularization with l1-norm)
 * ... + C * ||W||
 */
template<typename T>
class L1Regularizer : public Regularizer<T> {
    typedef Regularizer<T>              Precursor;
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    NnMatrix signMatrix_;
    NnVector signVector_;

public:
    L1Regularizer(const Core::Configuration& config);
    virtual ~L1Regularizer() {}
    virtual T    objectiveFunction(NeuralNetwork<T>& network, T factor);
    virtual void addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor);
};

/*
 * l2-Regularizer (regularization with l2-norm)
 * ... + C/2 * ||W||^2
 */
template<typename T>
class L2Regularizer : public Regularizer<T> {
    typedef Regularizer<T>              Precursor;
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

public:
    L2Regularizer(const Core::Configuration& config);
    virtual ~L2Regularizer() {}
    virtual T    objectiveFunction(NeuralNetwork<T>& network, T factor);
    virtual void addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor);
};

/*
 * Centered l2-Regularizer (regularization with l2-norm around W_0)
 * ... + C/2 * ||W - W_0||^2
 */
template<typename T>
class CenteredL2Regularizer : public Regularizer<T> {
    typedef Regularizer<T>              Precursor;
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    static const Core::ParameterString paramCenterParameters;
    NeuralNetwork<T>                   centerNetwork_;
    NnVector                           diffVector_;
    NnMatrix                           diffMatrix_;

public:
    CenteredL2Regularizer(const Core::Configuration& config);
    virtual ~CenteredL2Regularizer(){};
    virtual T    objectiveFunction(NeuralNetwork<T>& network, T factor);
    virtual void addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor);
};

}  // namespace Nn

#endif
