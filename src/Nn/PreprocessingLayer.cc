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
/* Implementation of neural networks */

#include <cmath>
#include <numeric>

#include "PreprocessingLayer.hh"

// for reading/writing vectors/matrices
#include <Core/MatrixParser.hh>
#include <Core/VectorParser.hh>
#include <Math/Matrix.hh>
#include <Math/Module.hh>
#include <Math/Vector.hh>

using namespace Nn;

/*===========================================================================*/
template<typename T>
LogarithmPreprocessingLayer<T>::LogarithmPreprocessingLayer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkLayer<T>(config) {}

template<typename T>
LogarithmPreprocessingLayer<T>::~LogarithmPreprocessingLayer() {}

/**	Transform the features by logarithm
 *
 * 	@param	source	Input features
 * 	@param	target	Transformed (output) features
 * 	@param	nFrames	Number of frames to be processed
 *
 */
template<typename T>
void LogarithmPreprocessingLayer<T>::_forward(const NnMatrix& input, NnMatrix& output) {
    if (&input != &output) {
        output.copy(input);
    }

    output.log();
}

template<typename T>
inline void LogarithmPreprocessingLayer<T>::_backpropagateActivations(const NnMatrix& errorSignalIn,
                                                                      NnMatrix& errorSignalOut, const NnMatrix& activations) {
    if (&errorSignalIn != &errorSignalOut) {
        errorSignalOut.copy(errorSignalIn);
    }
    // errorSignalOut = activations .^ (-1)
    errorSignalOut.elementwiseDivision(activations);
}

template<typename T>
void LogarithmPreprocessingLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    // one stream, same matrix size
    require_eq(input.size(), 1u);
    require_eq(input[0]->nRows(), output.nRows());
    require_eq(input[0]->nColumns(), output.nColumns());

    _forward(*(input[0]), output);
}

template<typename T>
inline void LogarithmPreprocessingLayer<T>::backpropagateActivations(const NnMatrix& errorSignalIn,
                                                                     NnMatrix& errorSignalOut, const NnMatrix& activations) {
    require_eq(errorSignalIn.nRows(), errorSignalOut.nRows());
    require_eq(errorSignalIn.nColumns(), errorSignalOut.nColumns());

    _backpropagateActivations(errorSignalIn, errorSignalOut, activations);
}

/*===========================================================================*/
/**
 *	normalize the features by mean and variance
 *
 * 	@param	source	Input features
 * 	@param	target	Transformed (output) features
 * 	@param	nFrames	Number of frames to be processed
 *
 */
template<typename T>
const Core::ParameterString MeanAndVarianceNormalizationPreprocessingLayer<T>::paramFilenameMean(
        "mean-file", "Filename of the mean vector", "");

template<typename T>
const Core::ParameterString MeanAndVarianceNormalizationPreprocessingLayer<T>::paramFilenameStandardDeviation(
        "standard-deviation-file", "Filename of the standard deviation vector", "");

template<typename T>
MeanAndVarianceNormalizationPreprocessingLayer<T>::MeanAndVarianceNormalizationPreprocessingLayer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkLayer<T>(config),
          filenameMean_(paramFilenameMean(config)),
          filenameStandardDeviation_(paramFilenameStandardDeviation(config)),
          needInit_(true),
          mean_(),
          standardDeviation_() {
    this->log("mean file: ") << filenameMean_;
    this->log("standard deviation file: ") << filenameStandardDeviation_;
}

template<typename T>
void MeanAndVarianceNormalizationPreprocessingLayer<T>::_forward(const NnMatrix& input, NnMatrix& output) {
    require(!needInit_);

    if (&input != &output) {
        output.copy(input);
    }
    output.addToAllColumns(mean_, -1.0);
    output.divideRowsByScalars(standardDeviation_);
}

template<typename T>
void MeanAndVarianceNormalizationPreprocessingLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    // one stream, same matrix size
    require_eq(input.size(), 1u);
    require_eq(input[0]->nRows(), output.nRows());
    require_eq(input[0]->nColumns(), output.nColumns());

    _forward(*(input[0]), output);
}

template<typename T>
void MeanAndVarianceNormalizationPreprocessingLayer<T>::loadNetworkParameterMean(const std::string& filename) {
    // parse the xml file
    Math::Vector<T>            parameters;
    Core::XmlVectorDocument<T> parser(Core::Component::getConfiguration(), parameters);
    parser.parseFile(filename.c_str());

    // Convert Math::Vector -> Math::FastVector
    mean_.resize(parameters.size());
    for (u32 index = 0; index < parameters.size(); ++index) {
        mean_.at(index) = parameters[index];
    }
}

template<typename T>
void MeanAndVarianceNormalizationPreprocessingLayer<T>::loadNetworkParameterVariance(const std::string& filename) {
    // parse the xml file
    Math::Vector<T>            parameters;
    Core::XmlVectorDocument<T> parser(Core::Component::getConfiguration(), parameters);
    parser.parseFile(filename.c_str());

    // Convert Math::Vector -> Math::FastVector
    standardDeviation_.resize(parameters.size());
    for (u32 index = 0; index < parameters.size(); ++index) {
        standardDeviation_.at(index) = parameters[index];
    }
}

template<typename T>
void MeanAndVarianceNormalizationPreprocessingLayer<T>::loadNetworkParameters(const std::string& filename) {
    // load the mean vector
    loadNetworkParameterMean(filenameMean_);

    // load the variance matrix
    loadNetworkParameterVariance(filenameStandardDeviation_);

    mean_.initComputation();
    standardDeviation_.initComputation();

    // initialization done
    needInit_ = false;
}

/*===========================================================================*/
namespace Nn {

// (explicit) template instantiation for type f32
template class LogarithmPreprocessingLayer<f32>;
template class MeanAndVarianceNormalizationPreprocessingLayer<f32>;

// (explicit) template instantiation for type f64
template class LogarithmPreprocessingLayer<f64>;
template class MeanAndVarianceNormalizationPreprocessingLayer<f64>;

}  // namespace Nn
