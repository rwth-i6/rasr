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
#include <Math/Complex.hh>
#include "Convolution.hh"

using namespace Signal;

// Convolution
//////////////

Convolution::Convolution() :
    maximalSignalSize_ (0),
    responseSize_(0),
    outputBegin_(0),
    outputEnd_(Core::Type<u32>::max)
{}


void Convolution::setResponse(const std::vector<Data> &response, u32 maximalSignalSize) {

    responseSize_ = response.size();

    maximalSignalSize_ = maximalSignalSize;


    RealFastFourierTransform fft(fourierTransformLength());
    fft.transform(responseSpectrum_ = response);
}


void Convolution::setSymmetricResponse(const std::vector<Data> &responsePositiveSide, u32 maximalSignalSize) {

    responseSize_ = responsePositiveSide.size();

    maximalSignalSize_ = std::max(maximalSignalSize, u32((responsePositiveSide.size() - 1) / 2) * 2);


    RealFastFourierTransform fft;

    responseSpectrum_.resize(fft.setLength(fourierTransformLength()));


    std::fill(responseSpectrum_.begin(), responseSpectrum_.end(), 0);

    if (!responsePositiveSide.empty()) {

        std::copy(responsePositiveSide.begin(), responsePositiveSide.end(),
                  responseSpectrum_.begin());
        std::copy(responsePositiveSide.begin() + 1, responsePositiveSide.end(),
                  responseSpectrum_.rbegin());
    }


    fft.transform(responseSpectrum_);
}


void Convolution::apply(std::vector<Data> &signal) {

    require(signal.size() <= maximalSignalSize_);
    verify(responseSize_ > 0);

    RealFastFourierTransform fft(fourierTransformLength());
    if (!fft.transform(signalSpectrum_ = signal))
        defect();


    Math::transformAlternatingComplexToAlternatingComplex(
        responseSpectrum_.begin(), responseSpectrum_.end(),
        signalSpectrum_.begin(), signalSpectrum_.begin(), std::multiplies<std::complex<Data> >());


    RealInverseFastFourierTransform iFft(fft.length(), fft.outputSampleRate());
    if (!iFft.transform(signalSpectrum_))
        defect();


    copyResult(signal.size(), signal);
}


void Convolution::copyResult(u32 signalSize, std::vector<Data> &output) const {

    u32 begin = outputBegin_;

    u32 end = (outputEnd_ == Core::Type<u32>::max) ? signalSize + responseSize_ - 1 : outputEnd_;


    verify(begin <= end);

    verify(end < signalSpectrum_.size());


    output.resize(end - begin);

    std::copy(signalSpectrum_.begin() + begin, signalSpectrum_.begin() + end, output.begin());
}


void Convolution::apply(const std::vector<Data> &response, std::vector<Data> &signal) {

    setResponse(response, signal.size());

    apply(signal);

    maximalSignalSize_ = 0;
}
