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
#ifndef _SIGNAL_CONVOLUTION_HH
#define _SIGNAL_CONVOLUTION_HH

#include <Core/Utility.hh>
#include "FastFourierTransform.hh"

namespace Signal {

/** convolution
 * y_t = sum_{tau} ( x_tau * h_{t - tau} )
 */

template<class T>
void convolution(const std::vector<T>& x, const std::vector<T>& h, std::vector<T>& y) {
    if (y.empty())
        y.resize(x.size() + h.size() - 1);

    std::fill(y.begin(), y.end(), 0);

    u32 tLast = std::min(y.size(), x.size() + h.size() - 1);

    for (u32 t = 0; t < tLast; t++) {
        u32 tauFirst = t < h.size() ? 0 : t - (h.size() - 1);  // max(0, t  - (h.size() - 1))
        u32 tauLast  = std::min(x.size(), t + 1);

        for (u32 tau = tauFirst; tau < tauLast; tau++)
            y[t] += x[tau] * h[t - tau];
    }
}

/** Convolution: Fast Fourier Transform based implementation of time doimain convolution */

class Convolution {
public:
    typedef f32 Data;

private:
    std::vector<Data> signalSpectrum_;
    u32               maximalSignalSize_;

    std::vector<Data> responseSpectrum_;
    u32               responseSize_;

    u32 outputBegin_;
    u32 outputEnd_;

private:
    /** minimum length which avoids wrap around */
    u32 fourierTransformLength() const {
        return maximalSignalSize_ + responseSize_ - 1;
    }

    void copyResult(u32 signalSize, std::vector<Data>& output) const;

public:
    Convolution();

    /** sets response function */
    void setResponse(const std::vector<Data>& response, u32 maximalSignalSize);
    /** set positive side (including zeroth element) of a symmetric response function
     *
     *  remark: for symmetric responses, input signal needs to be extended only by half
     *  of the full response size.
     */
    void setSymmetricResponse(const std::vector<Data>& responsePositiveSide, u32 maximalSignalSize);

    /** returns the convolution of signal with the response */
    void apply(const std::vector<Data>& response, std::vector<Data>& signal);
    /** returns the convolution of signal with the response */
    void apply(std::vector<Data>& signal);

    u32 maximalSignalSize() const {
        return maximalSignalSize_;
    }

    /** result of convolution is delived for the interval [outputBegin_..outputEnd_) */
    void setOutputBegin(u32 begin) {
        outputBegin_ = begin;
    }
    u32 outputBegin() const {
        return outputBegin_;
    }

    /** result of convolution is delived for the interval [outputBegin_..outputEnd_)
     *
     *  If outputEnd_ is set to u32::max then end of the output interval is set to
     *  signalSize + responseSize - 1.
     */
    void setOutputEnd(u32 end) {
        outputEnd_ = end;
    }
    u32 outputEnd() const {
        return outputEnd_;
    }
};
}  // namespace Signal

#endif  // _SIGNAL_CONVOLUTION_HH
