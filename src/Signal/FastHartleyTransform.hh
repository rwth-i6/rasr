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
#ifndef _SIGNAL_HARTLEY_TRANSFORM_HH
#define _SIGNAL_HARTLEY_TRANSFORM_HH

#include <vector>
//#include <function.h>
#include <functional>
//#include <algo.h>
#include <algorithm>

#include <Core/Types.hh>

namespace Signal {

/** HartleyFourierTransform: performs fourier transform of real vector based on Hartley transform
 * Delivers N / 2 + 1 complex values real and imaginary part alternating,
 * where N is number of FFT points.
 */

class FastHartleyTransform {
public:
    typedef f32 Data;

protected:
    u32 length_;

    f32 sampleRate_;

    std::vector<u32> bitReverse_;

protected:
    void setBitReserve(u32 length);

    void hartleyTransform(std::vector<Data>& fz) const;

    void zeroPadding(std::vector<Data>& data) const;

public:
    FastHartleyTransform(const u32 length = 0, const f32 sampleRate = 1);

    void transform(std::vector<Data>& data) const;

    void inverseTransform(std::vector<Data>& data) const;

    u32 length() const {
        return length_;
    }
    void setLength(u32 l) {
        if (l != length())
            setBitReserve(length_ = l);
    }

    f32 sampleRate() const {
        return sampleRate_;
    }
    void setSampleRate(const f32 sampleRate) {
        sampleRate_ = sampleRate;
    }
};

/** hartleyToFourier: converts  Hartley coefficients to Fourier coefficients
 */

void hartleyToFourier(const std::vector<f32>& hartley, std::vector<f32>& fourier);

/** hartleyToFourierAmplitude: converts Hartley coefficients to amplitude of Fourier coefficients
 *
 * @param hartley and @param amplitude can be the same object
 */

void hartleyToFourierAmplitude(const std::vector<f32>& hartley, std::vector<f32>& amplitude);

/** hartleyToFourierPhase: converts  Hartley coefficients to phase of Fourier coefficients
 *
 * @param hartley and @param phase can be the same object
 */

void hartleyToFourierPhase(const std::vector<f32>& hartley, std::vector<f32>& phase);

/** fourierToHartley: converts Fourier coefficients to Hartley coefficients
 */

void fourierToHartley(const std::vector<f32>& fourier, std::vector<f32>& hartley);
}  // namespace Signal

#endif  // _SIGNAL_HARTLEY_TRANSFORM_HH
