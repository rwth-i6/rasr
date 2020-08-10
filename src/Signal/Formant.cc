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
#include <math.h>

#include "Formant.hh"

using namespace Signal;

bool FormantExtraction::calculateProperties(const Flow::Vector<_float>& in, u8 formant_index,
                                            _float estimation_error, _float a1, _float a2, _float energy,
                                            Formant& formant) {
    bool ret = true;

    if (!calculateResonanceFrequency(a1, a2, formant.frequency_)) {
        if (formant_index == 0)
            formant.frequency_ = 0.0;
        else if (formant_index == (getMaxNrFormant() - 1))
            formant.frequency_ = 5000.0;
        else
            formant.frequency_ = 2000.0;

        in.dump(warning("%d. formant frequency set to %f. Frame=", formant_index, formant.frequency_));
    }

    if (!calculateMaxAmplitude(a1, a2, formant.amplitude_)) {
        formant.amplitude_ = 1.0;
        in.dump(warning("%d. formant amplitude set to %f. Frame=", formant_index, formant.amplitude_));
    }
    formant.amplitude_ *= sqrt(estimation_error);

    if (!calculateBandwidth(a1, a2, formant.bandwidth_)) {
        formant.bandwidth_ = 0.0;
        in.dump(warning("%d. formant bandwidth set to %f. Frame=", formant_index, formant.bandwidth_));
    }

    formant.energy_ = energy;

    return ret;
}

bool FormantExtraction::calculateResonanceFrequency(_float a1, _float a2, _float& frequency) {
    _float tmp = -a1 * (1 + a2) / (4.0 * a2);
    frequency  = acos(tmp);

    if (fabs(tmp) > 1.0) {
        _float relative_damping;
        if (!calculateContinuousModel(a1, a2, frequency, relative_damping)) {
            warning("calculateResonanceFrequency failed: a1 = %f, a2 = %f", a1, a2);
            return false;
        }
        warning("calculateResonanceFrequency: relative damping too high = %f fr = %f ; a1 = %f, a2 = %f",
                relative_damping, frequency * (_float)sample_rate_ / 2.0 / M_PI, a1, a2);
    }

    frequency *= (_float)sample_rate_ / 2.0 / M_PI;
    return true;
}

bool FormantExtraction::calculateMaxAmplitude(_float a1, _float a2, _float& amplitude) {
    _float tmp = a1 * a1 + (1 - a2) * (1 - a2) - (a1 * a1 * (1 + a2) * (1 + a2) / (4 * a2));
    if (a2 == 0 || tmp <= 0.0) {
        _float resonance_omega;
        _float relative_damping;
        if (!calculateContinuousModel(a1, a2, resonance_omega, relative_damping) ||
            relative_damping < 0.7) {
            warning("calculateMaxAmplitude failed: a1 = %f, a2 = %f", a1, a2);
            return false;
        }

        // take the amplitude at 0 frequency
        tmp = (1 + a1 + a2) * (1 + a1 + a2);
        warning("calculateMaxAmplitude: relative damping too high = %f fr = %f ; a1 = %f, a2 = %f",
                relative_damping, resonance_omega * (_float)sample_rate_ / 2.0 / M_PI, a1, a2);
    }

    amplitude = 1.0 / sqrt(tmp);
    return true;
}

bool FormantExtraction::calculateBandwidth(_float a1, _float a2, _float& bandwidth) {
    _float resonance_omega;
    _float relative_damping;
    if (!calculateContinuousModel(a1, a2, resonance_omega, relative_damping)) {
        warning("calculateBandwidth failed: a1 = %f, a2 = %f", a1, a2);
        return false;
    }

    _float tmp1 = 1 - 2 * relative_damping * relative_damping;
    _float tmp2 = 2 * relative_damping * sqrt(1 - relative_damping * relative_damping);

    if (tmp1 >= tmp2)
        bandwidth = resonance_omega * (sqrt(tmp1 + tmp2) - sqrt(tmp1 - tmp2));
    else
        bandwidth = resonance_omega * sqrt(sqrt(2));

    bandwidth *= (sample_rate_ / 2.0 / M_PI);
    return true;
}

bool FormantExtraction::calculateContinuousModel(_float  a1,
                                                 _float  a2,
                                                 _float& resonance_omega,
                                                 _float& relative_damping) {
    if (a2 < 0.0) {
        warning("calculateContinuousModel failed: a1 = %f, a2 = %f", a1, a2);
        return false;
    }
    _float delta = ::log(a2) / 2.0;

    _float omega = -a1 / 2.0 / sqrt(a2);
    if (fabs(omega) > 1.0) {
        if (fabs(omega) < 1.2)  // case of omega = 0. recover rounding error
        {
            warning("calculateContinuousModel omega rounded to 1(cont.: 0) a1 = %f, a2 = %f, delta = %f, omega = %f", a1, a2, delta, omega);
            omega = omega > 0 ? 1.0 : -1.0;
        }
        else {
            warning("calculateContinuousModel failed: a1 = %f, a2 = %f, delta = %f, omega = %f", a1, a2, delta, omega);
            return false;
        }
    }
    omega = acos(omega);

    resonance_omega  = sqrt(delta * delta + omega * omega);
    relative_damping = -delta / resonance_omega;
    return true;
}
