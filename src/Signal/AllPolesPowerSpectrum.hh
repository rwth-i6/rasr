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
#ifndef _SIGNAL_ALL_POLES_POWER_SPECTRUM_HH
#define _SIGNAL_ALL_POLES_POWER_SPECTRUM_HH

#include <Math/Nr/nr.h>
#include <Flow/Node.hh>

namespace Signal
{

    /** Calculate the all poles (also called as maximum entropy) power spectrum estimate.
     *  @param "a" is all poles (autoregression) coefficients a1, ..., aN,
     *  @param gain is all poles gain,
     *  @param totalLength is total length of the estimated power spectrum.
     *  @param powerSpectrum is result buffer. Its length is set to totalLength  / 2 + 1,
     *    since power spectrum of real functions is always symmetric.
     */
    template<class T> void allPolesPowerSpectrum(float gain, const std::vector<float> &a,
                                                 size_t totalLength, std::vector<T> &powerSpectrum)
    {
        require(totalLength > 0);
        size_t halfLength = totalLength / 2 + 1;
        powerSpectrum.resize(halfLength);
        for(size_t n = 0; n < halfLength; ++ n)
            powerSpectrum[n] = Math::Nr::evlmem((float)n / (float)totalLength, a, gain);
    }

    /** Calculate the all poles (also called as maximum entropy) power spectrum estimate.
     *  Input: autoregressive-parameter.
     *  Output power spectrum estimate. (It is multiplied by sampleRate^2 to make it conform with FFT.)
     *  Parameter: total length (given in discrete or continuos domain) of power spectrum.
     *    The acctual length is set to totalLength  / 2 + 1, since power spectrum of real
     *    functions is always symmetric.
     */
    class AllPolesPowerSpectrumNode : public Flow::SleeveNode {
        typedef Flow::SleeveNode Precursor;
    private:
        static const Core::ParameterInt paramDiscreteTotalLength;
        static const Core::ParameterFloat paramContinuousTotalLength;
    private:
        size_t discreteTotalLength_;
        f64 continuousTotalLength_;
        size_t totalLength_;
        f64 sampleRate_;
    private:
        void init(f64 sampleRate);
    public:
        static std::string filterName() {
            return std::string("signal-all-poles-power-spectrum");
        }
        AllPolesPowerSpectrumNode(const Core::Configuration &);
        virtual ~AllPolesPowerSpectrumNode();

        virtual bool configure();
        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool work(Flow::PortId p);
    };
}

#endif // _SIGNAL_ALL_POLES_POWER_SPECTRUM_HH
