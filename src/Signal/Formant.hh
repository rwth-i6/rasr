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
#ifndef _SIGNAL_FORMANT_HH
#define _SIGNAL_FORMANT_HH

#include <Core/Types.hh>
#include <Core/Component.hh>
#include <Core/XmlStream.hh>

#include <Flow/Vector.hh>

namespace Signal {

    class Formant {
    public:
        typedef f32 _float;

    public:
        _float frequency_;
        _float amplitude_;
        _float bandwidth_;
        _float energy_;

        Formant() : frequency_(0), amplitude_(0), bandwidth_(0), energy_(0) {}

        Core::XmlWriter& dump(Core::XmlWriter& o) {
            return o << Core::XmlEmpty("formant")
                + Core::XmlAttribute("frequency", frequency_)
                + Core::XmlAttribute("amplitude", amplitude_)
                + Core::XmlAttribute("bandwidth", bandwidth_)
                + Core::XmlAttribute("energy", energy_);
        }
    };

    class FormantExtraction :
        public virtual Core::Component
    {
    public:
        typedef f32 _float;

    private:
        u32 sample_rate_;
        u8 max_nr_formant_;

    protected:
        bool init() { return sample_rate_ != 0; }

        bool calculateProperties(const Flow::Vector<_float>& in, u8 formant_index,
                                 _float estimation_error, _float a1, _float a2, _float energy,
                                 Formant& formant);
        bool calculateResonanceFrequency(_float a1, _float a2, _float& frequency);
        bool calculateMaxAmplitude(_float a1, _float a2, _float& amplitude);
        bool calculateBandwidth(_float a1, _float a2, _float& bandwidth);
        bool calculateContinuousModel(_float a1, _float a2,
                                      _float& resonance_frequency, _float& relative_damping);

    public:
        FormantExtraction(const Core::Configuration &c) : Component(c), sample_rate_(0), max_nr_formant_(0) {}
        virtual ~FormantExtraction() {}

        u8 getMaxNrFormant() { return max_nr_formant_; }
        void setMaxNrFormant(u8 max_nr_formant) { max_nr_formant_ = max_nr_formant; }

        void setSampleRate(u32 sample_rate) { sample_rate_ = sample_rate; }
        u32 getSampleRate() { return sample_rate_; }
    };
}

#endif // _SIGNAL_FORMANT_HH
