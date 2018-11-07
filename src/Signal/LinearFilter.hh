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
#ifndef _SIGNAL_LINEARFILTER_HH_
#define _SIGNAL_LINEARFILTER_HH_

#include <Core/Assertions.hh>
#include <Core/Parameter.hh>
#include <Flow/Vector.hh>

#include "Node.hh"

namespace Signal {

    // LinearFilter
    ///////////////

    template<class T>
    class LinearFilter
    {
    private:
        std::vector<T> B_tilde_;
        std::vector<T> A_tilde_;

        std::vector<T> u_;
        std::vector<T> y_;

        T work(T u)
            {
                T y = 0.0;

                for(s16 j = B_tilde_.size() - 1; j >= 0; j --)
                    {
                        y += B_tilde_[j] * u_[j];
                        u_[j] = j ? u_[j - 1] : u;
                    }

                for(s16 i = A_tilde_.size() - 1; i >= 0; i --)
                    {
                        y -= A_tilde_[i] * y_[i];
                        y_[i] = i ? y_[i - 1] : y;
                    }

                return y;
            }

    public:
        LinearFilter() {}
        ~LinearFilter() {}

        void setB(const std::vector<T>& B_tilde)
            { B_tilde_ = B_tilde; u_.resize(B_tilde.size()); for(u8 i = 0; i < u_.size(); i++) u_[i] = 0; }
        void setA(const std::vector<T>& A_tilde)
            { A_tilde_ = A_tilde; y_.resize(A_tilde.size()); for(u8 j = 0; j < y_.size(); j++) y_[j] = 0; }
        void setY0(const std::vector<T>& y0)
            { verify(y_.size() == y0.size());
            for(u8 i = 0; i < u_.size(); i++) u_[i] = 0; for(u8 j = 0; j < y_.size(); j++) y_[j] = y0[j]; }

        bool work(std::vector<T>& s) { return work(&s, s); }
        bool work(const std::vector<T>* u, std::vector<T>& y)
            {
                if (u)
                    {
                        y.resize(u->size());
                        for(u32 t = 0; t < y.size(); t ++)
                            y[t] = work((*u)[t]);
                    }
                else
                    {
                        for(u32 t = 0; t < y.size(); t ++)
                            y[t] = work(0.0);
                    }

                return true;
            }

        void reset()
            { for(u8 i = 0; i < u_.size(); i++) u_[i] = 0; for(u8 j = 0; j < y_.size(); j++) y_[j] = 0; }
    };

    // LinearFilterParameter
    ////////////////////////

    class LinearFilterParameter : public Flow::Timestamp {
        typedef Flow::Timestamp Precursor;
        typedef LinearFilterParameter Self;
    public:
        typedef f32 _float;
    private:
        std::vector<_float> B_tilde_;
        std::vector<_float> A_tilde_;
        std::vector<_float> y0_;
    public:
        static const Flow::Datatype *type() {
            static Flow::DatatypeTemplate<LinearFilterParameter> dt("linear-filter-parameter");
            return &dt;
        }
        LinearFilterParameter() : Precursor(type()) {}
        virtual ~LinearFilterParameter() {}

        virtual Data* clone() const { return new LinearFilterParameter(*this); };

        virtual Core::XmlWriter& dump(Core::XmlWriter &o) const;

        std::vector<_float>& getB() { return B_tilde_; };
        std::vector<_float>& getA() { return A_tilde_; };
        std::vector<_float>& getY0() { return y0_; };
    };

    // LinearFilterNode
    ///////////////////

    class LinearFilterNode : public SleeveNode, public LinearFilter<f32>
    {
    private:
        static Core::ParameterString paramB;
        static Core::ParameterString paramA;
        static Core::ParameterString paramY0;
        static Core::ParameterInt paramZeroInputLength;

        u32 zero_input_length;

        bool parsePolinom(const std::string& value, std::vector<f32>& v);

    public:
        static std::string filterName() { return "signal-linear-filter"; }
        LinearFilterNode(const Core::Configuration &c);
        virtual ~LinearFilterNode() {}

        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool configure();

        virtual Flow::PortId getInput(const std::string &name) {
                return (name == "parameter" ? 1 : 0);
        }

        virtual bool work(Flow::PortId p);
        virtual void reset() { LinearFilter<f32>::reset(); }
    };

}

#endif // _SIGNAL_LINEARFILTER_HH_
