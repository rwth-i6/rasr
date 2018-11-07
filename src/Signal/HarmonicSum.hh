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
#ifndef _SIGNAL_HARMONIC_SUM_HH
#define _SIGNAL_HARMONIC_SUM_HH


#include <Core/Choice.hh>
#include <Core/Parameter.hh>
#include <Core/Utility.hh>

#include <Flow/Data.hh>
#include <Flow/Vector.hh>

#include "Node.hh"

namespace Signal {


    /** harmonicSum
     * s(n) = sum_{h = 1}^{h = H} (x_{(h * n) mod T) where
     *
     * The x is expected to be periodic (!) with a periodicity of T.
     * Thus @param x represents one periode, i.e. it is of length T.
     *
     * Length of @param s determines the interval for the index n.
     * @param H gives the maximum schrinkage.
     */


    template<class Type>
    void harmonicSum(const std::vector<Type> &x, std::vector<Type> &s, u32 H) {

        ensure(x.size() >= s.size());

        u32 N = s.size();
        u32 T = x.size();
        H = H ? H : T;

        std::fill(s.begin(), s.end(), 0);

        for(u32 n = 0; n < N; n ++) {
            for(u32 h = 1; h <= H; h ++)
                s[n] += x[(n * h) % T];
        }
    }

    /** harmonicProduct
     * s(n) = prod_{h = 1}^{h = H} (x_{(h * n) mod T) where
     *
     * The x is expected to be periodic (!) with a periodicity of T.
     * Thus @param x represents one periode, i.e. it is of length T.
     *
     * Length of @param s determines the interval for the index n.
     * @param H gives the maximum schrinkage.
     *
     * normalization: to keep the product representable at high values of H
     * - for each h the normalization factor is
     *   square root of the product of the energy of signal shrunken by h and
     *   the energy of the harmonic product at  h -1
     *  - this normalization is motivated by the fourier transformed equivalent of product
     *    of symethric functions: cross-correlation.
     */

    template<class Type>
    void harmonicProduct(const std::vector<Type> &x, std::vector<Type> &s, u32 H) {

        ensure(x.size() >= s.size());

        u32 N = s.size();
        u32 T = x.size();
        H = H ? H : T;

        s.resize(T);

        std::fill(s.begin(), s.end(), 1);

        for(u32 h = 1; h <= H; h ++) {

            Type energy = 0;

            for(u32 n = 0; n < T; n ++)
                energy += x[(n * h) % T] * x[(n * h) % T];

            Type harmonicProductEnergy = 0;

            for(u32 n = 0; n < T; n ++)
                harmonicProductEnergy += s[n] * s[n];

            Type normalize = sqrt(2 * energy * harmonicProductEnergy) / x.size();

            for(u32 n = 0; n < T; n ++)
                s[n] *= x[(n * h) % T] / normalize;
        }

        s.resize(N);
    }

    /** HarmonicSumNode */

    class HarmonicSumNode : public SleeveNode {
    public:

        static Core::ParameterFloat paramSize;
        static Core::ParameterInt paramH;

    protected:

        f32 continuousSize_;
        u32 size_;

        u32 H_;

        f64 sampleRate_;

        bool needInit_;

    protected:

        void init();

        void initOutput(const std::vector<f32> &x, std::vector<f32> &s);

        void setH(u32 H) { H_ = H; }

        void setContinuousSize(const f32 size) { continuousSize_ = size; needInit_ = true; }

        void setSampleRate(const f64 sampleRate) { sampleRate_ = sampleRate; needInit_ = true; }

        virtual void apply(const std::vector<f32> &x, std::vector<f32> &s) {
            harmonicSum(x, s, H_);
        }

    public:

        static std::string filterName() { return "signal-harmonic-sum"; }

        HarmonicSumNode(const Core::Configuration &c);

        virtual ~HarmonicSumNode() {}

        virtual bool setParameter(const std::string &name, const std::string &value);

        virtual bool configure();

        virtual bool work(Flow::PortId p);
    };

    /** HarmonicProductNode */

    class HarmonicProductNode : public HarmonicSumNode {
    public:

        typedef HarmonicSumNode Precursor;

    protected:

        virtual void apply(const std::vector<f32> &x, std::vector<f32> &s) {
            harmonicProduct(x, s, H_);
        }

    public:

        static std::string filterName() { return "signal-harmonic-product"; }

        HarmonicProductNode(const Core::Configuration &c) : Core::Component(c), Precursor(c) {}

        virtual ~HarmonicProductNode() {}
    };
}



#endif // _SIGNAL_HARMONIC_SUM_HH
