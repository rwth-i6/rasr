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
#ifndef _LATTICE_SMOOTHING_FUNCTION_HH
#define _LATTICE_SMOOTHING_FUNCTION_HH

#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Core/XmlStream.hh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Lattice {

    // Smoothing function f: identity + base class
    class SmoothingFunction
    {
    private:
                f64 sumF_;
    public:
                SmoothingFunction();
                virtual ~SmoothingFunction() {}
                virtual const std::string name() const {
                        return "identity";
                }
                virtual f64 f(f64 x) const {
                        return x;
                }
                f64 df(f64 x) const {
                        return dfx(x) / x;
                }
                // a more numerically stable version for df(x)*x
                virtual f64 dfx(f64 x) const {
                        return 1 * x;
                }
                virtual void updateStatistics(f64 x) {
                        sumF_ += f(x);
                }
                virtual void dumpStatistics(Core::XmlWriter &) const;

                // factory
                enum Type {
                        identity,
                        log,
                        sigmoid,
                        unsupervized };
                static Core::Choice choiceType;
                static Core::ParameterChoice paramType;
                static SmoothingFunction* createSmoothingFunction(const Core::Configuration &);
    };

    class LogSmoothingFunction : public SmoothingFunction
    {
                typedef SmoothingFunction Precursor;
    private:
                static const Core::ParameterFloat paramM;
    private:
                f64 xM_;
    public:
                LogSmoothingFunction(const Core::Configuration &);
                virtual ~LogSmoothingFunction() {}
                virtual const std::string name() const {
                        return Core::form("log(%f)", std::log(xM_));
                }
                // f(x)
                virtual f64 f(f64 x) const {
                        x = std::max(x, Core::Type<f64>::epsilon);
                        return std::log(x / (x + xM_ * (1 - x)));
                }
                // f'(x)*x
                virtual f64 dfx(f64 x) const {
                        x = std::max(x, (f64)Core::Type<f32>::epsilon);
                        return (1 / x) - (1 - xM_) / (x + xM_ * (1 - x));
                }
                virtual void dumpStatistics(Core::XmlWriter &) const;
    };

    class SigmoidSmoothingFunction : public SmoothingFunction
    {
                typedef SmoothingFunction Precursor;
    private:
                static const Core::ParameterFloat paramBeta;
                static const Core::ParameterFloat paramM;
    private:
                f64 beta_;
                f64 xM_;
    public:
                SigmoidSmoothingFunction(const Core::Configuration &);
                virtual ~SigmoidSmoothingFunction() {}
                virtual const std::string name() const {
                        return Core::form("sigmoid(%f,%f)", beta_, std::log(xM_));
                }
                // f(x)
                virtual f64 f(f64 x) const {
                        return pow(x, beta_) / (pow(xM_ * (1 - x), beta_) + pow(x, beta_));
                }
                // f'(x)*x
                virtual f64 dfx(f64 x) const {
                        x = std::max(x, (f64)Core::Type<f32>::epsilon);
                        return beta_ * f(x) * (1 - f(x)) / (1 - x);
                }
                virtual void dumpStatistics(Core::XmlWriter &) const;
    };

    class UnsupervizedSmoothingFunction : public SmoothingFunction
    {
                typedef SmoothingFunction Precursor;
    private:
                static const Core::ParameterFloat paramA;
                static const Core::ParameterFloat paramB;
    private:
                f64 dA_, dB_, dS_;
                f64 xBn_, xAn_, xAp_, xBp_;
                u32 nInfB_, nBA_, nAA_, nAB_, nBInf_;
    public:
                UnsupervizedSmoothingFunction(const Core::Configuration &);
                virtual ~UnsupervizedSmoothingFunction() {}
                virtual const std::string name() const {
                        return Core::form("unsupervized(%f,%f)", dA_, dB_);
                }
                // f(x)
                virtual f64 f(f64 x) const;
                // f'(x)*x
                virtual f64 dfx(f64 x) const;
                virtual void updateStatistics(f64 x);
                virtual void dumpStatistics(Core::XmlWriter &) const;
    };

}

#endif // _LATTICE_SMOOTHING_FUNCTION_HH
