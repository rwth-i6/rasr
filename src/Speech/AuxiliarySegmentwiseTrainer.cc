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
#include "AuxiliarySegmentwiseTrainer.hh"
#include <Core/BinaryStream.hh>
#include <Flow/Data.hh>
#include <Flow/DataAdaptor.hh>
#include <Flow/Datatype.hh>
#include "AlignmentNode.hh"

using namespace Speech;

/**
 *  SigmoidFunction used for MCE
 */
const Core::ParameterFloat SigmoidFunction::paramBeta(
        "beta",
        "sets the value of the sigmoidal smoothing function",
        0.0004,
        Core::Type<f32>::delta);

SigmoidFunction::SigmoidFunction(const Core::Configuration& c)
        : Precursor(c),
          beta_(paramBeta(config)) {}

f32 SigmoidFunction::argument(const Fsa::Weight& totalInvNum, const Fsa::Weight& totalInvDen) const {
    f32 x = Core::Type<f32>::max;
    if ((f32(totalInvDen) - f32(totalInvNum)) > Core::Type<f64>::delta) {
        x = f32(totalInvNum);
        x -= (f32(totalInvDen) + log1p(-exp(f32(totalInvNum) - f32(totalInvDen))));
    }
    return x;
}

Fsa::Weight SigmoidFunction::f(const Fsa::Weight& totalInvNum, const Fsa::Weight& totalInvDen) const {
    f32 x = argument(totalInvNum, totalInvDen);
    return Fsa::Weight((tanh(beta_ * x) + 1) / 2 / beta_);
}

Fsa::Weight SigmoidFunction::df(const Fsa::Weight& totalInvNum, const Fsa::Weight& totalInvDen) const {
    f32 x = argument(totalInvNum, totalInvDen);
    f32 f = tanh(beta_ * x);
    return Fsa::Weight(1 - f * f);
}

namespace Speech {

/**
 *  DenominatorWeightsAutomaton
 */
class DenominatorWeightsAutomaton : public Fsa::ModifyAutomaton {
    typedef Fsa::ModifyAutomaton Precursor;

private:
    Fsa::ConstAutomatonRef fsaNum_;
    Fsa::Weight            totalInv_;

public:
    DenominatorWeightsAutomaton(PosteriorFsa num, PosteriorFsa den)
            : Precursor(Fsa::extend(den.fsa, den.fsa->semiring()->invert(den.totalInv))),
              fsaNum_(Fsa::extend(num.fsa, den.fsa->semiring()->invert(num.totalInv))) {
        totalInv_ = semiring()->extend(den.totalInv, Fsa::Weight(log1p(-exp(f32(num.totalInv) - f32(den.totalInv)))));
    }
    virtual ~DenominatorWeightsAutomaton() {}

    virtual std::string describe() const {
        return "denominator-weights(" + fsa_->describe() + ")";
    }
    virtual void modifyState(Fsa::State* sp) const {
        Fsa::ConstStateRef         spNum = fsaNum_->getState(sp->id());
        Fsa::State::const_iterator aNum  = spNum->begin();
        for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++a, ++aNum) {
            /*
             * a->weight() contains the unnormalized denominator arc posterior probability.
             */
            if (semiring()->compare(aNum->weight(), semiring()->zero()) < 0) {
                a->weight_ =
                        semiring()->extend(
                                a->weight(),
                                Fsa::Weight(-log1p(-exp(f32(a->weight()) - f32(aNum->weight())))));
            }
            a->weight_ = semiring()->extend(a->weight(), totalInv_);
        }
    }
};

Fsa::ConstAutomatonRef getDenominatorWeights(PosteriorFsa num, PosteriorFsa den) {
    return Fsa::ConstAutomatonRef(new DenominatorWeightsAutomaton(num, den));
}

}  //namespace Speech
