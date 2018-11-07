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
#ifndef _SPEECH_AUXILIARY_SEGMENTWISE_TRAINER_HH
#define _SPEECH_AUXILIARY_SEGMENTWISE_TRAINER_HH

#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Flow/Cache.hh>
#include "Alignment.hh"
#include <Bliss/CorpusDescription.hh>
#include <Fsa/Automaton.hh>

namespace Speech {

    /**
     *  used for MCE
     */
    class SigmoidFunction : public Core::Component
    {
        typedef Core::Component Precursor;
    private:
        static const Core::ParameterFloat paramBeta;
    private:
        f32 beta_;
    private:
        f32 argument(const Fsa::Weight &totalInvNum, const Fsa::Weight &totalInvDen) const;
    public:
        SigmoidFunction(const Core::Configuration &);

        Fsa::Weight f(const Fsa::Weight &totalInvNum, const Fsa::Weight &totalInvDen) const;
        Fsa::Weight df(const Fsa::Weight &totalInvNum, const Fsa::Weight &totalInvDen) const;
    };

    struct PosteriorFsa
    {
        Fsa::ConstAutomatonRef fsa;
        Fsa::Weight totalInv;
        operator bool() const { return fsa; }
    };

    Fsa::ConstAutomatonRef getDenominatorWeights(PosteriorFsa num, PosteriorFsa den);

} // namespace Speech

#endif // _SPEECH_AUXILIARY_SEGMENTWISE_TRAINER_HH
