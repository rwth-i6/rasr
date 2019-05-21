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
#ifndef _SPEECH_ACOUSTIC_MODEL_RESCORER_HH
#define _SPEECH_ACOUSTIC_MODEL_RESCORER_HH

#include <Core/Configurable.hh>
#include <Fsa/Output.hh>
#include <Lattice/Rescorer.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/Types.hh>

namespace Speech {

template<class T>
class AcousticModelRescorer : public Core::Configurable {
private:
    const Mm::FeatureScorer*        scorer_;
    u32                             beamCount_;
    typename T::Weight              beamThreshold_;
    typedef Lattice::Rescorer<T, T> Rescorer;
    Rescorer*                       rescorer_;
    T                               trace_;

public:
    AcousticModelRescorer(const Core::Configuration& c)
            : Core::Configurable(c), scorer_(0), rescorer_(0) {
        beamCount_ = 0;
    }
    ~AcousticModelRescorer() {
        delete rescorer_;
    }

    void setAcousticModel(const Mm::FeatureScorer* scorer) {
        scorer_ = scorer;
        if (rescorer_)
            rescorer_->restart();
    }
    void setTransducer(const T& t) {
        delete rescorer_;
        rescorer_ = new Lattice::Rescorer<T, T>(t, trace_, beamThreshold_, beamCount_);
        trace_.setInputAlphabet(t.inputAlphabet());
        trace_.setOutputAlphabet(t.outputAlphabet());
    }
    void setBeamCount(typename T::Weight beamThreshold) {
        beamThreshold_ = beamThreshold;
    }
    void setBeamThreshold(u32 beamCount) {
        beamCount_ = beamCount;
    }

    void feed(Core::Ref<const Feature> f) {
        verify(scorer_);
        verify(rescorer_);
        rescorer_->feed(scorer_->get(f));
        Fsa::write(trace_, "/tmp/trace.fsm.gz");  // DEBUG
        getchar();
    }
    template<class Trace>
    void getBestPath(Trace& t) const {
        Fsa::write(trace_, "/tmp/trace.fsm.gz");  // DEBUG
    }
};

}  // namespace Speech

#endif  // _SPEECH_ACOUSTIC_MODEL_RESCORER_HH
