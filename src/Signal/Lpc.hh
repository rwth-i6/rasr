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
#ifndef _SIGNAL_LPC_HH
#define _SIGNAL_LPC_HH

#include <Core/Types.hh>

#include <Core/Parameter.hh>

#include <Flow/Vector.hh>

#include "ArxEstimator.hh"
#include "Node.hh"

namespace Signal {
class LinearPredictionCodingNode : public SleeveNode, public ArxEstimator {
private:
    static Core::ParameterInt paramOrder_B;
    static Core::ParameterInt paramOrder_A;

    bool sendVector(const Flow::Timestamp&  time_stamp,
                    const f32&              estimation_error,
                    const std::vector<f32>& B_tilde,
                    const std::vector<f32>& A_tilde);
    bool sendLinearFilterParameter(const Flow::Timestamp&  time_stamp,
                                   const f32&              estimation_error,
                                   const std::vector<f32>& B_tilde,
                                   const std::vector<f32>& A_tilde,
                                   bool                    initialize);

public:
    static std::string filterName() {
        return "signal-lpc";
    }
    LinearPredictionCodingNode(const Core::Configuration& c);
    virtual ~LinearPredictionCodingNode() {}

    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool configure();

    virtual Flow::PortId getInput(const std::string& name);
    virtual Flow::PortId getOutput(const std::string& name);

    virtual bool work(Flow::PortId p);
};

}  // namespace Signal

#endif  //_SIGNAL_LPC_HH
