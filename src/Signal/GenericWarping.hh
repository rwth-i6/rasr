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
#ifndef _SIGNAL_GENERIC_WARPING_HH
#define _SIGNAL_GENERIC_WARPING_HH

#include <Flow/StringExpressionNode.hh>
#include <Math/AnalyticFunctionFactory.hh>
#include "Warping.hh"

namespace Signal {

/** Warping node with arbitrary warping function
 *  Warping function can be any kind supported by Math::AnalyticFunctionFactory.
 *  Warping function can contain variables refering to one of the input ports
 *  (see Flow::StringExpressionNode.).
 */
class GenericWarpingNode : public WarpingNode, public Flow::StringExpressionNode {
private:
    static const Core::ParameterString paramWarpingFunction;

private:
    Warping warping_;

private:
    void updateWarping();

protected:
    virtual void apply(const Flow::Vector<f32>& in, std::vector<f32>& out);

public:
    static std::string filterName() {
        return "signal-warping";
    }
    GenericWarpingNode(const Core::Configuration& c);
    virtual ~GenericWarpingNode();

    Flow::PortId getInput(const std::string& name) {
        return name.empty() ? WarpingNode::getInput(name) : StringExpressionNode::getInput(name);
    }
    virtual bool configure();
    virtual bool setParameter(const std::string& name, const std::string& value);
};

}  // namespace Signal

#endif  //_SIGNAL_GENERIC_WARPING_HH
