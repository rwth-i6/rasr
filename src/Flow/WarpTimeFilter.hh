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
#ifndef _WARP_TIME_FILTER_HH
#define _WARP_TIME_FILTER_HH

#include "Node.hh"
#include "Vector.hh"

namespace Flow {

    /**
     *  Makes the timeframes of the incoming packets consecutive, and logs a warping map which describes the applied mapping.
     *  Input:
     *    default port: filtered stream
     *  Output:
     *    default port: warped stream
     */
    class WarpTimeFilterNode : public Flow::Node {
    private:
        Flow::Time currentTime_;
        static const Core::ParameterFloat paramStartTime;
        std::vector< std::pair<Flow::Time, Flow::Time> > warping_;
    public:
        WarpTimeFilterNode(const Core::Configuration &);
        static std::string filterName() { return "warp-time"; }

        virtual Flow::PortId getInput(const std::string &name) {
            return 0; }
        virtual Flow::PortId getOutput(const std::string &name) {
            return 0; }

        virtual bool configure();
        virtual bool work(Flow::PortId out);
        virtual bool setParameter(const std::string& name, const std::string& value);
    };

} // namespace Speech

#endif // _FLOW_SEQUENCE_FILTER_HH
